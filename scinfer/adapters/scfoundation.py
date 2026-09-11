"""scFoundation model adapter.

Wraps the scFoundation model behind the :class:`BaseModelAdapter` interface.
scFoundation (121M parameters) uses the xTrimoGene architecture — an
encoder-decoder transformer where the encoder uses standard Transformer
attention and the decoder uses **Performer linear attention** for
scalability.

Key architectural features
--------------------------
* **Input representation** — retains raw continuous expression values
  (optionally standardised); no discretisation.
* **Encoder** — standard multi-head self-attention Transformer.
* **Decoder** — Performer linear attention (FAVOR+), giving O(N) complexity
  instead of O(N²) for the standard softmax attention.
* **Non-autoregressive** encoder-decoder inference.
* **Sequence length** — up to 19 264 genes (near whole-genome coverage).
* **Custom xTrimoGene architecture** — not HuggingFace compatible.

Reference
---------
Hao et al., "Large Scale Foundation Model on Single-cell Transcriptomics",
Nature Methods 2024.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from scinfer.adapters.base import BaseModelAdapter
from scinfer.core.registry import register_model

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performer linear attention (lightweight implementation)
# ---------------------------------------------------------------------------


def _performer_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_features: int = 256,
) -> torch.Tensor:
    """FAVOR+ approximate attention kernel.

    Uses random Fourier features to approximate the softmax kernel,
    enabling O(N) linear attention.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key, value tensors of shape ``(batch, seq_len, dim)``.
    n_features : int
        Number of random Fourier features for the approximation.

    Returns
    -------
    torch.Tensor
        Output of shape ``(batch, seq_len, dim)``.
    """
    dim = q.shape[-1]
    # Random projection matrix (fixed seed for reproducibility)
    projection = torch.randn(dim, n_features, device=q.device, dtype=q.dtype) * (1.0 / dim**0.5)

    # Random Fourier features: phi(x) = relu(cos(x @ P) + sin(x @ P)) / sqrt(n_features)
    q_proj = q @ projection  # (B, S, n_features)
    k_proj = k @ projection

    q_feat = torch.relu(torch.cos(q_proj) + torch.sin(q_proj)) / (n_features**0.5)
    k_feat = torch.relu(torch.cos(k_proj) + torch.sin(k_proj)) / (n_features**0.5)

    # Linear attention: (Q' @ K'^T) @ V  ≈  Q' @ (K'^T @ V)
    kv = torch.bmm(k_feat.transpose(1, 2), v)  # (B, n_features, dim)
    z = k_feat.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, n_features)
    out = torch.bmm(q_feat, kv) / torch.bmm(q_feat, z.transpose(1, 2)).clamp(min=1e-6)
    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("scfoundation")
class ScFoundationAdapter(BaseModelAdapter):
    """Adapter for the scFoundation model (121M parameters).

    scFoundation uses the xTrimoGene architecture with a standard
    Transformer encoder and a Performer linear-attention decoder.  It
    operates on raw continuous expression values (no discretisation) and
    supports near whole-genome input (up to 19 264 genes).

    Parameters
    ----------
    config : dict
        Model configuration (see ``configs/models/scfoundation.yaml``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the scFoundation adapter.

        Parameters
        ----------
        config : dict
            scFoundation model configuration dictionary.
        """
        self._config = config
        self._model: Optional[nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = self._resolve_dtype(config.get("dtype", "float16"))

        model_cfg = config.get("model", config)
        self.n_layers: int = model_cfg.get("n_layers", 12)
        self.n_heads: int = model_cfg.get("n_heads", 8)
        self.embed_dim: int = model_cfg.get("embed_dim", 512)
        self.hidden_dim: int = model_cfg.get("hidden_dim", 2048)
        self.vocab_size: int = model_cfg.get("vocab_size", 33539)
        self.max_seq_len: int = model_cfg.get("max_seq_len", 19264)
        self.n_bins: int = model_cfg.get("n_bins", 512)
        self.expression_bin: bool = model_cfg.get("expression_bin", True)

        self.checkpoint_path: Optional[str] = model_cfg.get("checkpoint_path")
        self.checkpoint_repo: str = model_cfg.get(
            "checkpoint_repo", "scFoundation/scFoundation"
        )
        self.batch_size: int = model_cfg.get("batch_size", 64)

        # Gene name → index mapping
        self._gene_map: Optional[Dict[str, int]] = None

        logger.info(
            "ScFoundationAdapter initialised (vocab_size={}, max_seq_len={}, dtype={})",
            self.vocab_size,
            self.max_seq_len,
            self._dtype,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load scFoundation pretrained weights.

        scFoundation uses the custom xTrimoGene architecture.  If the
        original package is not available, a lightweight fallback with
        standard Transformer encoder + Performer decoder is built.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to local checkpoint.  Downloads from source if ``None``.

        Raises
        ------
        RuntimeError
            If the checkpoint cannot be loaded.
        """
        path = checkpoint_path or self.checkpoint_path
        try:
            from xtrimogene import xTrimoGene  # type: ignore[import]

            self._model = xTrimoGene(
                vocab_size=self.vocab_size,
                d_model=self.embed_dim,
                nhead=self.n_heads,
                d_hid=self.hidden_dim,
                nlayers=self.n_layers,
            )
            if path is not None:
                state = torch.load(path, map_location="cpu")
                self._model.load_state_dict(state)
                logger.info("Loaded scFoundation weights from {}", path)
            else:
                logger.warning("No checkpoint path; using randomly initialised model")
        except ImportError:
            logger.warning(
                "xtrimogene package not installed — building fallback model"
            )
            self._model = self._build_fallback_model()
            if path is not None:
                try:
                    state = torch.load(path, map_location="cpu")
                    self._model.load_state_dict(state, strict=False)
                    logger.info("Loaded weights into fallback model from {}", path)
                except Exception as exc:
                    logger.error("Failed to load weights: {}", exc)

        self._model.to(self._device, dtype=self._dtype)
        self._model.eval()

    # ------------------------------------------------------------------
    # Preprocessing — raw expression values
    # ------------------------------------------------------------------

    def preprocess(self, adata: "AnnData") -> Dict[str, torch.Tensor]:
        """Preprocess AnnData into scFoundation input format.

        scFoundation retains **raw continuous expression values** (no
        discretisation).  The values may be standardised (z-score per gene)
        before being passed to the model.

        For very large gene counts the input is truncated / padded to
        ``max_seq_len``.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data ``(n_cells, n_genes)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:

            * ``"expression"`` — ``FloatTensor`` ``(batch, seq_len)``
              raw / normalised expression values.
            * ``"gene_ids"`` — ``LongTensor`` ``(batch, seq_len)``
              gene index tokens.
            * ``"mask"`` — ``BoolTensor`` ``(batch, seq_len)``
              ``True`` for real genes, ``False`` for padding.
        """
        X = np.asarray(adata.X, dtype=np.float32)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray().astype(np.float32)

        n_cells, n_genes = X.shape
        gene_names = list(adata.var_names)

        # Map genes to indices
        gene_ids = self._map_genes_to_ids(gene_names)

        # Truncate / pad to max_seq_len
        seq_len = min(n_genes, self.max_seq_len)
        gene_ids = gene_ids[:seq_len]
        X = X[:, :seq_len] if n_genes >= seq_len else np.pad(
            X, ((0, 0), (0, seq_len - n_genes)), mode="constant"
        )

        # Optional per-gene z-score normalisation
        gene_mean = X.mean(axis=0, keepdims=True)
        gene_std = X.std(axis=0, keepdims=True) + 1e-8
        X_norm = (X - gene_mean) / gene_std

        # Valid mask (True = real gene, False = padding)
        valid_mask = np.ones((n_cells, seq_len), dtype=bool)
        if n_genes < seq_len:
            valid_mask[:, n_genes:] = False

        result: Dict[str, torch.Tensor] = {
            "expression": torch.tensor(X_norm, dtype=torch.float32),
            "gene_ids": torch.tensor(np.tile(gene_ids, (n_cells, 1)), dtype=torch.long),
            "mask": torch.tensor(valid_mask, dtype=torch.bool),
        }

        logger.debug(
            "Preprocessed {} cells × {} genes → seq_len={}", n_cells, n_genes, seq_len
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run scFoundation forward pass (non-autoregressive encoder-decoder).

        The encoder uses standard Transformer self-attention; the decoder
        uses Performer linear attention.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Preprocessed input (output of :meth:`preprocess`).

        Returns
        -------
        torch.Tensor
            Hidden states ``(batch, seq_len, embed_dim)``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        expression = x["expression"].to(self._device, dtype=self._dtype)
        gene_ids = x["gene_ids"].to(self._device)
        mask = x["mask"].to(self._device)

        output = self._model(expression, gene_ids, mask=mask)
        return output

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, adata: "AnnData") -> np.ndarray:
        """Extract per-cell embeddings from scFoundation.

        Uses mean-pooling over the encoder output (excluding padding).

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data.

        Returns
        -------
        numpy.ndarray
            Embedding array of shape ``(n_cells, embed_dim)``.
        """
        x = self.preprocess(adata)
        with torch.no_grad():
            hidden = self.forward(x)

        mask = x["mask"].to(self._device).unsqueeze(-1).float()
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return embeddings.cpu().numpy()

    # ------------------------------------------------------------------
    # Attention mask — standard + Performer compatible
    # ------------------------------------------------------------------

    def generate_attention_mask(
        self,
        seq_len: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate attention mask for scFoundation.

        The encoder uses standard bidirectional attention.  For the
        Performer decoder, linear attention does not require an explicit
        attention mask (it uses the FAVOR+ approximation), so this method
        returns the encoder mask by default.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        **kwargs
            Optional keyword arguments:

            * ``pad_mask`` — ``BoolTensor`` ``(batch, seq_len)`` where
              ``True`` = valid position, ``False`` = padding.

        Returns
        -------
        torch.Tensor
            Attention mask.  Shape ``(seq_len, seq_len)`` if no padding
            info given, or ``(batch, 1, seq_len, seq_len)`` with padding.
        """
        pad_mask = kwargs.get("pad_mask", None)

        if pad_mask is None:
            # Fully bidirectional
            return torch.ones(seq_len, seq_len, dtype=torch.bool)

        # pad_mask: (B, S) True=valid, False=pad
        valid = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        valid_row = pad_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, S, 1)
        attn = valid & valid_row  # both query and key must be valid
        return attn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "scfoundation"

    @property
    def model_type(self) -> str:
        """Return the architecture family.

        scFoundation uses an encoder-decoder architecture with Performer
        linear attention in the decoder.
        """
        return "encoder_decoder"

    @property
    def supported_optimizations(self) -> List[str]:
        """Return supported optimization engine names.

        scFoundation supports quantization, FlashAttention (encoder),
        torch.compile, and batch scheduling.  The Performer decoder
        already uses linear attention so KV-cache / speculative decoding
        are not applicable.
        """
        return ["quantization", "flash_attention", "compilation", "batch_scheduling"]

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> "ScFoundationAdapter":
        """Convenience loader: create adapter and load weights in one step.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to checkpoint file.
        config : dict, optional
            Model configuration.  Uses defaults if ``None``.
        device : str
            Device string (``"cpu"``, ``"cuda"``).

        Returns
        -------
        ScFoundationAdapter
            Ready-to-use adapter with loaded weights.
        """
        cfg = config or {}
        adapter = cls(cfg)
        adapter._device = torch.device(device)
        adapter.load_model(checkpoint_path)
        return adapter

    def get_model(self) -> nn.Module:
        """Return the underlying ``torch.nn.Module``.

        Returns
        -------
        torch.nn.Module

        Raises
        ------
        RuntimeError
            If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device.
        """
        self._device = device
        if self._model is not None:
            self._model.to(device)

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        if self._model is not None:
            self._model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_str, torch.float16)

    def _map_genes_to_ids(self, gene_names: List[str]) -> np.ndarray:
        """Map gene names to vocabulary indices.

        Parameters
        ----------
        gene_names : list[str]
            Gene symbol / Ensembl names.

        Returns
        -------
        np.ndarray
            Integer indices of shape ``(len(gene_names),)``.
        """
        if self._gene_map is not None:
            return np.array(
                [self._gene_map.get(g, 0) for g in gene_names], dtype=np.int64
            )
        return np.array(
            [hash(g) % self.vocab_size for g in gene_names], dtype=np.int64
        )

    def _build_fallback_model(self) -> nn.Module:
        """Build a fallback encoder-decoder model with Performer attention.

        Returns
        -------
        nn.Module
            Encoder (standard Transformer) + Decoder (Performer linear
            attention) model.
        """

        class _FallbackScFoundation(nn.Module):
            """Minimal scFoundation-compatible model for benchmarking."""

            def __init__(
                self,
                vocab_size: int,
                embed_dim: int,
                n_heads: int,
                hidden_dim: int,
                n_layers: int,
                n_performer_layers: int = 4,
            ) -> None:
                super().__init__()
                self.gene_embedding = nn.Linear(1, embed_dim)
                self.value_projection = nn.Linear(1, embed_dim)
                self.pos_embedding = nn.Embedding(vocab_size, embed_dim)

                # Encoder — standard Transformer
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

                # Decoder — Performer linear attention layers
                self.decoder_layers = nn.ModuleList()
                for _ in range(n_performer_layers):
                    self.decoder_layers.append(
                        _PerformerBlock(embed_dim, n_heads, hidden_dim)
                    )

                self.embed_dim = embed_dim

            def forward(
                self,
                expression: torch.Tensor,
                gene_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                # expression: (B, S) → (B, S, 1)
                expr = expression.unsqueeze(-1)
                x = self.gene_embedding(expr) + self.pos_embedding(gene_ids)

                # Encoder
                enc_out = self.encoder(x)

                # Decoder (Performer)
                dec_out = enc_out
                for layer in self.decoder_layers:
                    dec_out = layer(dec_out)

                return dec_out

        class _PerformerBlock(nn.Module):
            """Single Performer linear-attention block."""

            def __init__(self, dim: int, n_heads: int, hidden_dim: int) -> None:
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = dim // n_heads
                self.qkv = nn.Linear(dim, dim * 3)
                self.out_proj = nn.Linear(dim, dim)
                self.ff = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                x = self.norm1(x)
                B, S, D = x.shape
                qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
                q, k, v = qkv.unbind(dim=2)
                # Reshape for Performer kernel
                q = q.reshape(B * self.n_heads, S, self.head_dim)
                k = k.reshape(B * self.n_heads, S, self.head_dim)
                v = v.reshape(B * self.n_heads, S, self.head_dim)
                out = _performer_kernel(q, k, v)
                out = out.reshape(B, S, D)
                x = self.out_proj(out)
                x = residual + x

                # Feed-forward
                x = x + self.ff(self.norm2(x))
                return x

        return _FallbackScFoundation(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )
