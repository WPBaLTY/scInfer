"""scGPT model adapter.

Wraps the scGPT foundation model behind the :class:`BaseModelAdapter`
interface.  scGPT is a 53M-parameter encoder-decoder hybrid transformer
trained on single-cell transcriptomics data using a multi-task objective
including gene expression reconstruction, gene masking, and contrastive
learning.

Key architectural features
--------------------------
* **Value Binning** — continuous expression values are discretised into bin
  encodings; each gene is represented by a *Gene ID embedding* plus a
  *Value bin embedding*.
* **Dynamic attention mask** — known genes attend to each other
  (bidirectional); unknown genes can only attend to known genes
  (unidirectional).  This is *not* a standard causal mask.
* **Iterative autoregressive generation** — at each round the model predicts
  a batch of high-confidence genes (top 1/K), moves them from "unknown" to
  "known", updates the mask, and repeats for 10–20 rounds.
* **Max sequence length** — ~1200 genes (a subset of ~20 000 human genes).
* **FlashAttention-1** integrated.
* **Custom Transformer** — not standard HuggingFace format.

Reference
---------
Cui et al., "scGPT: toward building a foundation model for single-cell
multi-omics using generative AI", Nature Methods 2024.
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

# Optional heavy dependencies — the adapter must remain importable even
# when the model's original codebase is not installed.
try:
    import anndata  # noqa: F401
except ImportError:
    anndata = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_binning(
    values: np.ndarray,
    n_bins: int = 512,
    bin_method: str = "equal_width",
) -> np.ndarray:
    """Discretise continuous expression values into bin indices.

    Parameters
    ----------
    values : np.ndarray
        Raw or normalised expression matrix ``(n_cells, n_genes)``.
    n_bins : int
        Number of discrete bins.
    bin_method : str
        ``"equal_width"`` for uniform-width binning, ``"equal_freq"`` for
        quantile-based binning.

    Returns
    -------
    np.ndarray
        Integer bin indices of same shape as *values*, dtype ``int64``.
    """
    if bin_method == "equal_width":
        vmin = values.min()
        vmax = values.max()
        if vmax == vmin:
            return np.zeros_like(values, dtype=np.int64)
        bin_idx = ((values - vmin) / (vmax - vmin) * (n_bins - 1)).astype(np.int64)
    elif bin_method == "equal_freq":
        bin_idx = np.zeros_like(values, dtype=np.int64)
        for i in range(values.shape[0]):
            row = values[i]
            sorted_idx = np.argsort(row)
            ranks = np.empty_like(sorted_idx)
            ranks[sorted_idx] = np.arange(len(row))
            bin_idx[i] = (ranks / max(len(row) - 1, 1) * (n_bins - 1)).astype(np.int64)
    else:
        raise ValueError(f"Unknown bin_method: {bin_method}")
    return bin_idx


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("scgpt")
class ScGPTAdapter(BaseModelAdapter):
    """Adapter for the scGPT foundation model (53M parameters).

    scGPT uses a hybrid encoder-decoder architecture with Value Binning
    input encoding and a custom dynamic attention mask that distinguishes
    between *known* (observed) and *unknown* (to-be-predicted) genes.

    Parameters
    ----------
    config : dict
        Model configuration (see ``configs/models/scgpt.yaml``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the scGPT adapter.

        Parameters
        ----------
        config : dict
            scGPT model configuration dictionary.
        """
        self._config = config
        self._model: Optional[nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = self._resolve_dtype(config.get("dtype", "float16"))

        # Architecture hyper-parameters
        model_cfg = config.get("model", config)
        self.n_layers: int = model_cfg.get("n_layers", 12)
        self.n_heads: int = model_cfg.get("n_heads", 8)
        self.embed_dim: int = model_cfg.get("embed_dim", 512)
        self.hidden_dim: int = model_cfg.get("hidden_dim", 2048)
        self.vocab_size: int = model_cfg.get("vocab_size", 60697)
        self.max_seq_len: int = model_cfg.get("max_seq_len", 1536)
        self.use_batch_labels: bool = model_cfg.get("use_batch_labels", True)
        self.n_batch_labels: int = model_cfg.get("n_batch_labels", 8)

        # Value binning
        self.n_bins: int = model_cfg.get("n_bins", 512)

        # Checkpoint location
        self.checkpoint_path: Optional[str] = model_cfg.get("checkpoint_path")
        self.checkpoint_repo: str = model_cfg.get("checkpoint_repo", "scGPT/scGPT")

        # Inference
        self.batch_size: int = model_cfg.get("batch_size", 64)

        # Gene vocabulary mapping (gene symbol → token id)
        self._gene_vocab: Optional[Dict[str, int]] = None

        logger.info(
            "ScGPTAdapter initialised (vocab_size={}, max_seq_len={}, dtype={})",
            self.vocab_size,
            self.max_seq_len,
            self._dtype,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load scGPT pretrained weights.

        scGPT uses a custom Transformer architecture (not HuggingFace).
        The adapter attempts to import the original ``scgpt`` package; if
        unavailable it builds a lightweight compatible architecture so that
        the rest of the pipeline can still run (e.g. for benchmarking).

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to local checkpoint.  Downloads from HuggingFace if ``None``.

        Raises
        ------
        RuntimeError
            If the checkpoint cannot be loaded.
        """
        path = checkpoint_path or self.checkpoint_path
        try:
            # Attempt to use the official scGPT codebase
            from scgpt.model import TransformerModel  # type: ignore[import]

            self._model = TransformerModel(
                ntoken=self.vocab_size,
                d_model=self.embed_dim,
                nhead=self.n_heads,
                d_hid=self.hidden_dim,
                nlayers=self.n_layers,
                n_cls=1,
                use_batch_labels=self.use_batch_labels,
            )
            if path is not None:
                state = torch.load(path, map_location="cpu")
                self._model.load_state_dict(state)
                logger.info("Loaded scGPT weights from {}", path)
            else:
                logger.warning(
                    "No checkpoint path provided; using randomly initialised scGPT model"
                )
        except ImportError:
            logger.warning(
                "scgpt package not installed — building lightweight fallback model"
            )
            self._model = self._build_fallback_model()
            if path is not None:
                try:
                    state = torch.load(path, map_location="cpu")
                    self._model.load_state_dict(state, strict=False)
                    logger.info("Loaded weights into fallback model from {}", path)
                except Exception as exc:
                    logger.error("Failed to load weights into fallback model: {}", exc)

        self._model.to(self._device, dtype=self._dtype)
        self._model.eval()

    # ------------------------------------------------------------------
    # Preprocessing — Value Binning
    # ------------------------------------------------------------------

    def preprocess(self, adata: "AnnData") -> Dict[str, torch.Tensor]:
        """Preprocess AnnData into scGPT input format using Value Binning.

        Each gene is encoded as a pair of *(Gene ID token, Value bin ID)*.
        The continuous expression values are discretised into bins and each
        bin index is embedded separately.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data ``(n_cells, n_genes)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:

            * ``"gene_ids"`` — ``LongTensor`` of gene token IDs
              ``(batch, seq_len)``
            * ``"values"`` — ``LongTensor`` of bin indices
              ``(batch, seq_len)``
            * ``"batch_labels"`` — ``LongTensor`` of batch IDs (if enabled)
        """
        X = np.asarray(adata.X, dtype=np.float32)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray().astype(np.float32)

        n_cells, n_genes = X.shape
        gene_names = list(adata.var_names)

        # Map gene names → token ids
        gene_ids = self._map_genes_to_ids(gene_names)

        # Pad / truncate to max_seq_len
        seq_len = min(n_genes, self.max_seq_len)
        gene_ids = gene_ids[:seq_len]
        X = X[:, :seq_len] if n_genes >= seq_len else np.pad(
            X, ((0, 0), (0, seq_len - n_genes)), mode="constant"
        )

        # Value binning
        bin_ids = _value_binning(X, n_bins=self.n_bins)

        result: Dict[str, torch.Tensor] = {
            "gene_ids": torch.tensor(np.tile(gene_ids, (n_cells, 1)), dtype=torch.long),
            "values": torch.tensor(bin_ids, dtype=torch.long),
        }

        if self.use_batch_labels:
            # Default batch label = 0 (single-batch assumption)
            result["batch_labels"] = torch.zeros(n_cells, dtype=torch.long)

        logger.debug(
            "Preprocessed {} cells × {} genes → seq_len={}", n_cells, n_genes, seq_len
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run scGPT forward pass.

        Supports both standard encoding and iterative autoregressive
        generation mode.

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

        gene_ids = x["gene_ids"].to(self._device)
        values = x["values"].to(self._device)

        # Standard forward
        src_key_padding_mask = gene_ids.eq(0)  # pad token = 0
        output = self._model(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
        )
        return output

    def forward_autoregressive(
        self,
        x: Dict[str, torch.Tensor],
        n_known: int,
        n_rounds: int = 15,
        top_k_fraction: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Iterative autoregressive generation.

        At each round the model predicts high-confidence genes (top 1/K),
        moves them from *unknown* to *known*, and updates the attention
        mask accordingly.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Preprocessed input.
        n_known : int
            Number of initially known genes.
        n_rounds : int
            Number of iterative generation rounds (10–20 recommended).
        top_k_fraction : float
            Fraction of remaining unknown genes to predict per round.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(hidden_states, predicted_values)`` — final hidden states and
            predicted expression values.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gene_ids = x["gene_ids"].to(self._device)
        values = x["values"].to(self._device)
        batch_size, seq_len = gene_ids.shape

        # Track known / unknown positions
        known_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self._device)
        known_mask[:, :n_known] = True

        predicted_values = values.clone()

        for round_idx in range(n_rounds):
            n_unknown = (~known_mask).sum(dim=-1)
            if (n_unknown == 0).all():
                logger.debug("All genes known after {} rounds", round_idx)
                break

            # Build dynamic attention mask
            attn_mask = self.generate_attention_mask(
                seq_len, known_mask=known_mask, batch_size=batch_size
            ).to(self._device)

            # Forward with current mask
            hidden = self._model(
                gene_ids,
                predicted_values,
                attn_mask=attn_mask,
                src_key_padding_mask=~known_mask,
            )

            # Predict values for unknown positions
            unknown_indices = (~known_mask).nonzero(as_tuple=False)
            if unknown_indices.numel() == 0:
                break

            # Score unknown positions by prediction confidence
            n_predict = max(1, int(top_k_fraction * n_unknown.float().mean().item()))
            # Use L2 norm of hidden states as a simple confidence proxy
            scores = hidden[~known_mask].norm(dim=-1)
            top_idx = scores.topk(min(n_predict, scores.numel())).indices

            # Move top predictions from unknown → known
            batch_rows = unknown_indices[top_idx, 0]
            col_idx = unknown_indices[top_idx, 1]
            known_mask[batch_rows, col_idx] = True

            logger.debug(
                "Round {}/{}: predicted {} genes, known={}/{}",
                round_idx + 1,
                n_rounds,
                n_predict,
                known_mask.sum().item(),
                batch_size * seq_len,
            )

        # Final forward with all known
        final_hidden = self._model(gene_ids, predicted_values)
        return final_hidden, predicted_values

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, adata: "AnnData") -> np.ndarray:
        """Extract per-cell embeddings from scGPT.

        Uses mean-pooling over the hidden states (excluding padding).

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

        # Mean pooling over non-padding positions
        gene_ids = x["gene_ids"].to(self._device)
        mask = gene_ids.ne(0).unsqueeze(-1).float()
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return embeddings.cpu().numpy()

    # ------------------------------------------------------------------
    # Attention mask — dynamic known / unknown gene mask
    # ------------------------------------------------------------------

    def generate_attention_mask(
        self,
        seq_len: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate scGPT's dynamic known/unknown gene attention mask.

        Known genes attend to all other known genes (bidirectional).
        Unknown genes can only attend to known genes (unidirectional).
        This is **not** a standard causal mask.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        **kwargs
            Optional keyword arguments:

            * ``known_mask`` — ``BoolTensor`` of shape ``(batch, seq_len)``
              indicating which gene positions are known.
            * ``batch_size`` — int, required when ``known_mask`` is given.

        Returns
        -------
        torch.Tensor
            Attention mask.  Shape ``(seq_len, seq_len)`` if no
            ``known_mask`` is provided (returns all-ones = fully visible),
            or ``(batch, seq_len, seq_len)`` when ``known_mask`` is given.
            ``True`` / ``0.0`` means *attend*; ``False`` / ``-inf`` means
            *block*.
        """
        known_mask = kwargs.get("known_mask", None)
        batch_size = kwargs.get("batch_size", None)

        if known_mask is None:
            # Fully bidirectional (all genes known)
            return torch.ones(seq_len, seq_len, dtype=torch.bool)

        if batch_size is None:
            batch_size = known_mask.shape[0]

        # known_mask: (batch, seq_len) — True = known
        # Rule 1: known → known  (bidirectional) ✓
        # Rule 2: known → unknown ✗  (known genes don't need to see unknown)
        # Rule 3: unknown → known  ✓
        # Rule 4: unknown → unknown ✗
        known_float = known_mask.float()  # (B, S)
        # attn[b, i, j] = known[i] & known[j] | ~known[i] & known[j]
        #               = known[j]  (any position can attend to known)
        # But known should NOT attend to unknown:
        # attn[b, i, j] = known[j] if known[i] else known[j]
        # Simplification: attn[b, i, j] = known[j]
        # However, unknown should not attend to unknown:
        # attn[b, i, j] = known[j]  (both known and unknown can see known)
        # But unknown cannot see unknown → only known[j] matters
        # And known cannot see unknown → need known[j] AND known[i] OR known[j] AND ~known[i]
        # = known[j] in all cases where known[i] is True
        # For unknown[i]: can only see known[j]
        # So: attn[i,j] = known[j]  for all i
        # Wait — known genes should see other known genes bidirectionally:
        # known[i] sees known[j] ✓
        # known[i] sees unknown[j] ✗
        # unknown[i] sees known[j] ✓
        # unknown[i] sees unknown[j] ✗
        # → attn[i,j] = known[j]  for all i (both known and unknown query only see known keys)
        attn = known_float.unsqueeze(1).expand(-1, seq_len, seq_len)
        # attn[b, :, j] = known[b, j] — every row gets the same known pattern
        return attn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "scgpt"

    @property
    def model_type(self) -> str:
        """Return the architecture family.

        scGPT is a hybrid: encoder with autoregressive generation capability.
        """
        return "hybrid"

    @property
    def supported_optimizations(self) -> List[str]:
        """Return supported optimization engine names.

        scGPT supports FlashAttention-1, quantization, torch.compile, and
        batch scheduling.
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
    ) -> "ScGPTAdapter":
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
        ScGPTAdapter
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
        """Convert a string dtype to ``torch.dtype``."""
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_str, torch.float16)

    def _map_genes_to_ids(self, gene_names: List[str]) -> np.ndarray:
        """Map gene symbol names to vocabulary token IDs.

        If no vocabulary is loaded, uses a hash-based fallback so the
        pipeline remains functional for testing.

        Parameters
        ----------
        gene_names : list[str]
            Gene symbol names.

        Returns
        -------
        np.ndarray
            Token ID array of shape ``(len(gene_names),)``, dtype ``int64``.
        """
        if self._gene_vocab is not None:
            return np.array(
                [self._gene_vocab.get(g, 0) for g in gene_names], dtype=np.int64
            )
        # Hash-based fallback (deterministic, keeps values in vocab range)
        return np.array(
            [hash(g) % (self.vocab_size - 1) + 1 for g in gene_names], dtype=np.int64
        )

    def _build_fallback_model(self) -> nn.Module:
        """Build a lightweight fallback model when scgpt is not installed.

        Returns
        -------
        nn.Module
            A simple transformer encoder that mimics scGPT's interface.
        """

        class _FallbackScGPT(nn.Module):
            """Minimal scGPT-compatible model for benchmarking."""

            def __init__(
                self,
                vocab_size: int,
                embed_dim: int,
                n_heads: int,
                hidden_dim: int,
                n_layers: int,
                n_bins: int = 512,
            ) -> None:
                super().__init__()
                self.gene_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.value_embedding = nn.Embedding(n_bins, embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.embed_dim = embed_dim

            def forward(
                self,
                gene_ids: torch.Tensor,
                values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                x = self.gene_embedding(gene_ids) + self.value_embedding(values)
                # Convert boolean attn_mask to float for transformer
                float_attn_mask = None
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        float_attn_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                        float_attn_mask.masked_fill_(~attn_mask, float("-inf"))
                    else:
                        float_attn_mask = attn_mask
                return self.encoder(
                    x,
                    mask=float_attn_mask,
                    src_key_padding_mask=src_key_padding_mask,
                )

        return _FallbackScGPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_bins=self.n_bins,
        )
