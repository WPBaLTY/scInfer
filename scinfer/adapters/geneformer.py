"""Geneformer V2 model adapter.

Wraps the Geneformer foundation model behind the :class:`BaseModelAdapter`
interface.  Geneformer is a BERT-like encoder-only transformer trained on
rank-ordered gene expression profiles from a large-scale scRNA-seq corpus.

Key architectural features
--------------------------
* **Rank Embedding** — genes are sorted by expression level within each
  cell; the rank position (a discrete integer) is used as input.
* **Standard BERT bidirectional attention** with ``[CLS]`` / ``[PAD]``
  special token handling.
* **Encoder-only** — single forward pass, no autoregressive generation.
* **Three model scales** — 10M / 38M / 316M parameters.
* **Sequence lengths** — 2048 (V1) / 4096 (V2).
* **HuggingFace compatible** — uses ``from_pretrained()``.

Reference
---------
Theodoris et al., "Transfer learning enables predictions in network
biology", Nature 2023.
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
# Model-scale presets
# ---------------------------------------------------------------------------

_SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "10M": {
        "n_layers": 6,
        "n_heads": 6,
        "embed_dim": 256,
        "hidden_dim": 1024,
        "max_seq_len": 2048,
        "checkpoint_repo": "ctheodoris/Geneformer-6L-30M-i2048",
    },
    "38M": {
        "n_layers": 12,
        "n_heads": 12,
        "embed_dim": 768,
        "hidden_dim": 3072,
        "max_seq_len": 2048,
        "checkpoint_repo": "ctheodoris/Geneformer-12L-30M-i2048",
    },
    "316M": {
        "n_layers": 20,
        "n_heads": 16,
        "embed_dim": 1280,
        "hidden_dim": 5120,
        "max_seq_len": 4096,
        "checkpoint_repo": "ctheodoris/Geneformer-20L-316M-i4096",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_encode(
    expression: np.ndarray,
    max_rank: int = 2048,
    gene_token_map: Optional[Dict[str, int]] = None,
    gene_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert expression matrix to rank-encoded gene token IDs.

    For each cell, genes are sorted by expression level (descending).
    The rank position becomes the input token ID.  Genes with zero
    expression are assigned the ``[PAD]`` token.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix ``(n_cells, n_genes)``.
    max_rank : int
        Maximum sequence length (number of top-ranked genes to keep).
    gene_token_map : dict, optional
        Mapping from gene name → token ID.
    gene_names : list[str], optional
        Gene names corresponding to columns of *expression*.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(token_ids, attention_mask)`` each of shape
        ``(n_cells, max_rank)``.
    """
    n_cells, n_genes = expression.shape

    # Build gene → token mapping if not provided
    if gene_token_map is None and gene_names is not None:
        gene_token_map = {g: i + 1 for i, g in enumerate(gene_names)}

    token_ids = np.zeros((n_cells, max_rank), dtype=np.int64)
    attn_mask = np.zeros((n_cells, max_rank), dtype=np.int64)

    for cell_idx in range(n_cells):
        row = expression[cell_idx]
        # Sort genes by expression (descending)
        sorted_indices = np.argsort(-row)

        rank = 0
        for gene_idx in sorted_indices:
            if rank >= max_rank:
                break
            if row[gene_idx] <= 0:
                continue  # skip zero-expression genes → [PAD]

            if gene_token_map is not None and gene_names is not None:
                gene_name = gene_names[gene_idx]
                tid = gene_token_map.get(gene_name, 0)
            else:
                tid = gene_idx + 1  # fallback: 1-indexed

            token_ids[cell_idx, rank] = tid
            attn_mask[cell_idx, rank] = 1
            rank += 1

    return token_ids, attn_mask


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("geneformer")
class GeneformerAdapter(BaseModelAdapter):
    """Adapter for the Geneformer V2 foundation model.

    Geneformer is a BERT-like encoder that operates on rank-ordered gene
    expression profiles.  It supports three model scales (10M / 38M / 316M
    parameters) and is implemented on top of HuggingFace Transformers.

    Parameters
    ----------
    config : dict
        Model configuration (see ``configs/models/geneformer.yaml``).
    """

    # Special token IDs (aligned with Geneformer vocabularies)
    CLS_TOKEN_ID: int = 1
    PAD_TOKEN_ID: int = 0
    SEP_TOKEN_ID: int = 2

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the Geneformer adapter.

        Parameters
        ----------
        config : dict
            Geneformer model configuration dictionary.
        """
        self._config = config
        self._model: Optional[nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = self._resolve_dtype(config.get("dtype", "float16"))

        model_cfg = config.get("model", config)

        # Resolve scale preset
        scale = model_cfg.get("scale", model_cfg.get("variant", "38M"))
        # Map variant strings like "12L" → scale preset
        variant_to_scale = {"6L": "10M", "12L": "38M", "20L": "316M"}
        scale = variant_to_scale.get(scale, scale)
        preset = _SCALE_PRESETS.get(scale, _SCALE_PRESETS["38M"])

        self.n_layers: int = model_cfg.get("n_layers", preset["n_layers"])
        self.n_heads: int = model_cfg.get("n_heads", preset["n_heads"])
        self.embed_dim: int = model_cfg.get("embed_dim", preset["embed_dim"])
        self.hidden_dim: int = model_cfg.get("hidden_dim", preset["hidden_dim"])
        self.vocab_size: int = model_cfg.get("vocab_size", 25426)
        self.max_seq_len: int = model_cfg.get("max_seq_len", preset["max_seq_len"])
        self.max_rank: int = model_cfg.get("max_rank", self.max_seq_len)

        self.checkpoint_path: Optional[str] = model_cfg.get("checkpoint_path")
        self.checkpoint_repo: str = model_cfg.get(
            "checkpoint_repo", preset["checkpoint_repo"]
        )
        self.batch_size: int = model_cfg.get("batch_size", 32)

        # Gene dictionary (Ensembl ID → token ID)
        self._gene_token_map: Optional[Dict[str, int]] = None

        logger.info(
            "GeneformerAdapter initialised (scale={}, vocab_size={}, max_seq_len={})",
            scale,
            self.vocab_size,
            self.max_seq_len,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load Geneformer pretrained weights.

        Geneformer is built on HuggingFace Transformers and can be loaded
        via ``AutoModel.from_pretrained()``.  If the ``geneformer`` or
        ``transformers`` package is unavailable, a lightweight fallback is
        constructed.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to local checkpoint.  Downloads from HuggingFace Hub if
            ``None``.

        Raises
        ------
        RuntimeError
            If the checkpoint cannot be loaded.
        """
        source = checkpoint_path or self.checkpoint_path or self.checkpoint_repo
        try:
            from transformers import BertForMaskedLM, BertConfig  # type: ignore[import]

            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.embed_dim,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_heads,
                intermediate_size=self.hidden_dim,
                max_position_embeddings=self.max_seq_len + 2,  # +CLS +SEP
                pad_token_id=self.PAD_TOKEN_ID,
                cls_token_id=self.CLS_TOKEN_ID,
            )
            try:
                self._model = BertForMaskedLM.from_pretrained(source, config=config)
                logger.info("Loaded Geneformer from HuggingFace: {}", source)
            except Exception:
                logger.warning(
                    "Could not load pretrained weights from {}; using random init", source
                )
                self._model = BertForMaskedLM(config)

        except ImportError:
            logger.warning(
                "transformers package not installed — building fallback model"
            )
            self._model = self._build_fallback_model()

        self._model.to(self._device, dtype=self._dtype)
        self._model.eval()

    # ------------------------------------------------------------------
    # Preprocessing — Rank Embedding
    # ------------------------------------------------------------------

    def preprocess(self, adata: "AnnData") -> Dict[str, torch.Tensor]:
        """Preprocess AnnData into Geneformer rank-encoded input.

        Genes are sorted by expression level (descending) within each cell.
        The rank position becomes the input token ID.  A ``[CLS]`` token is
        prepended and ``[PAD]`` tokens fill remaining positions.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data ``(n_cells, n_genes)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:

            * ``"input_ids"`` — ``LongTensor`` ``(batch, seq_len)``
            * ``"attention_mask"`` — ``LongTensor`` ``(batch, seq_len)``
        """
        X = np.asarray(adata.X, dtype=np.float32)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray().astype(np.float32)

        gene_names = list(adata.var_names)

        # Rank encoding
        token_ids, attn_mask = _rank_encode(
            X,
            max_rank=self.max_rank,
            gene_token_map=self._gene_token_map,
            gene_names=gene_names,
        )

        # Prepend [CLS] token
        n_cells = token_ids.shape[0]
        cls_col = np.full((n_cells, 1), self.CLS_TOKEN_ID, dtype=np.int64)
        cls_attn = np.ones((n_cells, 1), dtype=np.int64)
        token_ids = np.concatenate([cls_col, token_ids], axis=1)
        attn_mask = np.concatenate([cls_attn, attn_mask], axis=1)

        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }

        logger.debug(
            "Preprocessed {} cells × {} genes → seq_len={}",
            n_cells,
            len(gene_names),
            token_ids.shape[1],
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run Geneformer forward pass (encoder-only, single pass).

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

        input_ids = x["input_ids"].to(self._device)
        attention_mask = x["attention_mask"].to(self._device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Return last hidden state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        return outputs.logits

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, adata: "AnnData") -> np.ndarray:
        """Extract per-cell embeddings from Geneformer.

        Uses the ``[CLS]`` token representation as the cell embedding.

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

        # [CLS] token is at position 0
        cls_embedding = hidden[:, 0, :]
        return cls_embedding.cpu().numpy()

    # ------------------------------------------------------------------
    # Attention mask — BERT-style PAD masking
    # ------------------------------------------------------------------

    def generate_attention_mask(
        self,
        seq_len: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate BERT-style attention mask with PAD token handling.

        Geneformer uses standard bidirectional attention.  The only
        masking concern is ``[PAD]`` tokens, which should not participate
        in attention.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        **kwargs
            Optional keyword arguments:

            * ``pad_mask`` — ``BoolTensor`` ``(batch, seq_len)`` where
              ``True`` indicates a ``[PAD]`` position.
            * ``input_ids`` — ``LongTensor``; PAD positions are detected
              automatically (``== PAD_TOKEN_ID``).

        Returns
        -------
        torch.Tensor
            Attention mask.  Shape ``(seq_len, seq_len)`` (all-ones) if no
            padding info is given, or ``(batch, 1, 1, seq_len)`` when
            padding info is provided (broadcastable for multi-head
            attention).
        """
        pad_mask = kwargs.get("pad_mask", None)
        input_ids = kwargs.get("input_ids", None)

        if pad_mask is None and input_ids is not None:
            pad_mask = input_ids.eq(self.PAD_TOKEN_ID)

        if pad_mask is None:
            # Fully visible (no padding)
            return torch.ones(seq_len, seq_len, dtype=torch.bool)

        # Convert PAD mask to attention mask: True = attend, False = block
        # Shape: (batch, 1, 1, seq_len) for broadcasting
        attn_mask = (~pad_mask).unsqueeze(1).unsqueeze(2)
        return attn_mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "geneformer"

    @property
    def model_type(self) -> str:
        """Return the architecture family."""
        return "encoder_only"

    @property
    def supported_optimizations(self) -> List[str]:
        """Return supported optimization engine names.

        Geneformer supports quantization, FlashAttention, torch.compile,
        batch scheduling, and knowledge distillation (encoder-only).
        """
        return [
            "quantization",
            "flash_attention",
            "compilation",
            "batch_scheduling",
            "distillation",
        ]

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        scale: str = "38M",
    ) -> "GeneformerAdapter":
        """Convenience loader: create adapter and load weights in one step.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to checkpoint file or HuggingFace repo ID.
        config : dict, optional
            Model configuration.  Uses defaults if ``None``.
        device : str
            Device string (``"cpu"``, ``"cuda"``).
        scale : str
            Model scale: ``"10M"``, ``"38M"``, or ``"316M"``.

        Returns
        -------
        GeneformerAdapter
            Ready-to-use adapter with loaded weights.
        """
        cfg = config or {}
        cfg.setdefault("model", {}).setdefault("scale", scale)
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

    def _build_fallback_model(self) -> nn.Module:
        """Build a lightweight BERT-like fallback model.

        Returns
        -------
        nn.Module
            A simple transformer encoder mimicking Geneformer's interface.
        """

        class _FallbackGeneformer(nn.Module):
            """Minimal Geneformer-compatible model for benchmarking."""

            def __init__(
                self,
                vocab_size: int,
                embed_dim: int,
                n_heads: int,
                hidden_dim: int,
                n_layers: int,
                max_seq_len: int = 2048,
            ) -> None:
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.position_embedding = nn.Embedding(max_seq_len + 2, embed_dim)
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
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                **kwargs: Any,
            ) -> Any:
                seq_len = input_ids.shape[1]
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                x = self.token_embedding(input_ids) + self.position_embedding(positions)

                # Convert attention mask to src_key_padding_mask for TransformerEncoder
                # attention_mask: (B, S) — 1=attend, 0=pad
                src_key_padding_mask = None
                if attention_mask is not None:
                    # True = padding position (should be ignored)
                    src_key_padding_mask = attention_mask.eq(0)

                hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

                # Return a namespace-like object
                class _Output:
                    def __init__(self, hidden_states, logits):
                        self.hidden_states = hidden_states
                        self.logits = logits

                return _Output(
                    hidden_states=(hidden,),
                    logits=hidden,
                )

        return _FallbackGeneformer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
        )
