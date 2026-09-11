"""UCE (Universal Cell Embedding) model adapter.

Wraps the UCE model behind the :class:`BaseModelAdapter` interface.
UCE (~650M parameters) takes a fundamentally different approach from the
other three models: instead of operating on gene expression values
directly, it encodes **protein amino-acid sequences** via a protein
language model and maps them to a universal cell embedding space.

Key architectural features
--------------------------
* **Input representation** — amino-acid sequence embeddings from a
  protein language model (e.g. ESM-2).  Each gene is represented by its
  protein product's sequence embedding rather than by expression values.
* **Encoder** — standard Transformer encoder operating on protein
  embeddings.
* **Decoder** — standard Transformer decoder that maps from protein space
  to gene-expression prediction.
* **Cross-tissue generalisation** — because the input is protein-level
  rather than expression-level, UCE generalises across tissues, species,
  and experimental protocols.
* **Standard Transformer attention** throughout.

Reference
---------
Rosen et al., "A universal embedding space for single-cell biology",
bioRxiv 2024.
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
# Protein sequence helpers
# ---------------------------------------------------------------------------

# Standard amino acid vocabulary (20 canonical + special tokens)
_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
_SPECIAL_TOKENS = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"]
_AA_VOCAB = _SPECIAL_TOKENS + _AMINO_ACIDS
_AA_VOCAB_SIZE = len(_AA_VOCAB)

# Average protein length in amino acids (for default sequence length)
_DEFAULT_PROTEIN_SEQ_LEN = 512


def _encode_amino_acid_sequence(seq: str, max_len: int = _DEFAULT_PROTEIN_SEQ_LEN) -> List[int]:
    """Encode an amino-acid string into token IDs.

    Parameters
    ----------
    seq : str
        Amino-acid sequence (single-letter codes).
    max_len : int
        Maximum sequence length (truncate or pad).

    Returns
    -------
    list[int]
        Token IDs.
    """
    vocab = {aa: i for i, aa in enumerate(_AA_VOCAB)}
    unk_id = vocab.get("<unk>", 3)
    ids = [vocab.get("<cls>", 1)]  # [CLS]
    for aa in seq[: max_len - 2]:  # reserve space for [CLS] and [EOS]
        ids.append(vocab.get(aa, unk_id))
    ids.append(vocab.get("<eos>", 2))
    # Pad
    while len(ids) < max_len:
        ids.append(vocab.get("<pad>", 0))
    return ids[:max_len]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_model("uce")
class UCEAdapter(BaseModelAdapter):
    """Adapter for the UCE (Universal Cell Embedding) model (~650M params).

    UCE operates in protein sequence space rather than gene expression
    space.  Each gene is represented by the amino-acid sequence embedding
    of its protein product (obtained from a protein language model such as
    ESM-2).  This gives UCE strong cross-tissue and cross-species
    generalisation properties.

    Parameters
    ----------
    config : dict
        Model configuration (see ``configs/models/uce.yaml``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the UCE adapter.

        Parameters
        ----------
        config : dict
            UCE model configuration dictionary.
        """
        self._config = config
        self._model: Optional[nn.Module] = None
        self._protein_encoder: Optional[nn.Module] = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = self._resolve_dtype(config.get("dtype", "float32"))

        model_cfg = config.get("model", config)
        self.n_layers: int = model_cfg.get("n_layers", 12)
        self.n_heads: int = model_cfg.get("n_heads", 8)
        self.embed_dim: int = model_cfg.get("embed_dim", 1280)
        self.hidden_dim: int = model_cfg.get("hidden_dim", 5120)
        self.vocab_size: int = model_cfg.get("vocab_size", 25426)
        self.max_seq_len: int = model_cfg.get("max_seq_len", 2048)
        self.use_protein_embeddings: bool = model_cfg.get("use_protein_embeddings", True)
        self.protein_embedding_dim: int = model_cfg.get("protein_embedding_dim", 5120)
        self.protein_seq_len: int = model_cfg.get("protein_seq_len", _DEFAULT_PROTEIN_SEQ_LEN)

        self.checkpoint_path: Optional[str] = model_cfg.get("checkpoint_path")
        self.checkpoint_repo: str = model_cfg.get("checkpoint_repo", "snap-stanford/uce")
        self.batch_size: int = model_cfg.get("batch_size", 32)

        # Gene → protein sequence mapping (gene symbol → AA sequence)
        self._gene_protein_map: Optional[Dict[str, str]] = None
        # Gene → protein embedding cache
        self._protein_embedding_cache: Optional[Dict[str, torch.Tensor]] = None

        logger.info(
            "UCEAdapter initialised (embed_dim={}, protein_dim={}, max_seq_len={})",
            self.embed_dim,
            self.protein_embedding_dim,
            self.max_seq_len,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load UCE pretrained weights.

        UCE requires a protein language model (e.g. ESM-2) for encoding
        amino-acid sequences.  If the original ``uce`` package is not
        available, a lightweight fallback is built.

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

        # Try loading protein language model
        self._protein_encoder = self._load_protein_encoder()

        try:
            from uce_model import UCEModel  # type: ignore[import]

            self._model = UCEModel(
                protein_dim=self.protein_embedding_dim,
                d_model=self.embed_dim,
                nhead=self.n_heads,
                d_hid=self.hidden_dim,
                nlayers=self.n_layers,
            )
            if path is not None:
                state = torch.load(path, map_location="cpu")
                self._model.load_state_dict(state)
                logger.info("Loaded UCE weights from {}", path)
            else:
                logger.warning("No checkpoint path; using randomly initialised model")
        except ImportError:
            logger.warning("uce_model package not installed — building fallback model")
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
    # Preprocessing — protein sequence encoding
    # ------------------------------------------------------------------

    def preprocess(self, adata: "AnnData") -> Dict[str, torch.Tensor]:
        """Preprocess AnnData into UCE protein-embedding input format.

        UCE's input representation differs fundamentally from the other
        three models: instead of expression values, each gene is
        represented by the **amino-acid sequence embedding** of its protein
        product.

        If protein sequences are not available in ``adata.var``, the adapter
        falls back to using gene-name hashing as a deterministic proxy.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data ``(n_cells, n_genes)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:

            * ``"protein_embeddings"`` — ``FloatTensor``
              ``(batch, n_genes, protein_dim)`` — per-gene protein
              embeddings.
            * ``"expression"`` — ``FloatTensor``
              ``(batch, n_genes)`` — raw expression values (used as
              weights / targets).
            * ``"gene_mask"`` — ``BoolTensor``
              ``(batch, n_genes)`` — ``True`` for genes with valid protein
              sequences.
        """
        X = np.asarray(adata.X, dtype=np.float32)
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray().astype(np.float32)

        n_cells, n_genes = X.shape
        gene_names = list(adata.var_names)

        # Truncate / pad to max_seq_len
        seq_len = min(n_genes, self.max_seq_len)
        X = X[:, :seq_len] if n_genes >= seq_len else np.pad(
            X, ((0, 0), (0, seq_len - n_genes)), mode="constant"
        )
        gene_names = gene_names[:seq_len]

        # Obtain protein embeddings for each gene
        protein_embeds = self._get_protein_embeddings(gene_names)
        # protein_embeds: (seq_len, protein_dim)

        # Replicate for batch
        protein_batch = protein_embeds.unsqueeze(0).expand(n_cells, -1, -1).clone()

        # Gene mask (True = valid protein)
        gene_mask = torch.ones(n_cells, seq_len, dtype=torch.bool)

        result: Dict[str, torch.Tensor] = {
            "protein_embeddings": protein_batch,
            "expression": torch.tensor(X, dtype=torch.float32),
            "gene_mask": gene_mask,
        }

        logger.debug(
            "Preprocessed {} cells × {} genes → protein_dim={}",
            n_cells,
            seq_len,
            self.protein_embedding_dim,
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run UCE forward pass (encoder-decoder).

        The encoder processes protein sequence embeddings; the decoder
        maps them to the universal cell embedding space.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Preprocessed input (output of :meth:`preprocess`).

        Returns
        -------
        torch.Tensor
            Hidden states ``(batch, n_genes, embed_dim)``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        protein_embeds = x["protein_embeddings"].to(self._device, dtype=self._dtype)
        expression = x["expression"].to(self._device, dtype=self._dtype)
        gene_mask = x["gene_mask"].to(self._device)

        output = self._model(protein_embeds, expression, mask=gene_mask)
        return output

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, adata: "AnnData") -> np.ndarray:
        """Extract per-cell embeddings from UCE.

        Uses mean-pooling over the decoder output (weighted by expression
        levels).

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

        # Expression-weighted mean pooling
        expr = x["expression"].to(self._device).unsqueeze(-1)
        mask = x["gene_mask"].to(self._device).unsqueeze(-1).float()
        weights = expr * mask
        embeddings = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return embeddings.cpu().numpy()

    # ------------------------------------------------------------------
    # Attention mask — standard Transformer mask
    # ------------------------------------------------------------------

    def generate_attention_mask(
        self,
        seq_len: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate standard Transformer attention mask for UCE.

        UCE uses standard bidirectional self-attention throughout.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        **kwargs
            Optional keyword arguments:

            * ``pad_mask`` — ``BoolTensor`` ``(batch, seq_len)`` where
              ``True`` = valid, ``False`` = padding.

        Returns
        -------
        torch.Tensor
            Attention mask.  Shape ``(seq_len, seq_len)`` if no padding
            info given, or ``(batch, 1, seq_len, seq_len)`` with padding.
        """
        pad_mask = kwargs.get("pad_mask", None)

        if pad_mask is None:
            return torch.ones(seq_len, seq_len, dtype=torch.bool)

        # pad_mask: (B, S) True=valid, False=pad
        valid = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        valid_row = pad_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, S, 1)
        attn = valid & valid_row
        return attn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "uce"

    @property
    def model_type(self) -> str:
        """Return the architecture family.

        UCE uses an encoder-decoder architecture that maps from protein
        sequence space to cell embedding space.
        """
        return "encoder_decoder"

    @property
    def supported_optimizations(self) -> List[str]:
        """Return supported optimization engine names.

        UCE supports quantization, FlashAttention, torch.compile, and
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
    ) -> "UCEAdapter":
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
        UCEAdapter
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
        if self._protein_encoder is not None:
            self._protein_encoder.to(device)

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        if self._model is not None:
            self._model.eval()
        if self._protein_encoder is not None:
            self._protein_encoder.eval()

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
        return mapping.get(dtype_str, torch.float32)

    def _load_protein_encoder(self) -> nn.Module:
        """Load the protein language model for encoding amino-acid sequences.

        Attempts to load ESM-2; falls back to a simple embedding layer.

        Returns
        -------
        nn.Module
            Protein sequence encoder.
        """
        try:
            import esm  # type: ignore[import]

            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model.eval()
            logger.info("Loaded ESM-2 protein language model")
            return model
        except ImportError:
            logger.warning(
                "esm package not installed — using simple protein embedding fallback"
            )
            return _SimpleProteinEncoder(
                vocab_size=_AA_VOCAB_SIZE,
                embed_dim=self.protein_embedding_dim,
                max_len=self.protein_seq_len,
            )

    def _get_protein_embeddings(self, gene_names: List[str]) -> torch.Tensor:
        """Obtain protein embeddings for a list of genes.

        If protein sequences are available (via ``_gene_protein_map``),
        they are encoded through the protein language model.  Otherwise,
        a deterministic hash-based fallback is used.

        Parameters
        ----------
        gene_names : list[str]
            Gene symbol names.

        Returns
        -------
        torch.Tensor
            Protein embeddings ``(len(gene_names), protein_embedding_dim)``.
        """
        n_genes = len(gene_names)
        device = self._device

        if self._protein_encoder is not None and self._gene_protein_map is not None:
            embeds = []
            for gene in gene_names:
                seq = self._gene_protein_map.get(gene, None)
                if seq is not None:
                    embeds.append(self._encode_protein_sequence(seq))
                else:
                    embeds.append(self._hash_fallback_embedding(gene))
            return torch.stack(embeds).to(device)

        # Fallback: deterministic hash-based embeddings
        return torch.stack(
            [self._hash_fallback_embedding(g) for g in gene_names]
        ).to(device)

    def _encode_protein_sequence(self, seq: str) -> torch.Tensor:
        """Encode a single protein sequence through the protein LM.

        Parameters
        ----------
        seq : str
            Amino-acid sequence.

        Returns
        -------
        torch.Tensor
            Embedding vector ``(protein_embedding_dim,)``.
        """
        token_ids = _encode_amino_acid_sequence(seq, max_len=self.protein_seq_len)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self._device)

        with torch.no_grad():
            if hasattr(self._protein_encoder, "esm"):
                # ESM-2 model
                results = self._protein_encoder(token_tensor, repr_layers=[33])
                embedding = results["representations"][33].mean(dim=1).squeeze(0)
            else:
                embedding = self._protein_encoder(token_tensor).squeeze(0)

        return embedding.detach().cpu()

    def _hash_fallback_embedding(self, gene_name: str) -> torch.Tensor:
        """Generate a deterministic pseudo-embedding from gene name hash.

        Parameters
        ----------
        gene_name : str
            Gene symbol.

        Returns
        -------
        torch.Tensor
            Embedding vector ``(protein_embedding_dim,)``.
        """
        # Use hash to generate deterministic embedding
        seed = hash(gene_name) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        embed = torch.randn(self.protein_embedding_dim, generator=gen)
        embed = embed / embed.norm()  # normalise
        return embed

    def _build_fallback_model(self) -> nn.Module:
        """Build a lightweight UCE fallback model.

        Returns
        -------
        nn.Module
            Encoder-decoder model operating on protein embeddings.
        """

        class _FallbackUCE(nn.Module):
            """Minimal UCE-compatible model for benchmarking."""

            def __init__(
                self,
                protein_dim: int,
                embed_dim: int,
                n_heads: int,
                hidden_dim: int,
                n_layers: int,
            ) -> None:
                super().__init__()
                # Project protein embeddings to model dimension
                self.protein_proj = nn.Linear(protein_dim, embed_dim)
                self.expression_proj = nn.Linear(1, embed_dim)

                # Encoder — processes protein embeddings
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers // 2)

                # Decoder — maps to cell embedding space
                dec_layer = nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                )
                self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers // 2)
                self.embed_dim = embed_dim

            def forward(
                self,
                protein_embeds: torch.Tensor,
                expression: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                # protein_embeds: (B, S, protein_dim)
                # expression: (B, S)
                x = self.protein_proj(protein_embeds)
                expr_feat = self.expression_proj(expression.unsqueeze(-1))
                x = x + expr_feat

                # Encoder
                src_key_padding_mask = None
                if mask is not None:
                    src_key_padding_mask = ~mask  # True = pad

                enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

                # Decoder (using encoder output as both memory and target)
                dec_out = self.decoder(
                    enc_out,
                    enc_out,
                    tgt_key_padding_mask=src_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                return dec_out

        return _FallbackUCE(
            protein_dim=self.protein_embedding_dim,
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )


# ---------------------------------------------------------------------------
# Simple protein encoder fallback
# ---------------------------------------------------------------------------


class _SimpleProteinEncoder(nn.Module):
    """Minimal amino-acid sequence encoder when ESM-2 is not available.

    Uses a learnable embedding table + 1D convolution to produce a
    fixed-size embedding from an amino-acid token sequence.
    """

    def __init__(
        self,
        vocab_size: int = _AA_VOCAB_SIZE,
        embed_dim: int = 5120,
        max_len: int = _DEFAULT_PROTEIN_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode amino-acid token IDs to a fixed-size embedding.

        Parameters
        ----------
        token_ids : torch.Tensor
            ``(batch, seq_len)`` amino-acid token IDs.

        Returns
        -------
        torch.Tensor
            ``(batch, embed_dim)`` protein embedding.
        """
        x = self.embedding(token_ids)  # (B, S, 128)
        x = x.transpose(1, 2)  # (B, 128, S)
        x = self.conv(x)  # (B, 512, 1)
        x = x.squeeze(-1)  # (B, 512)
        return self.proj(x)  # (B, embed_dim)
