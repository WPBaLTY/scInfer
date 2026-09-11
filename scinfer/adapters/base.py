"""Abstract base class for all single-cell foundation model adapters.

Every supported model (scGPT, Geneformer, scFoundation, UCE) must provide
a concrete adapter that inherits from :class:`BaseModelAdapter`.  This
ensures a uniform interface for the optimisation engines and the top-level
API regardless of the underlying model architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn


class BaseModelAdapter(ABC):
    """Abstract base class that every model adapter must implement.

    The adapter is responsible for:
    * Loading the model from a checkpoint.
    * Preprocessing AnnData objects into model-specific tensor inputs.
    * Running the forward pass.
    * Extracting cell embeddings.
    * Generating attention masks compatible with the model architecture.

    Parameters
    ----------
    config : dict
        Model-specific configuration dictionary (typically loaded from
        ``configs/models/<model>.yaml``).
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the adapter with model configuration.

        Parameters
        ----------
        config : dict
            Model configuration dictionary.
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights from disk or hub.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to a local checkpoint file.  If ``None``, the adapter
            should download / locate the default pretrained weights.
        """
        ...

    @abstractmethod
    def preprocess(self, adata: "AnnData") -> torch.Tensor:  # noqa: F821
        """Convert an AnnData object into the tensor format expected by the model.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data (cells × genes).

        Returns
        -------
        torch.Tensor
            Model-ready input tensor.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input tensor (output of :meth:`preprocess`).

        Returns
        -------
        torch.Tensor
            Raw model output (logits or hidden states).
        """
        ...

    @abstractmethod
    def get_embeddings(self, adata: "AnnData") -> np.ndarray:  # noqa: F821
        """Extract per-cell embedding vectors.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell expression data.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_cells, embed_dim)``.
        """
        ...

    @abstractmethod
    def generate_attention_mask(
        self, seq_len: int, **kwargs: Any
    ) -> torch.Tensor:
        """Create an attention mask compatible with the model architecture.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        **kwargs
            Model-specific mask parameters (e.g. padding length, batch size).

        Returns
        -------
        torch.Tensor
            Attention mask tensor.
        """
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``"scgpt"``)."""
        ...

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Architecture family: ``"encoder_only"``, ``"encoder_decoder"``, etc."""
        ...

    @property
    @abstractmethod
    def supported_optimizations(self) -> List[str]:
        """List of optimization engine names this adapter is compatible with."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (default implementations provided)
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        """Return the underlying ``torch.nn.Module``.

        Returns
        -------
        torch.nn.Module

        Raises
        ------
        RuntimeError
            If the adapter has not loaded a model yet.
        """
        raise RuntimeError(
            "No model loaded. Call load_model() before accessing the model."
        )

    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device.
        """
        raise RuntimeError(
            "No model loaded. Call load_model() before moving to device."
        )

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        raise RuntimeError(
            "No model loaded. Call load_model() before setting eval mode."
        )
