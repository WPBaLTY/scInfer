"""Abstract base class for all inference optimisation engines.

Every optimisation strategy (quantization, FlashAttention, torch.compile,
batch scheduling, etc.) must provide a concrete engine that inherits from
:class:`BaseOptimizationEngine`.  This ensures a uniform interface for the
optimisation combiner and the top-level API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from loguru import logger
from torch import nn

from scinfer.adapters.base import BaseModelAdapter


class BaseOptimizationEngine(ABC):
    """Abstract base class that every optimisation engine must implement.

    An engine is responsible for:
    * Applying a specific optimisation to a model (e.g. quantization).
    * Benchmarking the optimised model (latency, throughput, memory).
    * Checking compatibility with a given model adapter.
    """

    @abstractmethod
    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Apply the optimisation to the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to optimise.
        config : dict
            Engine-specific configuration parameters.

        Returns
        -------
        torch.nn.Module
            The optimised model (may be the same object or a new wrapper).
        """
        ...

    @abstractmethod
    def benchmark(
        self, model: nn.Module, input_data: torch.Tensor
    ) -> Dict[str, float]:
        """Run a short benchmark to measure the optimisation's impact.

        Parameters
        ----------
        model : torch.nn.Module
            The (optimised) model to benchmark.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Dictionary of metric name → value, e.g.
            ``{"throughput_fps": 1234.5, "latency_ms": 8.1, "peak_memory_mb": 2048}``.
        """
        ...

    @abstractmethod
    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check whether this engine is compatible with the given adapter.

        Parameters
        ----------
        adapter : BaseModelAdapter
            The model adapter to check.

        Returns
        -------
        bool
            ``True`` if the engine can optimise this adapter's model.
        """
        ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine identifier (e.g. ``"quantization"``)."""
        ...

    @property
    @abstractmethod
    def optimization_type(self) -> str:
        """Optimisation category (e.g. ``"quantization"``, ``"flash_attention"``)."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Pre-flight validation before applying the optimisation.

        Default implementation returns True. Subclasses should override
        this method to provide custom validation logic.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        config : dict
            Engine-specific configuration.

        Returns
        -------
        bool
            ``True`` if validation passes.
        """
        return True

    def revert(self, model: nn.Module) -> nn.Module:
        """Revert the optimisation and return the original model.

        Default implementation logs a warning and returns the original model.
        Subclasses should override this method to provide custom revert logic.

        Parameters
        ----------
        model : torch.nn.Module
            The optimised model.

        Returns
        -------
        torch.nn.Module
            The model with the optimisation removed.
        """
        logger.warning(
            f"{self.__class__.__name__} does not implement revert(), "
            "returning original model"
        )
        return model
