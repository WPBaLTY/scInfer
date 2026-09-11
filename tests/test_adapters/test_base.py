"""Tests for the BaseModelAdapter abstract base class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from scinfer.adapters.base import BaseModelAdapter


class TestBaseModelAdapterABC:
    """Tests for BaseModelAdapter abstract interface."""

    def test_cannot_instantiate_directly(self):
        """BaseModelAdapter cannot be instantiated (it's abstract)."""
        with pytest.raises(TypeError):
            BaseModelAdapter(config={})  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all(self):
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompleteAdapter(BaseModelAdapter):
            def __init__(self, config: Dict[str, Any]) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteAdapter(config={})  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        """A fully-implemented subclass can be instantiated."""

        class CompleteAdapter(BaseModelAdapter):
            def __init__(self, config: Dict[str, Any]) -> None:
                self._config = config

            def load_model(self, checkpoint_path: Optional[str] = None) -> None:
                pass

            def preprocess(self, adata: Any) -> torch.Tensor:
                return torch.zeros(1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def get_embeddings(self, adata: Any) -> np.ndarray:
                return np.zeros((1, 10))

            def generate_attention_mask(self, seq_len: int, **kwargs: Any) -> torch.Tensor:
                return torch.ones(seq_len, seq_len)

            @property
            def model_name(self) -> str:
                return "test"

            @property
            def model_type(self) -> str:
                return "encoder_only"

            @property
            def supported_optimizations(self) -> List[str]:
                return ["quantization"]

        adapter = CompleteAdapter(config={"test": True})
        assert adapter.model_name == "test"
        assert adapter.model_type == "encoder_only"
        assert "quantization" in adapter.supported_optimizations

    def test_optional_hooks_raise_runtime_error(self):
        """Optional hooks raise RuntimeError when no model is loaded."""

        class MinimalAdapter(BaseModelAdapter):
            def __init__(self, config: Dict[str, Any]) -> None:
                pass

            def load_model(self, checkpoint_path: Optional[str] = None) -> None:
                pass

            def preprocess(self, adata: Any) -> torch.Tensor:
                return torch.zeros(1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def get_embeddings(self, adata: Any) -> np.ndarray:
                return np.zeros((1, 10))

            def generate_attention_mask(self, seq_len: int, **kwargs: Any) -> torch.Tensor:
                return torch.ones(seq_len, seq_len)

            @property
            def model_name(self) -> str:
                return "test"

            @property
            def model_type(self) -> str:
                return "encoder_only"

            @property
            def supported_optimizations(self) -> List[str]:
                return []

        adapter = MinimalAdapter(config={})
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.get_model()
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.to_device(torch.device("cpu"))
        with pytest.raises(RuntimeError, match="No model loaded"):
            adapter.eval()
