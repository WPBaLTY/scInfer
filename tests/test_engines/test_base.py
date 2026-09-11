"""Tests for the BaseOptimizationEngine abstract base class."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from torch import nn

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine


class TestBaseOptimizationEngineABC:
    """Tests for BaseOptimizationEngine abstract interface."""

    def test_cannot_instantiate_directly(self):
        """BaseOptimizationEngine cannot be instantiated (it's abstract)."""
        with pytest.raises(TypeError):
            BaseOptimizationEngine()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all(self):
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompleteEngine(BaseOptimizationEngine):
            def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
                return model

        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        """A fully-implemented subclass can be instantiated."""

        class CompleteEngine(BaseOptimizationEngine):
            def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
                return model

            def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
                return {"throughput": 100.0}

            def is_compatible(self, adapter: BaseModelAdapter) -> bool:
                return True

            @property
            def engine_name(self) -> str:
                return "test_engine"

            @property
            def optimization_type(self) -> str:
                return "test"

        engine = CompleteEngine()
        assert engine.engine_name == "test_engine"
        assert engine.optimization_type == "test"

    def test_optional_hooks_default_implementations(self):
        """Optional hooks have sensible default implementations."""

        class MinimalEngine(BaseOptimizationEngine):
            def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
                return model

            def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
                return {}

            def is_compatible(self, adapter: BaseModelAdapter) -> bool:
                return True

            @property
            def engine_name(self) -> str:
                return "minimal"

            @property
            def optimization_type(self) -> str:
                return "test"

        engine = MinimalEngine()
        model = nn.Linear(10, 10)
        # validate() defaults to returning True
        assert engine.validate(model, {}) is True
        # revert() defaults to returning the original model
        assert engine.revert(model) is model
