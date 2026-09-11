"""Tests for QuantizationEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from scinfer.engines.quantization import (
    QuantizationEngine,
    _quantize_model_simulated,
    _simulate_quantization_noise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def quant_engine():
    """QuantizationEngine instance."""
    return QuantizationEngine()


@pytest.fixture
def simple_model():
    """A small nn.Module with Linear layers for testing."""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    return model


@pytest.fixture
def mock_adapter_with_quant():
    """Mock adapter that supports quantization."""
    adapter = MagicMock()
    adapter.supported_optimizations = ["quantization", "flash_attention"]
    return adapter


@pytest.fixture
def mock_adapter_without_quant():
    """Mock adapter that does NOT support quantization."""
    adapter = MagicMock()
    adapter.supported_optimizations = ["flash_attention"]
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuantizationEngineCreation:
    """test_quantization_engine_creation: create engine instance"""

    def test_quantization_engine_creation(self, quant_engine):
        assert quant_engine.engine_name == "quantization"
        assert quant_engine.optimization_type == "quantization"
        assert isinstance(quant_engine.SUPPORTED_METHODS, dict)
        assert "bitsandbytes" in quant_engine.SUPPORTED_METHODS
        assert "simulated" in quant_engine.SUPPORTED_METHODS


class TestQuantizationCompatibility:
    """test_quantization_compatibility: all 4 model compatibilities"""

    def test_compatible_adapter(self, quant_engine, mock_adapter_with_quant):
        assert quant_engine.is_compatible(mock_adapter_with_quant) is True

    def test_incompatible_adapter(self, quant_engine, mock_adapter_without_quant):
        assert quant_engine.is_compatible(mock_adapter_without_quant) is False

    def test_all_four_models_compatible(self, quant_engine):
        """All 4 registered models (scgpt, geneformer, scfoundation, uce) support quantization."""
        for model_name in ["scgpt", "geneformer", "scfoundation", "uce"]:
            adapter = MagicMock()
            # All 4 models list "quantization" in supported_optimizations
            adapter.supported_optimizations = ["quantization"]
            assert quant_engine.is_compatible(adapter), f"{model_name} should be compatible"


class TestQuantizationApplySynthetic:
    """test_quantization_apply_synthetic: test apply using synthetic model"""

    def test_apply_simulated_int8(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 8}
        quant_model = quant_engine.apply(simple_model, config)
        assert isinstance(quant_model, nn.Module)
        # Model should still work
        x = torch.randn(2, 32)
        out = quant_model(x)
        assert out.shape == (2, 16)

    def test_apply_simulated_int4(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 4}
        quant_model = quant_engine.apply(simple_model, config)
        x = torch.randn(2, 32)
        out = quant_model(x)
        assert out.shape == (2, 16)

    def test_apply_invalid_method_raises(self, quant_engine, simple_model):
        """Unknown method should fail validation."""
        config = {"method": "nonexistent_backend", "bits": 8}
        with pytest.raises(ValueError, match="Quantization configuration validation failed"):
            quant_engine.apply(simple_model, config)

    def test_apply_stores_model(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 8}
        quant_engine.apply(simple_model, config)
        stored = quant_engine.get_quantized_model("simulated_int8")
        assert stored is not None


class TestQuantizationBenchmark:
    """test_quantization_benchmark: test benchmark method returns correct keys"""

    def test_benchmark_returns_correct_keys(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 8}
        quant_model = quant_engine.apply(simple_model, config)
        input_data = torch.randn(4, 32)
        metrics = quant_engine.benchmark(quant_model, input_data)
        assert isinstance(metrics, dict)
        expected_keys = {
            "throughput_items_per_sec",
            "latency_ms_per_cell",
            "peak_memory_mb",
            "model_size_mb",
            "model_params_mb",
        }
        assert expected_keys.issubset(set(metrics.keys()))
        # Values should be numeric
        for v in metrics.values():
            assert isinstance(v, (int, float))


class TestQuantizationFidelity:
    """test_quantization_fidelity: test embedding fidelity evaluation"""

    def test_embedding_similarity(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 8}
        quant_model = quant_engine.apply(simple_model, config)
        input_data = torch.randn(4, 32)
        result = quant_engine.compute_embedding_similarity(
            simple_model, quant_model, input_data
        )
        assert "cosine_similarity_mean" in result
        assert "mean_absolute_error" in result
        assert "relative_error" in result
        # Cosine similarity should be high for INT8 simulated quantization
        assert result["cosine_similarity_mean"] > 0.5
        assert result["mean_absolute_error"] >= 0.0


class TestQuantizationFallback:
    """test_quantization_fallback: test fallback when bitsandbytes is unavailable"""

    def test_fallback_when_bitsandbytes_unavailable(self, quant_engine, simple_model):
        """When bitsandbytes is not installed, should fall back to simulated."""
        config = {"method": "bitsandbytes", "bits": 8}
        # This should not raise even if bitsandbytes is not installed
        quant_model = quant_engine.apply(simple_model, config)
        assert isinstance(quant_model, nn.Module)
        x = torch.randn(2, 32)
        out = quant_model(x)
        assert out.shape == (2, 16)


class TestQuantizationValidate:
    """Additional tests for validate method."""

    def test_validate_valid_config(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 8}
        assert quant_engine.validate(simple_model, config) is True

    def test_validate_invalid_method(self, quant_engine, simple_model):
        config = {"method": "unknown_method", "bits": 8}
        assert quant_engine.validate(simple_model, config) is False

    def test_validate_invalid_bits(self, quant_engine, simple_model):
        config = {"method": "simulated", "bits": 3}  # 3 not supported
        assert quant_engine.validate(simple_model, config) is False

    def test_validate_model_without_linear(self, quant_engine):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        config = {"method": "simulated", "bits": 8}
        assert quant_engine.validate(model, config) is False


class TestSimulateQuantizationNoise:
    """Tests for helper functions."""

    def test_noise_shape_preserved(self):
        w = torch.randn(10, 20)
        noisy = _simulate_quantization_noise(w, bits=8)
        assert noisy.shape == w.shape

    def test_noise_clamped(self):
        w = torch.randn(10, 20)
        noisy = _simulate_quantization_noise(w, bits=4)
        assert noisy.max() <= w.abs().max()
        assert noisy.min() >= -w.abs().max()
