"""Integration tests for top-level scinfer API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import scinfer
from scinfer.api import OptimizedModel, encode, optimize, benchmark, profile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEncodeAPI:
    """test_encode_api: 测试scinfer.encode()（使用mock）"""

    def test_encode_exists_and_callable(self):
        """encode() is a callable function."""
        assert callable(encode)

    def test_encode_calls_adapter_with_mock(self):
        """encode() calls adapter.get_embeddings and writes to adata.obsm."""
        mock_adata = MagicMock()
        mock_adata.obsm = {}
        mock_adapter = MagicMock()
        mock_adapter.get_embeddings.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])

        with patch("scinfer.api.ModelRegistry") as mock_registry, \
             patch("scinfer.api._load_config", return_value={}):
            mock_registry.get.return_value = MagicMock(return_value=mock_adapter)
            result = encode(mock_adata, model="scgpt", optimize=False)
            mock_adapter.load_model.assert_called_once()
            mock_adapter.get_embeddings.assert_called_once_with(mock_adata)
            assert "X_scinfer" in result.obsm

    def test_encode_signature(self):
        """encode() has expected parameters."""
        import inspect
        sig = inspect.signature(encode)
        params = list(sig.parameters.keys())
        assert "adata" in params
        assert "model" in params
        assert "optimize" in params


class TestOptimizeAPI:
    """test_optimize_api: 测试scinfer.optimize()"""

    def test_optimize_exists_and_callable(self):
        """optimize() is a callable function."""
        assert callable(optimize)

    def test_optimize_returns_optimized_model(self):
        """optimize() returns an OptimizedModel instance."""
        mock_adapter = MagicMock()
        mock_model = MagicMock()
        mock_adapter.get_model.return_value = mock_model

        with patch("scinfer.api._get_adapter", return_value=mock_adapter), \
             patch("scinfer.api._load_config", return_value={}):
            result = optimize("scgpt")
            assert isinstance(result, OptimizedModel)
            assert result.model_name == "scgpt"
            mock_adapter.load_model.assert_called_once()

    def test_optimize_signature(self):
        """optimize() has expected parameters."""
        import inspect
        sig = inspect.signature(optimize)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "optimization_config" in params


class TestBenchmarkAPI:
    """test_benchmark_api: 测试scinfer.benchmark()"""

    def test_benchmark_exists_and_callable(self):
        """benchmark() is a callable function."""
        assert callable(benchmark)

    def test_benchmark_returns_benchmark_result(self):
        """benchmark() returns a BenchmarkResult."""
        from scinfer.evaluation.benchmark import BenchmarkResult

        mock_adapter = MagicMock()
        mock_nn = MagicMock()
        mock_nn.parameters.return_value = iter([MagicMock(shape=(8, 32))])
        mock_adapter.get_model.return_value = mock_nn

        with patch("scinfer.api._get_adapter", return_value=mock_adapter), \
             patch("scinfer.evaluation.benchmark.BenchmarkRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner.run.return_value = BenchmarkResult(
                results={"scgpt": {"throughput_ips": 100.0}}
            )
            mock_runner_cls.return_value = mock_runner
            result = benchmark(model_names=["scgpt"])
            assert isinstance(result, BenchmarkResult)

    def test_benchmark_signature(self):
        """benchmark() has expected parameters."""
        import inspect
        sig = inspect.signature(benchmark)
        params = list(sig.parameters.keys())
        assert "model_names" in params
        assert "datasets" in params
        assert "metrics" in params


class TestProfileAPI:
    """test_profile_api: 测试scinfer.profile()"""

    def test_profile_exists_and_callable(self):
        """profile() is a callable function."""
        assert callable(profile)

    def test_profile_returns_dict(self):
        """profile() returns a dict with profiling results."""
        from scinfer.evaluation.profiler import ProfileResult

        mock_adapter = MagicMock()
        mock_param = MagicMock()
        mock_param.shape = (8, 32)
        mock_param.dim = MagicMock(return_value=2)
        mock_nn = MagicMock()
        mock_nn.parameters.return_value = iter([mock_param])
        mock_adapter.get_model.return_value = mock_nn

        fake_result = ProfileResult(
            model_name="scgpt",
            total_latency_ms=5.0,
            peak_memory_mb=100.0,
        )

        with patch("scinfer.api._get_adapter", return_value=mock_adapter), \
             patch("scinfer.evaluation.profiler.InferenceProfiler") as mock_profiler_cls:
            mock_profiler = MagicMock()
            mock_profiler.profile.return_value = fake_result
            mock_profiler_cls.return_value = mock_profiler
            result = profile("scgpt")
            assert isinstance(result, dict)
            assert result["model_name"] == "scgpt"
            assert result["total_latency_ms"] == 5.0

    def test_profile_signature(self):
        """profile() has expected parameters."""
        import inspect
        sig = inspect.signature(profile)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "input_lengths" in params


class TestVersion:
    """test_version: 测试版本号存在"""

    def test_version_exists(self):
        """scinfer.__version__ is defined."""
        assert hasattr(scinfer, "__version__")
        assert isinstance(scinfer.__version__, str)
        assert len(scinfer.__version__) > 0

    def test_version_format(self):
        """Version follows semver-like format (X.Y.Z)."""
        parts = scinfer.__version__.split(".")
        assert len(parts) >= 2
        # Each part should be numeric
        for part in parts:
            assert part.isdigit()


class TestOptimizedModel:
    """Tests for the OptimizedModel dataclass."""

    def test_optimized_model_creation(self):
        """OptimizedModel can be instantiated."""
        om = OptimizedModel(
            model_name="scgpt",
            adapter=MagicMock(),
            engines_applied=["quantization"],
            config={"bits": 8},
            metrics={"throughput": 100.0},
        )
        assert om.model_name == "scgpt"
        assert om.engines_applied == ["quantization"]
        assert om.metrics["throughput"] == 100.0

    def test_optimized_model_defaults(self):
        """OptimizedModel has sensible defaults."""
        om = OptimizedModel(model_name="geneformer")
        assert om.adapter is None
        assert om.engines_applied == []
        assert om.config == {}
        assert om.metrics == {}


class TestTopLevelImports:
    """Verify all expected symbols are importable from scinfer."""

    def test_import_encode(self):
        from scinfer import encode
        assert callable(encode)

    def test_import_optimize(self):
        from scinfer import optimize
        assert callable(optimize)

    def test_import_benchmark(self):
        from scinfer import benchmark
        assert callable(benchmark)

    def test_import_profile(self):
        from scinfer import profile
        assert callable(profile)

    def test_import_config(self):
        from scinfer import Config
        assert Config is not None

    def test_import_registries(self):
        from scinfer import ModelRegistry, EngineRegistry
        assert ModelRegistry is not None
        assert EngineRegistry is not None
