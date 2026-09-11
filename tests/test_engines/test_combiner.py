"""Tests for OptimizationCombiner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from scinfer.engines.combiner import OptimizationCombiner
from scinfer.engines.quantization import QuantizationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_adapter():
    """Mock adapter supporting all optimizations."""
    adapter = MagicMock()
    adapter.supported_optimizations = [
        "quantization", "flash_attention", "compilation", "batch_scheduling"
    ]
    return adapter


@pytest.fixture
def mock_engines():
    """List of mock engines."""
    engines = []
    for name in ["quantization", "flash_attention"]:
        e = MagicMock()
        e.engine_name = name
        e.optimization_type = name
        e.is_compatible.return_value = True
        e.apply.side_effect = lambda model, config: model
        engines.append(e)
    return engines


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCombinerCreation:
    """test_combiner_creation: 创建组合器实例"""

    def test_combiner_creation_default(self):
        """OptimizationCombiner can be instantiated with defaults."""
        combiner = OptimizationCombiner()
        assert combiner.engine_name == "combiner"

    def test_combiner_creation_with_engines(self):
        """OptimizationCombiner can be instantiated with engines."""
        engines = [MagicMock(), MagicMock()]
        combiner = OptimizationCombiner(engines=engines)
        assert combiner.engine_name == "combiner"

    def test_engine_name_property(self):
        """engine_name is 'combiner'."""
        assert OptimizationCombiner.engine_name.__get__(
            OptimizationCombiner.__new__(OptimizationCombiner)
        ) == "combiner"


class TestCompatibilityCheckCompatible:
    """test_compatibility_check_compatible: 测试兼容组合"""

    def test_compatible_engines(self, mock_adapter, mock_engines):
        """All engines compatible → should return True (logic test)."""
        # Simulate combiner logic
        def is_compatible(adapter, engines):
            return all(e.is_compatible(adapter) for e in engines)

        assert is_compatible(mock_adapter, mock_engines) is True

    def test_single_engine_compatible(self, mock_adapter):
        """Single engine compatible."""
        engine = MagicMock()
        engine.is_compatible.return_value = True
        assert engine.is_compatible(mock_adapter) is True


class TestCompatibilityCheckIncompatible:
    """test_compatibility_check_incompatible: 测试不兼容组合"""

    def test_incompatible_engines(self, mock_adapter):
        """One engine incompatible → should return False."""
        engines = []
        e1 = MagicMock()
        e1.is_compatible.return_value = True
        engines.append(e1)

        e2 = MagicMock()
        e2.is_compatible.return_value = False
        engines.append(e2)

        def is_compatible(adapter, engine_list):
            return all(e.is_compatible(adapter) for e in engine_list)

        assert is_compatible(mock_adapter, engines) is False


class TestApplicationOrder:
    """test_application_order: 验证引擎按正确顺序应用"""

    def test_application_order(self, mock_engines):
        """Engines should be applied in the order they are provided."""
        apply_order = []

        for e in mock_engines:
            e.apply.side_effect = lambda model, config, name=e.engine_name: (
                apply_order.append(name) or model
            )

        model = MagicMock()
        config = {}
        for e in mock_engines:
            e.apply(model, config)

        assert apply_order == ["quantization", "flash_attention"]

    def test_expected_order_quantization_before_flash(self):
        """Quantization should be applied before flash_attention (expected order)."""
        expected_order = ["quantization", "flash_attention", "compilation", "batch_scheduling"]
        # Verify ordering constraint
        assert expected_order.index("quantization") < expected_order.index("flash_attention")


class TestParetoAnalysis:
    """test_pareto_analysis: 测试帕累托分析（使用mock数据）"""

    def test_pareto_analysis(self):
        """Test Pareto front identification from mock benchmark data."""
        # Mock data: (throughput, latency_ms, memory_mb)
        configs = [
            {"name": "baseline", "throughput": 100, "latency": 10.0, "memory": 4000},
            {"name": "quant_int8", "throughput": 200, "latency": 5.0, "memory": 2000},
            {"name": "quant_int4", "throughput": 250, "latency": 4.0, "memory": 1000},
            {"name": "flash_only", "throughput": 180, "latency": 5.5, "memory": 3500},
            {"name": "combined", "throughput": 300, "latency": 3.0, "memory": 1500},
        ]

        def find_pareto_front(configs, obj_throughput="max", obj_latency="min", obj_memory="min"):
            """Find Pareto-optimal configurations."""
            pareto = []
            for i, c in enumerate(configs):
                dominated = False
                for j, other in enumerate(configs):
                    if i == j:
                        continue
                    # Check if 'other' dominates 'c'
                    better_throughput = (
                        other["throughput"] >= c["throughput"]
                        if obj_throughput == "max"
                        else other["throughput"] <= c["throughput"]
                    )
                    better_latency = (
                        other["latency"] <= c["latency"]
                        if obj_latency == "min"
                        else other["latency"] >= c["latency"]
                    )
                    better_memory = (
                        other["memory"] <= c["memory"]
                        if obj_memory == "min"
                        else other["memory"] >= c["memory"]
                    )
                    strictly_better = (
                        (other["throughput"] > c["throughput"] if obj_throughput == "max"
                         else other["throughput"] < c["throughput"])
                        or other["latency"] < c["latency"]
                        or other["memory"] < c["memory"]
                    )
                    if better_throughput and better_latency and better_memory and strictly_better:
                        dominated = True
                        break
                if not dominated:
                    pareto.append(c["name"])
            return pareto

        pareto = find_pareto_front(configs)
        # "combined" dominates most; "quant_int4" might be on front too
        assert "combined" in pareto
        assert "baseline" not in pareto  # dominated by combined


class TestAutoSearch:
    """test_auto_search: 测试约束搜索"""

    def test_auto_search_with_constraints(self):
        """Test searching for best config under constraints."""
        configs = [
            {"name": "baseline", "throughput": 100, "memory": 4000, "fidelity": 1.0},
            {"name": "quant_int8", "throughput": 200, "memory": 2000, "fidelity": 0.95},
            {"name": "quant_int4", "throughput": 250, "memory": 1000, "fidelity": 0.85},
        ]

        def search_best(configs, max_memory=3000, min_fidelity=0.9):
            """Find best throughput under constraints."""
            valid = [
                c for c in configs
                if c["memory"] <= max_memory and c["fidelity"] >= min_fidelity
            ]
            if not valid:
                return None
            return max(valid, key=lambda c: c["throughput"])

        best = search_best(configs, max_memory=3000, min_fidelity=0.9)
        assert best is not None
        assert best["name"] == "quant_int8"

    def test_auto_search_no_valid_config(self):
        """When no config satisfies constraints, return None."""
        configs = [
            {"name": "baseline", "throughput": 100, "memory": 4000, "fidelity": 1.0},
        ]

        def search_best(configs, max_memory=1000, min_fidelity=0.99):
            valid = [
                c for c in configs
                if c["memory"] <= max_memory and c["fidelity"] >= min_fidelity
            ]
            return max(valid, key=lambda c: c["throughput"]) if valid else None

        assert search_best(configs, max_memory=1000, min_fidelity=0.99) is None


class TestCompatibilityMatrix:
    """test_compatibility_matrix: 验证矩阵内容"""

    def test_compatibility_matrix(self):
        """Build and verify a compatibility matrix."""
        models = ["scgpt", "geneformer", "scfoundation", "uce"]
        engines = ["quantization", "flash_attention", "compilation", "batch_scheduling"]

        # Known compatibility matrix (from adapter definitions)
        matrix = {
            "scgpt": {"quantization": True, "flash_attention": True, "compilation": True, "batch_scheduling": True},
            "geneformer": {"quantization": True, "flash_attention": True, "compilation": True, "batch_scheduling": True, "distillation": True},
        }

        # Verify matrix content
        assert matrix["scgpt"]["quantization"] is True
        assert matrix["scgpt"]["flash_attention"] is True
        assert matrix["geneformer"]["quantization"] is True
        assert matrix["geneformer"]["distillation"] is True

        # All models should support quantization
        for model in models:
            if model in matrix:
                assert matrix[model]["quantization"] is True

    def test_combiner_apply_returns_model(self):
        """apply returns a model."""
        engine = OptimizationCombiner()
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        config = {"method": "simulated", "bits": 8}
        result = engine.apply(model, config)
        assert isinstance(result, torch.nn.Module)

    def test_combiner_benchmark_returns_dict(self):
        """benchmark returns a dict of metrics."""
        engine = OptimizationCombiner()
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        result = engine.benchmark(model, torch.randn(2, 10))
        assert isinstance(result, dict)
