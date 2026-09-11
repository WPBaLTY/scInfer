"""Tests for BenchmarkResult, BenchmarkRunner, and BenchmarkProtocol."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from scinfer.evaluation.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkProtocol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    """A BenchmarkResult with sample data."""
    return BenchmarkResult(
        results={
            "model_a": {"throughput": 100.0, "latency": 10.0, "memory": 2000.0},
            "model_b": {"throughput": 200.0, "latency": 5.0, "memory": 1000.0},
        },
        config={"warmup_runs": 5, "benchmark_runs": 20},
        metadata={"platform": "test", "torch_version": "2.0.0"},
    )


@pytest.fixture
def simple_model():
    """A small nn.Module for benchmark testing."""
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


# ---------------------------------------------------------------------------
# BenchmarkResult Tests
# ---------------------------------------------------------------------------


class TestBenchmarkResultCreation:
    """test_benchmark_result_creation: 创建结果实例"""

    def test_benchmark_result_creation(self, sample_result):
        assert "model_a" in sample_result.results
        assert "model_b" in sample_result.results
        assert sample_result.results["model_a"]["throughput"] == 100.0
        assert sample_result.config["warmup_runs"] == 5

    def test_empty_result(self):
        result = BenchmarkResult()
        assert result.results == {}
        assert result.summary() == "(no results)"

    def test_summary_display(self, sample_result):
        summary = sample_result.summary()
        assert "model_a" in summary
        assert "model_b" in summary
        assert "throughput" in summary


class TestBenchmarkResultToJSON:
    """test_benchmark_result_to_json: 导出JSON"""

    def test_to_json(self, sample_result, tmp_path):
        out_path = tmp_path / "results.json"
        returned = sample_result.to_json(out_path)
        assert returned == out_path
        assert out_path.exists()

        with open(out_path) as f:
            data = json.load(f)
        assert "results" in data
        assert "model_a" in data["results"]
        assert data["results"]["model_a"]["throughput"] == 100.0


class TestBenchmarkResultToCSV:
    """test_benchmark_result_to_csv: 导出CSV"""

    def test_to_csv(self, sample_result, tmp_path):
        out_path = tmp_path / "results.csv"
        returned = sample_result.to_csv(out_path)
        assert returned == out_path
        assert out_path.exists()

        with open(out_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "model" in header
            rows = list(reader)
            assert len(rows) == 2  # model_a and model_b


class TestBenchmarkResultCompare:
    """test_benchmark_result_compare: 测试两个结果对比"""

    def test_compare(self):
        baseline = BenchmarkResult(
            results={
                "model_a": {"throughput": 100.0, "latency": 10.0},
            }
        )
        optimized = BenchmarkResult(
            results={
                "model_a": {"throughput": 200.0, "latency": 5.0},
            }
        )
        comparison = baseline.compare(optimized)
        assert "model_a" in comparison
        assert comparison["model_a"]["throughput"] == 2.0  # 200/100
        assert comparison["model_a"]["latency"] == 0.5  # 5/10

    def test_compare_no_overlap(self):
        baseline = BenchmarkResult(results={"model_a": {"throughput": 100.0}})
        other = BenchmarkResult(results={"model_b": {"throughput": 200.0}})
        comparison = baseline.compare(other)
        assert comparison == {}  # no common models


class TestBenchmarkProtocolCreation:
    """test_benchmark_protocol_creation: 创建协议实例"""

    def test_protocol_creation(self, tmp_path):
        protocol = BenchmarkProtocol(
            output_dir=tmp_path / "bench",
            warmup_runs=5,
            benchmark_runs=10,
        )
        assert protocol.warmup_runs == 5
        assert protocol.benchmark_runs == 10
        assert protocol.output_dir.exists()

    def test_register_model(self, tmp_path, simple_model):
        protocol = BenchmarkProtocol(output_dir=tmp_path / "bench")
        protocol.register_model("test_model", simple_model)
        assert "test_model" in protocol._models


class TestBenchmarkRunnerSynthetic:
    """test_benchmark_runner_synthetic: 使用合成数据测试"""

    def test_runner_creation(self):
        runner = BenchmarkRunner(
            model_names=["model_a", "model_b"],
            metrics=["throughput", "latency"],
            warmup_runs=2,
            benchmark_runs=5,
        )
        assert runner.model_names == ["model_a", "model_b"]
        assert runner.warmup_runs == 2

    def test_runner_run_with_synthetic_data(self, simple_model):
        """Run benchmark with synthetic model and data."""
        runner = BenchmarkRunner(
            model_names=["test_model"],
            metrics=["throughput", "latency"],
            warmup_runs=1,
            benchmark_runs=3,
        )
        input_data = {"test_model": torch.randn(10, 16)}
        models = {"test_model": simple_model}

        result = runner.run(input_data=input_data, models=models)
        assert isinstance(result, BenchmarkResult)
        assert "test_model" in result.results
        assert "throughput_ips" in result.results["test_model"]
        assert "latency_median_ms" in result.results["test_model"]
        # Metadata should be populated
        assert "timestamp" in result.metadata
        assert "platform" in result.metadata

    def test_runner_skip_missing_model(self):
        """Runner should skip models with no data/model provided."""
        runner = BenchmarkRunner(
            model_names=["missing_model"],
            warmup_runs=1,
            benchmark_runs=2,
        )
        result = runner.run(input_data={}, models={})
        assert result.results == {}

    def test_runner_to_dict(self, simple_model):
        runner = BenchmarkRunner(
            model_names=["m"],
            metrics=["throughput"],
            warmup_runs=1,
            benchmark_runs=2,
        )
        result = runner.run(
            input_data={"m": torch.randn(5, 16)},
            models={"m": simple_model},
        )
        d = result.to_dict()
        assert "results" in d
        assert "config" in d
        assert "metadata" in d
