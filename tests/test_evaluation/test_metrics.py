"""Tests for evaluation metrics (inference and biological)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scinfer.evaluation.metrics.inference import (
    InferenceMetrics,
    ThroughputResult,
    LatencyResult,
    MemoryResult,
)
from scinfer.evaluation.metrics.biological import (
    BiologicalMetrics,
    CellTypeResult,
    EmbeddingFidelityResult,
    PerturbationResult,
    DEResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """A small nn.Module for inference metrics testing."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    return model


@pytest.fixture
def input_data():
    """Synthetic input tensor."""
    return torch.randn(20, 16)


# ---------------------------------------------------------------------------
# Inference Metrics Tests
# ---------------------------------------------------------------------------


class TestInferenceMetricsThroughput:
    """test_inference_metrics_throughput: 测试吞吐量测量"""

    def test_throughput_measurement(self, simple_model, input_data):
        result = InferenceMetrics.measure_throughput(
            simple_model, input_data, batch_size=8, n_warmup=2, n_measure=5
        )
        assert isinstance(result, ThroughputResult)
        assert result.median > 0
        assert result.std >= 0
        assert len(result.all_measurements) == 5

    def test_throughput_repr(self):
        result = ThroughputResult(median=1000.0, std=50.0)
        assert "1000" in repr(result)


class TestInferenceMetricsLatency:
    """test_inference_metrics_latency: 测试延迟测量"""

    def test_latency_measurement(self, simple_model, input_data):
        result = InferenceMetrics.measure_latency(
            simple_model, input_data, n_warmup=2, n_measure=10
        )
        assert isinstance(result, LatencyResult)
        assert result.median_ms > 0
        assert result.std_ms >= 0
        assert "p50" in result.percentiles
        assert "p90" in result.percentiles
        assert "p99" in result.percentiles
        assert len(result.all_measurements_ms) == 10

    def test_latency_percentiles_ordered(self, simple_model, input_data):
        result = InferenceMetrics.measure_latency(
            simple_model, input_data, n_warmup=2, n_measure=20
        )
        assert result.percentiles["p50"] <= result.percentiles["p90"]
        assert result.percentiles["p90"] <= result.percentiles["p99"]


class TestInferenceMetricsMemory:
    """test_inference_metrics_memory: 测试内存测量"""

    def test_memory_measurement(self, simple_model, input_data):
        result = InferenceMetrics.measure_peak_memory(
            simple_model, input_data, batch_size=8
        )
        assert isinstance(result, MemoryResult)
        assert result.model_params_mb >= 0
        # On CPU, peak_gpu_mb should be 0
        assert result.peak_gpu_mb == 0.0

    def test_memory_model_params_positive(self, simple_model, input_data):
        result = InferenceMetrics.measure_peak_memory(simple_model, input_data)
        # Model has parameters → params_mb > 0
        assert result.model_params_mb > 0


# ---------------------------------------------------------------------------
# Biological Metrics Tests
# ---------------------------------------------------------------------------


class TestBiologicalMetricsCellAnnotation:
    """test_biological_metrics_cell_annotation: 测试ARI/NMI/F1计算"""

    def test_cell_type_annotation(self):
        rng = np.random.default_rng(42)
        # Create well-separated clusters
        n_per_cluster = 30
        embeddings = np.vstack([
            rng.standard_normal((n_per_cluster, 10)) + i * 5
            for i in range(3)
        ]).astype(np.float64)
        labels = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

        result = BiologicalMetrics.cell_type_annotation(embeddings, labels)
        assert isinstance(result, CellTypeResult)
        # Well-separated clusters → high ARI/NMI (permutation-invariant)
        assert result.ari > 0.5
        assert result.nmi > 0.5
        # F1 depends on label alignment; ARI/NMI are the primary metrics
        assert result.f1_macro >= 0.0
        assert result.f1_weighted >= 0.0

    def test_cell_type_annotation_with_cluster_labels(self):
        embeddings = np.random.randn(50, 5).astype(np.float64)
        labels = np.array([0] * 25 + [1] * 25)
        cluster_labels = np.array([0] * 25 + [1] * 25)  # perfect clustering

        result = BiologicalMetrics.cell_type_annotation(
            embeddings, labels, cluster_labels=cluster_labels
        )
        assert result.ari == 1.0
        assert result.nmi == 1.0


class TestBiologicalMetricsEmbeddingFidelity:
    """test_biological_metrics_embedding_fidelity: 测试kNN accuracy"""

    def test_embedding_fidelity_identical(self):
        """Identical embeddings → kNN accuracy = 1.0."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((50, 10)).astype(np.float64)
        result = BiologicalMetrics.embedding_fidelity(emb, emb, k=5)
        assert isinstance(result, EmbeddingFidelityResult)
        assert result.knn_accuracy == 1.0

    def test_embedding_fidelity_different(self):
        """Very different embeddings → lower kNN accuracy."""
        rng = np.random.default_rng(42)
        emb_orig = rng.standard_normal((50, 10)).astype(np.float64)
        emb_opt = rng.standard_normal((50, 10)).astype(np.float64) * 100  # very different
        result = BiologicalMetrics.embedding_fidelity(emb_orig, emb_opt, k=5)
        # kNN accuracy should be lower than 1.0 for random embeddings
        assert result.knn_accuracy < 1.0
        assert 0.0 <= result.trustworthiness <= 1.0
        assert 0.0 <= result.continuity <= 1.0


class TestBiologicalMetricsPerturbation:
    """test_biological_metrics_perturbation: 测试PCC/Spearman"""

    def test_perturbation_prediction_1d(self):
        """1D perturbation: perfectly correlated → PCC ≈ 1."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        truth = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = BiologicalMetrics.perturbation_prediction(pred, truth)
        assert isinstance(result, PerturbationResult)
        assert result.pcc > 0.99
        assert result.spearman_rho > 0.99

    def test_perturbation_prediction_2d(self):
        """2D batched perturbation."""
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        truth = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
        results = BiologicalMetrics.perturbation_prediction(pred, truth)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r.pcc > 0.99

    def test_perturbation_uncorrelated(self):
        """Uncorrelated predictions → low PCC."""
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100)
        truth = rng.standard_normal(100)
        result = BiologicalMetrics.perturbation_prediction(pred, truth)
        assert abs(result.pcc) < 0.3  # should be near 0


class TestBiologicalMetricsDEConsistency:
    """test_biological_metrics_de_consistency: 测试Jaccard index"""

    def test_de_consistency_identical(self):
        """Identical gene lists → Jaccard = 1.0."""
        genes = np.array(["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"])
        result = BiologicalMetrics.differential_expression_consistency(genes, genes, top_n=5)
        assert isinstance(result, DEResult)
        assert result.jaccard_topn == 1.0

    def test_de_consistency_disjoint(self):
        """Completely different gene lists → Jaccard = 0.0."""
        genes_orig = np.array(["A", "B", "C"])
        genes_opt = np.array(["X", "Y", "Z"])
        result = BiologicalMetrics.differential_expression_consistency(
            genes_orig, genes_opt, top_n=3
        )
        assert result.jaccard_topn == 0.0

    def test_de_consistency_partial_overlap(self):
        """Partial overlap → 0 < Jaccard < 1."""
        genes_orig = ["A", "B", "C", "D"]
        genes_opt = ["B", "C", "E", "F"]
        result = BiologicalMetrics.differential_expression_consistency(
            genes_orig, genes_opt, top_n=4
        )
        # Intersection: {B, C}, Union: {A, B, C, D, E, F} → 2/6 = 0.333
        assert abs(result.jaccard_topn - 2 / 6) < 1e-6

    def test_de_consistency_list_of_lists(self):
        """Multiple comparisons as list of lists."""
        orig = [["A", "B", "C"], ["D", "E", "F"]]
        opt = [["A", "B", "X"], ["D", "E", "Y"]]
        result = BiologicalMetrics.differential_expression_consistency(
            orig, opt, top_n=3
        )
        assert len(result.jaccard_per_comparison) == 2
        # First: {A,B} / {A,B,C,X} = 2/4 = 0.5
        assert abs(result.jaccard_per_comparison[0] - 0.5) < 1e-6
