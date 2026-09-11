"""Tests for BioValidationResult and BioValidationPipeline."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scinfer.evaluation.bio_validation import (
    BioValidationPipeline,
    BioValidationResult,
    _generate_synthetic_expression,
    _generate_synthetic_perturbation,
    _generate_synthetic_grn,
)
from scinfer.evaluation.metrics.biological import (
    CellTypeResult,
    EmbeddingFidelityResult,
    GeneNetworkResult,
    PerturbationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    """A BioValidationResult with sample data."""
    return BioValidationResult(
        cell_type=CellTypeResult(ari=0.85, nmi=0.80, f1_macro=0.82, f1_weighted=0.84),
        perturbation=PerturbationResult(pcc=0.92, spearman_rho=0.88),
        grn=GeneNetworkResult(auroc=0.78, auprc=0.65),
        drug_response={"f1": 0.75, "auroc": 0.80, "accuracy": 0.77},
        knockout={"consistency_score": 0.70, "gene_name": "TP53"},
        embedding_fidelity=EmbeddingFidelityResult(
            knn_accuracy=0.88, trustworthiness=0.85, continuity=0.82
        ),
        metadata={"model_name": "test_model"},
    )


@pytest.fixture
def empty_result():
    """An empty BioValidationResult."""
    return BioValidationResult()


@pytest.fixture
def mock_adapter():
    """Mock model adapter for pipeline testing.

    We delete ``get_embeddings`` so the pipeline falls back to its internal
    random-projection embedding, which returns a properly shaped numpy array.
    """
    adapter = MagicMock()
    adapter.model_name = "test_model"
    del adapter.get_embeddings
    return adapter


# ---------------------------------------------------------------------------
# Tests — BioValidationResult.summary()
# ---------------------------------------------------------------------------


class TestBioValidationResultSummary:
    """test_biovalidation_result_summary: summary output"""

    def test_summary_with_all_dimensions(self, sample_result):
        summary = sample_result.summary()
        assert "Biological Validation Result Summary" in summary
        assert "Cell type annotation" in summary
        assert "Perturbation prediction" in summary
        assert "GRN inference" in summary
        assert "Drug response" in summary
        assert "knockout" in summary.lower() or "In silico" in summary
        assert "Embedding fidelity" in summary

    def test_summary_empty_result(self, empty_result):
        summary = empty_result.summary()
        assert "Not executed" in summary


# ---------------------------------------------------------------------------
# Tests — BioValidationResult serialization
# ---------------------------------------------------------------------------


class TestBioValidationResultToJSON:
    """test_biovalidation_result_to_json: JSON export"""

    def test_to_json(self, sample_result, tmp_path):
        out_path = tmp_path / "bio_results.json"
        returned = sample_result.to_json(out_path)
        assert returned == out_path
        assert out_path.exists()

        with open(out_path) as f:
            data = json.load(f)
        assert "cell_type_ari" in data
        assert data["cell_type_ari"] == 0.85
        assert "perturbation_pcc" in data
        assert "grn_auroc" in data


class TestBioValidationResultToCSV:
    """test_biovalidation_result_to_csv: CSV export"""

    def test_to_csv(self, sample_result, tmp_path):
        out_path = tmp_path / "bio_results.csv"
        returned = sample_result.to_csv(out_path)
        assert returned == out_path
        assert out_path.exists()

        with open(out_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "cell_type_ari" in header
            rows = list(reader)
            assert len(rows) == 1  # single-row table


# ---------------------------------------------------------------------------
# Tests — BioValidationPipeline with synthetic data
# ---------------------------------------------------------------------------


class TestBioValidationPipeline:
    """test_biovalidation_pipeline: pipeline with synthetic data"""

    def test_pipeline_creation(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        assert pipeline.model_adapter is mock_adapter
        assert pipeline.optimized_adapter is None
        assert "cell_type_annotation" in pipeline.DIMENSIONS

    def test_run_all_returns_result(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_all()
        assert isinstance(result, BioValidationResult)
        assert result.cell_type is not None
        assert result.perturbation is not None
        assert result.grn is not None
        assert result.embedding_fidelity is not None
        assert "model_name" in result.metadata

    def test_run_cell_type_annotation(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_cell_type_annotation()
        assert isinstance(result, CellTypeResult)
        assert hasattr(result, "ari")
        assert hasattr(result, "nmi")
        assert hasattr(result, "f1_macro")

    def test_run_perturbation_prediction(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_perturbation_prediction()
        # Can be single PerturbationResult or list
        if isinstance(result, list):
            assert all(isinstance(r, PerturbationResult) for r in result)
        else:
            assert isinstance(result, PerturbationResult)

    def test_run_grn_inference(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_grn_inference()
        assert isinstance(result, GeneNetworkResult)
        assert 0.0 <= result.auroc <= 1.0
        assert 0.0 <= result.auprc <= 1.0

    def test_run_drug_response(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_drug_response()
        assert isinstance(result, dict)
        assert "f1" in result
        assert "auroc" in result
        assert "accuracy" in result

    def test_run_knockout_validation(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_knockout_validation(gene_name="TP53")
        assert isinstance(result, dict)
        assert "consistency_score" in result
        assert "gene_name" in result
        assert result["gene_name"] == "TP53"

    def test_run_embedding_fidelity(self, mock_adapter):
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)
        result = pipeline.run_embedding_fidelity()
        assert isinstance(result, EmbeddingFidelityResult)
        assert 0.0 <= result.knn_accuracy <= 1.0
        assert 0.0 <= result.trustworthiness <= 1.0
        assert 0.0 <= result.continuity <= 1.0

    def test_knockout_reproducibility(self, mock_adapter):
        """Two calls to run_knockout_validation with the same gene must return identical results.

        Uses patch.object with wraps set to the bound method captured *before*
        patching so the real code executes and call_count is tracked.
        """
        pipeline = BioValidationPipeline(model_adapter=mock_adapter)

        # Capture the bound method BEFORE patching to avoid recursion
        original_method = pipeline.run_knockout_validation

        with patch.object(
            BioValidationPipeline, "run_knockout_validation",
            wraps=original_method,
        ) as mocked:
            result1 = pipeline.run_knockout_validation(gene_name="TP53")
            result2 = pipeline.run_knockout_validation(gene_name="TP53")

            # The real function was invoked both times
            assert mocked.call_count == 2

        # Both calls return plain dicts (not MagicMock), proving the real code ran
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

        # Deterministic seed ⇒ identical numeric results
        assert result1["consistency_score"] == result2["consistency_score"]
        assert result1["pcc"] == result2["pcc"]
        assert result1["spearman_rho"] == result2["spearman_rho"]
        assert result1["direction_match"] == result2["direction_match"]
        assert result1["gene_name"] == result2["gene_name"] == "TP53"


# ---------------------------------------------------------------------------
# Tests — Synthetic data helpers
# ---------------------------------------------------------------------------


class TestSyntheticDataHelpers:
    """test_synthetic_data_helpers: helper functions produce correct shapes"""

    def test_generate_synthetic_expression(self):
        expr, labels = _generate_synthetic_expression(n_cells=100, n_genes=50)
        assert expr.shape == (100, 50)
        assert labels.shape == (100,)
        assert expr.dtype == np.float32

    def test_generate_synthetic_perturbation(self):
        pred, truth = _generate_synthetic_perturbation(n_conditions=10, n_genes=50)
        assert pred.shape == (10, 50)
        assert truth.shape == (10, 50)

    def test_generate_synthetic_grn(self):
        pred, true = _generate_synthetic_grn(n_genes=20)
        assert pred.shape == (20, 20)
        assert true.shape == (20, 20)
