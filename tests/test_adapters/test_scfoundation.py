"""Tests for ScFoundation Adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scinfer.adapters.scfoundation import ScFoundationAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scfoundation_config():
    """Minimal scFoundation config for testing."""
    return {
        "model": {
            "n_layers": 2,
            "n_heads": 4,
            "embed_dim": 64,
            "hidden_dim": 128,
            "vocab_size": 1000,
            "max_seq_len": 100,
            "n_bins": 64,
            "batch_size": 8,
        },
        "dtype": "float32",
    }


@pytest.fixture
def scfoundation_adapter(scfoundation_config):
    """ScFoundationAdapter instance (no model loaded)."""
    return ScFoundationAdapter(scfoundation_config)


@pytest.fixture
def synthetic_adata():
    """Synthetic AnnData-like object for testing."""
    try:
        import anndata as ad
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 20)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE_{i}" for i in range(20)]
        return adata
    except ImportError:
        adata = MagicMock()
        rng = np.random.default_rng(42)
        adata.X = rng.standard_normal((10, 20)).astype(np.float32)
        adata.var_names = [f"GENE_{i}" for i in range(20)]
        return adata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScFoundationAdapterCreation:
    """test_scfoundation_adapter_creation: create adapter instance"""

    def test_scfoundation_adapter_creation(self, scfoundation_config):
        adapter = ScFoundationAdapter(scfoundation_config)
        assert adapter.model_name == "scfoundation"
        assert adapter.vocab_size == 1000
        assert adapter.max_seq_len == 100
        assert adapter.embed_dim == 64
        assert adapter.n_layers == 2
        assert adapter.n_heads == 4


class TestScFoundationModelType:
    """test_scfoundation_model_type: encoder_decoder architecture"""

    def test_scfoundation_model_type(self, scfoundation_adapter):
        assert scfoundation_adapter.model_type == "encoder_decoder"


class TestScFoundationSupportedOptimizations:
    """test_scfoundation_supported_optimizations: verify optimization list"""

    def test_scfoundation_supported_optimizations(self, scfoundation_adapter):
        opts = scfoundation_adapter.supported_optimizations
        assert isinstance(opts, list)
        assert "quantization" in opts
        assert "flash_attention" in opts
        assert "compilation" in opts
        assert "batch_scheduling" in opts


class TestScFoundationPreprocess:
    """test_scfoundation_preprocess: raw expression value preprocessing"""

    def test_scfoundation_preprocess(self, scfoundation_adapter, synthetic_adata):
        result = scfoundation_adapter.preprocess(synthetic_adata)
        assert "expression" in result
        assert "gene_ids" in result
        assert "mask" in result
        # Check shapes
        n_cells = 10
        seq_len = min(20, scfoundation_adapter.max_seq_len)
        assert result["expression"].shape == (n_cells, seq_len)
        assert result["gene_ids"].shape == (n_cells, seq_len)
        assert result["mask"].shape == (n_cells, seq_len)
        # Check dtypes
        assert result["expression"].dtype == torch.float32
        assert result["gene_ids"].dtype == torch.long
        assert result["mask"].dtype == torch.bool

    def test_scfoundation_preprocess_mask_all_true(self, scfoundation_adapter, synthetic_adata):
        """When n_genes <= max_seq_len, all mask values should be True."""
        result = scfoundation_adapter.preprocess(synthetic_adata)
        # 20 genes < 100 max_seq_len → all True
        assert result["mask"].all()


class TestScFoundationAttentionMask:
    """test_scfoundation_attention_mask: standard bidirectional attention"""

    def test_attention_mask_no_padding(self, scfoundation_adapter):
        mask = scfoundation_adapter.generate_attention_mask(10)
        assert mask.shape == (10, 10)
        assert mask.all()

    def test_attention_mask_with_pad_mask(self, scfoundation_adapter):
        pad_mask = torch.ones(2, 8, dtype=torch.bool)
        pad_mask[:, 6:] = False  # last 2 are padding
        attn = scfoundation_adapter.generate_attention_mask(8, pad_mask=pad_mask)
        assert attn.shape == (2, 1, 8, 8)
        # Padding positions should block attention
        assert not attn[0, 0, 0, 6].item()
        assert not attn[0, 0, 0, 7].item()
        # Valid positions should allow attention
        assert attn[0, 0, 0, 0].item()


class TestScFoundationFromPretrained:
    """test_scfoundation_from_pretrained: from_pretrained class method (mock)"""

    def test_from_pretrained(self, scfoundation_config):
        with patch.object(ScFoundationAdapter, "load_model") as mock_load:
            adapter = ScFoundationAdapter.from_pretrained(
                checkpoint_path=None, config=scfoundation_config, device="cpu"
            )
            mock_load.assert_called_once_with(None)
            assert isinstance(adapter, ScFoundationAdapter)
            assert adapter._device == torch.device("cpu")
