"""Tests for UCE Adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scinfer.adapters.uce import UCEAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uce_config():
    """Minimal UCE config for testing."""
    return {
        "model": {
            "n_layers": 2,
            "n_heads": 4,
            "embed_dim": 64,
            "hidden_dim": 128,
            "vocab_size": 1000,
            "max_seq_len": 100,
            "protein_embedding_dim": 64,
            "batch_size": 8,
        },
        "dtype": "float32",
    }


@pytest.fixture
def uce_adapter(uce_config):
    """UCEAdapter instance (no model loaded)."""
    return UCEAdapter(uce_config)


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


class TestUCEAdapterCreation:
    """test_uce_adapter_creation: create adapter instance"""

    def test_uce_adapter_creation(self, uce_config):
        adapter = UCEAdapter(uce_config)
        assert adapter.model_name == "uce"
        assert adapter.vocab_size == 1000
        assert adapter.max_seq_len == 100
        assert adapter.embed_dim == 64
        assert adapter.n_layers == 2
        assert adapter.n_heads == 4


class TestUCEModelType:
    """test_uce_model_type: encoder_decoder architecture"""

    def test_uce_model_type(self, uce_adapter):
        assert uce_adapter.model_type == "encoder_decoder"


class TestUCESupportedOptimizations:
    """test_uce_supported_optimizations: verify optimization list"""

    def test_uce_supported_optimizations(self, uce_adapter):
        opts = uce_adapter.supported_optimizations
        assert isinstance(opts, list)
        assert "quantization" in opts
        assert "flash_attention" in opts
        assert "compilation" in opts
        assert "batch_scheduling" in opts


class TestUCEPreprocess:
    """test_uce_preprocess: protein embedding preprocessing"""

    def test_uce_preprocess(self, uce_adapter, synthetic_adata):
        result = uce_adapter.preprocess(synthetic_adata)
        assert "protein_embeddings" in result
        assert "expression" in result
        assert "gene_mask" in result
        # Check shapes
        n_cells = 10
        seq_len = min(20, uce_adapter.max_seq_len)
        protein_dim = uce_adapter.protein_embedding_dim
        assert result["protein_embeddings"].shape == (n_cells, seq_len, protein_dim)
        assert result["expression"].shape == (n_cells, seq_len)
        assert result["gene_mask"].shape == (n_cells, seq_len)
        # Check dtypes
        assert result["protein_embeddings"].dtype == torch.float32
        assert result["expression"].dtype == torch.float32
        assert result["gene_mask"].dtype == torch.bool

    def test_uce_preprocess_gene_mask_all_true(self, uce_adapter, synthetic_adata):
        """When n_genes <= max_seq_len, all gene_mask values should be True."""
        result = uce_adapter.preprocess(synthetic_adata)
        # 20 genes < 100 max_seq_len → all True
        assert result["gene_mask"].all()


class TestUCEAttentionMask:
    """test_uce_attention_mask: standard Transformer attention"""

    def test_attention_mask_no_padding(self, uce_adapter):
        mask = uce_adapter.generate_attention_mask(10)
        assert mask.shape == (10, 10)
        assert mask.all()

    def test_attention_mask_with_pad_mask(self, uce_adapter):
        pad_mask = torch.ones(2, 8, dtype=torch.bool)
        pad_mask[:, 6:] = False  # last 2 are padding
        attn = uce_adapter.generate_attention_mask(8, pad_mask=pad_mask)
        assert attn.shape == (2, 1, 8, 8)
        # Padding positions should block attention
        assert not attn[0, 0, 0, 6].item()
        assert not attn[0, 0, 0, 7].item()
        # Valid positions should allow attention
        assert attn[0, 0, 0, 0].item()


class TestUCEFromPretrained:
    """test_uce_from_pretrained: from_pretrained class method (mock)"""

    def test_from_pretrained(self, uce_config):
        with patch.object(UCEAdapter, "load_model") as mock_load:
            adapter = UCEAdapter.from_pretrained(
                checkpoint_path=None, config=uce_config, device="cpu"
            )
            mock_load.assert_called_once_with(None)
            assert isinstance(adapter, UCEAdapter)
            assert adapter._device == torch.device("cpu")
