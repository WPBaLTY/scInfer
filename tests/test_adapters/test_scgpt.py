"""Tests for ScGPT Adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scinfer.adapters.scgpt import ScGPTAdapter, _value_binning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scgpt_config():
    """Minimal scGPT config for testing."""
    return {
        "model": {
            "n_layers": 2,
            "n_heads": 4,
            "embed_dim": 64,
            "hidden_dim": 128,
            "vocab_size": 1000,
            "max_seq_len": 100,
            "n_bins": 64,
            "use_batch_labels": True,
            "batch_size": 8,
        },
        "dtype": "float32",
    }


@pytest.fixture
def scgpt_adapter(scgpt_config):
    """ScGPTAdapter instance (no model loaded)."""
    return ScGPTAdapter(scgpt_config)


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
        # Fallback: mock AnnData
        adata = MagicMock()
        rng = np.random.default_rng(42)
        adata.X = rng.standard_normal((10, 20)).astype(np.float32)
        adata.var_names = [f"GENE_{i}" for i in range(20)]
        return adata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScGPTAdapterCreation:
    """test_scgpt_adapter_creation: 创建适配器实例"""

    def test_scgpt_adapter_creation(self, scgpt_config):
        adapter = ScGPTAdapter(scgpt_config)
        assert adapter.model_name == "scgpt"
        assert adapter.vocab_size == 1000
        assert adapter.max_seq_len == 100
        assert adapter.n_bins == 64
        assert adapter.embed_dim == 64
        assert adapter.n_layers == 2
        assert adapter.n_heads == 4


class TestScGPTPreprocess:
    """test_scgpt_preprocess: 测试Value Binning预处理"""

    def test_scgpt_preprocess(self, scgpt_adapter, synthetic_adata):
        result = scgpt_adapter.preprocess(synthetic_adata)
        assert "gene_ids" in result
        assert "values" in result
        assert "batch_labels" in result
        # Check shapes
        n_cells = 10
        seq_len = min(20, scgpt_adapter.max_seq_len)
        assert result["gene_ids"].shape == (n_cells, seq_len)
        assert result["values"].shape == (n_cells, seq_len)
        assert result["batch_labels"].shape == (n_cells,)
        # Check dtypes
        assert result["gene_ids"].dtype == torch.long
        assert result["values"].dtype == torch.long
        # Value binning: bin ids should be in [0, n_bins)
        assert result["values"].min() >= 0
        assert result["values"].max() < scgpt_adapter.n_bins


class TestScGPTAttentionMask:
    """test_scgpt_attention_mask: 测试动态已知/未知基因掩码生成"""

    def test_scgpt_attention_mask_no_known_mask(self, scgpt_adapter):
        # Without known_mask: fully bidirectional (all ones)
        mask = scgpt_adapter.generate_attention_mask(10)
        assert mask.shape == (10, 10)
        assert mask.all()

    def test_scgpt_attention_mask_with_known_mask(self, scgpt_adapter):
        # With known_mask: dynamic known/unknown pattern
        batch_size = 2
        seq_len = 8
        known_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        known_mask[:, :4] = True  # first 4 genes known

        attn = scgpt_adapter.generate_attention_mask(
            seq_len, known_mask=known_mask, batch_size=batch_size
        )
        assert attn.shape == (batch_size, seq_len, seq_len)
        # Known positions (j<4) should be attendable by all
        assert attn[:, :, :4].all()
        # Unknown positions (j>=4) should NOT be attendable
        assert not attn[:, :, 4:].any()


class TestScGPTSupportedOptimizations:
    """test_scgpt_supported_optimizations: 验证支持的优化列表"""

    def test_scgpt_supported_optimizations(self, scgpt_adapter):
        opts = scgpt_adapter.supported_optimizations
        assert isinstance(opts, list)
        assert "quantization" in opts
        assert "flash_attention" in opts
        assert "compilation" in opts
        assert "batch_scheduling" in opts


class TestScGPTModelType:
    """test_scgpt_model_type: 验证model_type为'hybrid'"""

    def test_scgpt_model_type(self, scgpt_adapter):
        assert scgpt_adapter.model_type == "hybrid"


class TestScGPTFromPretrained:
    """test_scgpt_from_pretrained: 测试from_pretrained类方法（mock）"""

    def test_scgpt_from_pretrained(self, scgpt_config):
        with patch.object(ScGPTAdapter, "load_model") as mock_load:
            adapter = ScGPTAdapter.from_pretrained(
                checkpoint_path=None, config=scgpt_config, device="cpu"
            )
            mock_load.assert_called_once_with(None)
            assert isinstance(adapter, ScGPTAdapter)
            assert adapter._device == torch.device("cpu")


class TestValueBinning:
    """Additional tests for the _value_binning helper."""

    def test_equal_width_binning(self):
        values = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        bins = _value_binning(values, n_bins=5, bin_method="equal_width")
        assert bins.shape == values.shape
        assert bins.dtype == np.int64
        assert bins.min() == 0
        assert bins.max() == 4

    def test_equal_freq_binning(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal((3, 50)).astype(np.float32)
        bins = _value_binning(values, n_bins=10, bin_method="equal_freq")
        assert bins.shape == values.shape
        assert bins.min() >= 0
        assert bins.max() < 10

    def test_constant_values(self):
        values = np.ones((2, 5), dtype=np.float32)
        bins = _value_binning(values, n_bins=8)
        # All same value → all zeros
        assert (bins == 0).all()

    def test_invalid_method_raises(self):
        values = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown bin_method"):
            _value_binning(values, bin_method="invalid")
