"""Tests for Geneformer Adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from scinfer.adapters.geneformer import GeneformerAdapter, _rank_encode, _SCALE_PRESETS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geneformer_config():
    """Minimal Geneformer config for testing."""
    return {
        "model": {
            "scale": "10M",
            "vocab_size": 5000,
            "batch_size": 8,
        },
        "dtype": "float32",
    }


@pytest.fixture
def geneformer_adapter(geneformer_config):
    """GeneformerAdapter instance (no model loaded)."""
    return GeneformerAdapter(geneformer_config)


@pytest.fixture
def synthetic_adata():
    """Synthetic AnnData-like object for testing."""
    try:
        import anndata as ad
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 15)).astype(np.float32)
        # Make some values zero to test PAD handling
        X[0, 0] = 0.0
        X[1, 1] = 0.0
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE_{i}" for i in range(15)]
        return adata
    except ImportError:
        adata = MagicMock()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((8, 15)).astype(np.float32)
        X[0, 0] = 0.0
        X[1, 1] = 0.0
        adata.X = X
        adata.var_names = [f"GENE_{i}" for i in range(15)]
        return adata


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGeneformerAdapterCreation:
    """test_geneformer_adapter_creation: 创建适配器实例"""

    def test_geneformer_adapter_creation(self, geneformer_config):
        adapter = GeneformerAdapter(geneformer_config)
        assert adapter.model_name == "geneformer"
        assert adapter.model_type == "encoder_only"
        assert adapter.vocab_size == 5000
        # 10M preset
        assert adapter.n_layers == 6
        assert adapter.n_heads == 6
        assert adapter.embed_dim == 256


class TestGeneformerRankEmbedding:
    """test_geneformer_rank_embedding: 测试Rank Embedding预处理"""

    def test_geneformer_preprocess(self, geneformer_adapter, synthetic_adata):
        result = geneformer_adapter.preprocess(synthetic_adata)
        assert "input_ids" in result
        assert "attention_mask" in result
        # CLS token prepended: seq_len = max_rank + 1
        n_cells = 8
        expected_seq_len = geneformer_adapter.max_rank + 1
        assert result["input_ids"].shape == (n_cells, expected_seq_len)
        assert result["attention_mask"].shape == (n_cells, expected_seq_len)
        # First position should be CLS token
        assert (result["input_ids"][:, 0] == GeneformerAdapter.CLS_TOKEN_ID).all()
        # CLS position should have attention=1
        assert (result["attention_mask"][:, 0] == 1).all()

    def test_rank_encode_helper(self):
        """Test _rank_encode function directly."""
        expression = np.array([[3.0, 1.0, 0.0, 2.0]], dtype=np.float32)
        gene_names = ["A", "B", "C", "D"]
        token_ids, attn_mask = _rank_encode(
            expression, max_rank=4, gene_names=gene_names
        )
        # Genes sorted by expression desc: A(3), D(2), B(1), C(0→PAD)
        assert token_ids.shape == (1, 4)
        # First token should be gene A (highest expression)
        assert token_ids[0, 0] != 0  # non-zero token
        # Last position should be PAD (zero expression gene C)
        assert attn_mask[0, -1] == 0  # C has 0 expression → PAD


class TestGeneformerAttentionMask:
    """test_geneformer_attention_mask: 测试PAD token掩码"""

    def test_geneformer_attention_mask_no_padding(self, geneformer_adapter):
        mask = geneformer_adapter.generate_attention_mask(5)
        assert mask.shape == (5, 5)
        assert mask.all()

    def test_geneformer_attention_mask_with_input_ids(self, geneformer_adapter):
        input_ids = torch.tensor([[1, 3, 5, 0, 0], [2, 4, 0, 0, 0]])
        mask = geneformer_adapter.generate_attention_mask(5, input_ids=input_ids)
        # Shape: (batch, 1, 1, seq_len) for broadcasting
        assert mask.shape == (2, 1, 1, 5)
        # PAD positions (id=0) should be blocked (False)
        assert not mask[0, 0, 0, 3].item()  # PAD
        assert not mask[0, 0, 0, 4].item()  # PAD
        # Non-PAD positions should be attendable (True)
        assert mask[0, 0, 0, 0].item()
        assert mask[0, 0, 0, 1].item()


class TestGeneformerModelSizes:
    """test_geneformer_model_sizes: 测试3种模型规模切换(10M/38M/316M)"""

    @pytest.mark.parametrize("scale,expected_layers,expected_dim", [
        ("10M", 6, 256),
        ("38M", 12, 768),
        ("316M", 20, 1280),
    ])
    def test_geneformer_model_sizes(self, scale, expected_layers, expected_dim):
        config = {"model": {"scale": scale, "vocab_size": 5000}}
        adapter = GeneformerAdapter(config)
        assert adapter.n_layers == expected_layers
        assert adapter.embed_dim == expected_dim
        assert adapter.max_seq_len == _SCALE_PRESETS[scale]["max_seq_len"]


class TestGeneformerSupportedOptimizations:
    """test_geneformer_supported_optimizations: 包含'distillation'"""

    def test_geneformer_supported_optimizations(self, geneformer_adapter):
        opts = geneformer_adapter.supported_optimizations
        assert isinstance(opts, list)
        assert "quantization" in opts
        assert "flash_attention" in opts
        assert "compilation" in opts
        assert "batch_scheduling" in opts
        assert "distillation" in opts


class TestGeneformerFromPretrained:
    """test_geneformer_from_pretrained: 测试from_pretrained类方法（mock）"""

    def test_geneformer_from_pretrained(self):
        config = {"model": {"scale": "10M"}}
        with patch.object(GeneformerAdapter, "load_model") as mock_load:
            adapter = GeneformerAdapter.from_pretrained(
                checkpoint_path=None, config=config, device="cpu", scale="10M"
            )
            mock_load.assert_called_once_with(None)
            assert isinstance(adapter, GeneformerAdapter)
            assert adapter.n_layers == 6  # 10M preset
