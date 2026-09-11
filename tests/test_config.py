"""Tests for the configuration system (scinfer.core.config)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scinfer.core.config import Config, _deep_merge


class TestConfigFromDict:
    """Tests for Config.from_dict()."""

    def test_basic_creation(self, sample_config_dict):
        """Config can be created from a dictionary."""
        cfg = Config.from_dict(sample_config_dict)
        assert cfg.get("model.name") == "scgpt"

    def test_nested_access(self, sample_config_dict):
        """Dot-notation access works for nested keys."""
        cfg = Config.from_dict(sample_config_dict)
        assert cfg.model.name == "scgpt"
        assert cfg.global_.seed == 42  # global is a reserved name, but works

    def test_get_default(self, sample_config_dict):
        """get() returns default when key is missing."""
        cfg = Config.from_dict(sample_config_dict)
        assert cfg.get("nonexistent.key", "default") == "default"

    def test_contains(self, sample_config_dict):
        """__contains__ works for existing and missing keys."""
        cfg = Config.from_dict(sample_config_dict)
        assert "model.name" in cfg
        assert "nonexistent" not in cfg


class TestConfigFromFiles:
    """Tests for Config.from_files()."""

    def test_load_single_file(self, tmp_config_dir):
        """Load a single YAML config file."""
        cfg = Config.from_files(tmp_config_dir / "default.yaml")
        assert cfg.get("global.seed") == 42

    def test_merge_multiple_files(self, tmp_config_dir):
        """Merge multiple YAML files (later files override)."""
        cfg = Config.from_files(
            tmp_config_dir / "default.yaml",
            tmp_config_dir / "scgpt.yaml",
        )
        # model.name should be "scgpt" from both files
        assert cfg.get("model.name") == "scgpt"
        # n_layers should come from scgpt.yaml
        assert cfg.get("model.n_layers") == 12
        # batch_size should come from default.yaml
        assert cfg.get("model.batch_size") == 32

    def test_missing_file_skipped(self, tmp_config_dir):
        """Missing files are skipped with a warning."""
        cfg = Config.from_files(
            tmp_config_dir / "default.yaml",
            tmp_config_dir / "nonexistent.yaml",
        )
        assert cfg.get("global.seed") == 42


class TestConfigValidation:
    """Tests for config validation."""

    def test_require_existing_key(self, sample_config_dict):
        """require() returns value for existing keys."""
        cfg = Config.from_dict(sample_config_dict)
        assert cfg.require("model.name") == "scgpt"

    def test_require_missing_key_raises(self, sample_config_dict):
        """require() raises KeyError for missing keys."""
        cfg = Config.from_dict(sample_config_dict)
        with pytest.raises(KeyError, match="Required config key"):
            cfg.require("nonexistent.key")

    def test_validate_required(self, sample_config_dict):
        """validate_required() passes when all keys present."""
        cfg = Config.from_dict(sample_config_dict)
        cfg.validate_required(["model.name", "global.seed"])

    def test_validate_required_fails(self, sample_config_dict):
        """validate_required() raises when keys are missing."""
        cfg = Config.from_dict(sample_config_dict)
        with pytest.raises(KeyError, match="Missing required"):
            cfg.validate_required(["model.name", "nonexistent.key"])


class TestConfigMerge:
    """Tests for Config.merge()."""

    def test_merge_returns_new_config(self, sample_config_dict):
        """merge() returns a new Config without modifying the original."""
        cfg = Config.from_dict(sample_config_dict)
        override = {"model": {"batch_size": 128}}
        merged = cfg.merge(override)
        assert merged.get("model.batch_size") == 128
        assert cfg.get("model.batch_size") == 32  # original unchanged

    def test_immutability(self, sample_config_dict):
        """Config is immutable — setting attributes raises."""
        cfg = Config.from_dict(sample_config_dict)
        with pytest.raises(AttributeError, match="immutable"):
            cfg.new_attr = "value"


class TestDeepMerge:
    """Tests for the _deep_merge helper."""

    def test_simple_merge(self):
        """Simple dict merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Nested dict merge preserves un-overridden keys."""
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3}}
