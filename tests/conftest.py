"""Shared pytest fixtures for scInfer tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample config files.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture.

    Returns
    -------
    Path
        Path to the temporary config directory.
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Write a minimal default config
    default_config = {
        "global": {"seed": 42, "device": "cpu", "verbose": "DEBUG"},
        "model": {"name": "scgpt", "batch_size": 32},
        "optimization": {"enabled": False, "engines": []},
    }
    with open(config_dir / "default.yaml", "w") as f:
        yaml.dump(default_config, f)

    # Write a model-specific config
    model_config = {
        "model": {
            "name": "scgpt",
            "type": "encoder_only",
            "n_layers": 12,
            "embed_dim": 512,
        },
    }
    with open(config_dir / "scgpt.yaml", "w") as f:
        yaml.dump(model_config, f)

    return config_dir


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Return a sample configuration dictionary for testing.

    Returns
    -------
    dict
        Sample configuration.
    """
    return {
        "global": {"seed": 42, "device": "cpu", "verbose": "DEBUG"},
        "model": {
            "name": "scgpt",
            "type": "encoder_only",
            "n_layers": 12,
            "embed_dim": 512,
            "batch_size": 32,
        },
        "optimization": {
            "enabled": True,
            "engines": ["quantization"],
            "quantization": {"method": "int8"},
        },
    }
