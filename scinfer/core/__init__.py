"""scinfer.core — Core infrastructure for the scInfer framework.

This module provides the foundational building blocks:
- ``Config``: YAML-based hierarchical configuration system.
- ``ModelRegistry`` / ``EngineRegistry``: decorator-based registries for models and engines.
- Type enumerations used throughout the framework.
"""

from scinfer.core.config import Config
from scinfer.core.registry import EngineRegistry, ModelRegistry, register_engine, register_model
from scinfer.core.types import ModelType, OptimizationType

__all__ = [
    "Config",
    "ModelRegistry",
    "EngineRegistry",
    "register_model",
    "register_engine",
    "ModelType",
    "OptimizationType",
]
