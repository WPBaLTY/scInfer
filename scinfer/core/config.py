"""Hierarchical YAML-based configuration system.

Supports loading from multiple YAML files with deep-merge semantics,
dot-notation access, and required-field validation.

Example
-------
>>> cfg = Config.from_files("configs/default.yaml", "configs/models/scgpt.yaml")
>>> cfg.model.name
'scgpt'
>>> cfg.get("global.seed", 0)
42
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from loguru import logger


class Config:
    """Immutable-after-build configuration container with dot-notation access.

    Parameters
    ----------
    data : dict
        The raw configuration dictionary.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, "_data", data or {})

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_files(cls, *paths: Union[str, Path]) -> "Config":
        """Load and deep-merge multiple YAML config files (later files win).

        Parameters
        ----------
        *paths : str | Path
            One or more YAML file paths.  Files are merged left-to-right;
            values in later files override earlier ones.

        Returns
        -------
        Config
            Merged configuration object.
        """
        merged: Dict[str, Any] = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                logger.warning("Config file not found, skipping: {}", p)
                continue
            with open(p, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if raw is None:
                logger.debug("Empty config file: {}", p)
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"Config file must contain a YAML mapping, got {type(raw)}: {p}")
            merged = _deep_merge(merged, raw)
            logger.debug("Loaded config from {}", p)
        return cls(merged)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config from a plain dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        Config
        """
        return cls(copy.deepcopy(data))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by dot-separated key path.

        Parameters
        ----------
        key : str
            Dot-separated path, e.g. ``"model.name"``.
        default : Any
            Returned when the key does not exist.

        Returns
        -------
        Any
        """
        keys = key.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def require(self, key: str) -> Any:
        """Like ``get`` but raises ``KeyError`` if the key is missing or ``None``.

        Parameters
        ----------
        key : str
            Dot-separated path.

        Returns
        -------
        Any

        Raises
        ------
        KeyError
            If the key is absent or its value is ``None``.
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required config key is missing or null: {key}")
        return value

    def validate_required(self, keys: Sequence[str]) -> None:
        """Check that all *keys* are present and non-null.

        Parameters
        ----------
        keys : sequence of str
            Dot-separated key paths to validate.

        Raises
        ------
        KeyError
            If any required key is missing.
        """
        missing = [k for k in keys if self.get(k) is None]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")

    # ------------------------------------------------------------------
    # Mutation (returns new Config — original is immutable)
    # ------------------------------------------------------------------

    def merge(self, other: Union["Config", Dict[str, Any]]) -> "Config":
        """Return a new Config that is the deep-merge of *self* and *other*.

        Parameters
        ----------
        other : Config | dict
            Override values.

        Returns
        -------
        Config
        """
        other_data = other._data if isinstance(other, Config) else other
        return Config(_deep_merge(copy.deepcopy(self._data), copy.deepcopy(other_data)))

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying dictionary.

        Returns
        -------
        dict
        """
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    # Dot-notation attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        # Try exact match first, then strip trailing underscore (for reserved words like global)
        if name in data:
            value = data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        # Allow accessing Python reserved words via trailing-underscore convention
        if name.endswith("_") and name[:-1] in data:
            value = data[name[:-1]]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is immutable after construction. Use .merge() to override.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __str__(self) -> str:
        return yaml.dump(self._data, default_flow_style=False, sort_keys=False).rstrip()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        return NotImplemented


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*).

    - Dicts are merged recursively.
    - All other types in *override* replace the value in *base*.

    Parameters
    ----------
    base : dict
        Base configuration dictionary (mutated in place).
    override : dict
        Override dictionary whose values take precedence.

    Returns
    -------
    dict
        The merged dictionary (same object as *base*).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
