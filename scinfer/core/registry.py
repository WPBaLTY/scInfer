"""Decorator-based registries for models and optimisation engines.

The registries follow a simple pattern: classes are registered via decorators
and can later be retrieved by name.  This keeps the framework open for
extension without modifying core code.

Example
-------
>>> @register_model("scgpt")
... class ScGPTAdapter(BaseModelAdapter):
...     ...
>>> ModelRegistry.get("scgpt")
<class 'ScGPTAdapter'>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from loguru import logger


class _BaseRegistry:
    """Generic name → class registry with decorator support."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._registry: Dict[str, Type[Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str) -> Any:
        """Return a decorator that registers a class under *name*.

        Parameters
        ----------
        name : str
            Unique identifier for the class.

        Returns
        -------
        callable
            Class decorator.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """

        def decorator(cls: Type[Any]) -> Type[Any]:
            if name in self._registry:
                existing = self._registry[name]
                if existing is not cls and existing.__name__ != cls.__name__:
                    raise ValueError(
                        f"{self._kind.capitalize()} '{name}' is already registered "
                        f"to {existing.__name__}. Cannot re-register to {cls.__name__}."
                    )
                # Same class or same class name — allow re-registration (overwrite)
                logger.debug("{} '{}' already registered to same class name, updating.", self._kind, name)
            self._registry[name] = cls
            logger.info("Registered {} '{}'", self._kind, name)
            return cls

        return decorator

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Type[Any]:
        """Retrieve a registered class by name.

        Parameters
        ----------
        name : str
            The registered name.

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        KeyError
            If *name* is not found.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"{self._kind.capitalize()} '{name}' not found. Available: {available}"
            )
        return self._registry[name]

    def get_optional(self, name: str) -> Optional[Type[Any]]:
        """Like ``get`` but returns ``None`` instead of raising.

        Parameters
        ----------
        name : str
            The registered name.

        Returns
        -------
        type | None
        """
        return self._registry.get(name)

    def list(self) -> List[str]:
        """Return sorted list of all registered names.

        Returns
        -------
        list of str
        """
        return sorted(self._registry.keys())

    def all(self) -> Dict[str, Type[Any]]:
        """Return a copy of the full registry mapping.

        Returns
        -------
        dict
        """
        return dict(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"{self._kind.capitalize()}Registry(registered={self.list()})"


# ---------------------------------------------------------------------------
# Singleton registries
# ---------------------------------------------------------------------------

ModelRegistry = _BaseRegistry("model")
EngineRegistry = _BaseRegistry("engine")


# ---------------------------------------------------------------------------
# Convenience decorators
# ---------------------------------------------------------------------------


def register_model(name: str) -> Any:
    """Decorator to register a model adapter class.

    Parameters
    ----------
    name : str
        Unique model identifier (e.g. ``"scgpt"``).

    Returns
    -------
    callable
        Class decorator.

    Example
    -------
    >>> @register_model("scgpt")
    ... class ScGPTAdapter(BaseModelAdapter):
    ...     ...
    """
    return ModelRegistry.register(name)


def register_engine(name: str) -> Any:
    """Decorator to register an optimisation engine class.

    Parameters
    ----------
    name : str
        Unique engine identifier (e.g. ``"quantization"``).

    Returns
    -------
    callable
        Class decorator.

    Example
    -------
    >>> @register_engine("quantization")
    ... class QuantizationEngine(BaseOptimizationEngine):
    ...     ...
    """
    return EngineRegistry.register(name)
