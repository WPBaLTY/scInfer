"""Tests for the model and engine registries (scinfer.core.registry)."""

from __future__ import annotations

import pytest

from scinfer.core.registry import EngineRegistry, ModelRegistry, _BaseRegistry, register_engine, register_model


class TestBaseRegistry:
    """Tests for the generic _BaseRegistry class."""

    def test_register_and_get(self):
        """Register a class and retrieve it by name."""
        registry = _BaseRegistry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        assert registry.get("my_class") is MyClass

    def test_duplicate_registration_same_class(self):
        """Re-registering the same class is a no-op."""
        registry = _BaseRegistry("test")

        @registry.register("my_class")
        class MyClass:
            pass

        # Re-register same class — should not raise
        @registry.register("my_class")
        class MyClass:  # noqa: F811
            pass

        assert registry.get("my_class") is MyClass

    def test_duplicate_registration_different_class_raises(self):
        """Re-registering a different class under the same name raises."""
        registry = _BaseRegistry("test")

        @registry.register("my_class")
        class MyClassA:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("my_class")
            class MyClassB:
                pass

    def test_get_missing_raises(self):
        """Getting a non-existent name raises KeyError."""
        registry = _BaseRegistry("test")
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list(self):
        """list() returns sorted registered names."""
        registry = _BaseRegistry("test")

        @registry.register("b")
        class B:
            pass

        @registry.register("a")
        class A:
            pass

        assert registry.list() == ["a", "b"]

    def test_contains(self):
        """__contains__ works correctly."""
        registry = _BaseRegistry("test")

        @registry.register("exists")
        class Exists:
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self):
        """__len__ returns the number of registered items."""
        registry = _BaseRegistry("test")
        assert len(registry) == 0

        @registry.register("one")
        class One:
            pass

        assert len(registry) == 1


class TestModelRegistry:
    """Tests for the global ModelRegistry singleton."""

    def test_register_model_decorator(self):
        """@register_model decorator registers in ModelRegistry."""

        @register_model("test_model_xyz")
        class TestModel:
            pass

        assert "test_model_xyz" in ModelRegistry
        assert ModelRegistry.get("test_model_xyz") is TestModel

        # Cleanup
        del ModelRegistry._registry["test_model_xyz"]


class TestEngineRegistry:
    """Tests for the global EngineRegistry singleton."""

    def test_register_engine_decorator(self):
        """@register_engine decorator registers in EngineRegistry."""

        @register_engine("test_engine_xyz")
        class TestEngine:
            pass

        assert "test_engine_xyz" in EngineRegistry
        assert EngineRegistry.get("test_engine_xyz") is TestEngine

        # Cleanup
        del EngineRegistry._registry["test_engine_xyz"]
