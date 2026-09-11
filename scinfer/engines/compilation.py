"""Compilation optimisation engine.

Uses ``torch.compile`` (PyTorch 2.0+) to fuse operations, eliminate
intermediate allocations, and generate optimised GPU kernels.  This
engine wraps the model with ``torch.compile`` using configurable modes
(``default``, ``reduce-overhead``, ``max-autotune``).

Expected implementation phases:
- Phase 4: torch.compile integration
- Phase 4: ONNX export support
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from loguru import logger

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine
from scinfer.core.registry import register_engine


# ---------------------------------------------------------------------------
# Torch version check
# ---------------------------------------------------------------------------

def _torch_supports_compile() -> bool:
    """Return True if the installed PyTorch version supports torch.compile."""
    try:
        ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
        return ver >= (2, 0)
    except (ValueError, AttributeError):
        return False


@register_engine("compilation")
class CompilationEngine(BaseOptimizationEngine):
    """Inference optimisation engine for graph compilation.

    Leverages ``torch.compile`` to:
    - Fuse element-wise operations into single kernels.
    - Reduce memory allocation overhead.
    - Auto-tune kernel selection for the target GPU.
    """

    # Supported backends and their torch.compile modes
    _BACKEND_MODES = {
        "inductor": "default",
        "cudagraphs": "reduce-overhead",
        "max-autotune": "max-autotune",
    }

    def __init__(self) -> None:
        """Initialise the compilation engine."""
        self._torch_compile_available = _torch_supports_compile()
        self._compiled_models: Dict[str, nn.Module] = {}
        self._compile_times: Dict[str, float] = {}
        logger.debug(
            "CompilationEngine initialised — torch.compile available: {}",
            self._torch_compile_available,
        )

    # ------------------------------------------------------------------
    # BaseOptimizationEngine interface
    # ------------------------------------------------------------------

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Compile the model with ``torch.compile``.

        Parameters
        ----------
        model : torch.nn.Module
            The model to compile.
        config : dict
            Compilation configuration (mode, fullgraph, dynamic, etc.).

        Returns
        -------
        torch.nn.Module
            The compiled model.
        """
        backend = config.get("backend", "inductor")
        fullgraph = config.get("fullgraph", False)
        dynamic = config.get("dynamic", False)

        logger.info("Applying compilation: backend={}", backend)

        if not self._torch_compile_available:
            logger.warning(
                "torch.compile requires PyTorch >= 2.0. "
                "Falling back to torch.jit.trace."
            )
            return self._apply_jit_trace(model, config)

        # Determine compile mode
        mode = self._BACKEND_MODES.get(backend, "default")

        try:
            t0 = time.perf_counter()
            compiled = torch.compile(
                model,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )

            # Eagerly test the compiled model to catch lazy compilation errors
            device = (
                next(compiled.parameters()).device
                if list(compiled.parameters())
                else torch.device("cpu")
            )
            test_input = self._create_example_input(compiled, device)
            if test_input is not None:
                compiled.eval()
                with torch.no_grad():
                    _ = compiled(test_input)

            compile_time = time.perf_counter() - t0

            model_id = f"compile_{backend}_{id(model) % 10000}"
            self._compiled_models[model_id] = compiled
            self._compile_times[model_id] = compile_time

            logger.info(
                "torch.compile completed in {:.2f}s (mode={}, backend={})",
                compile_time, mode, backend,
            )
            return compiled

        except Exception as e:
            logger.warning(
                "torch.compile failed ({}). Falling back to torch.jit.trace.", e
            )
            return self._apply_jit_trace(model, config)

    def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark compiled model speed.

        Parameters
        ----------
        model : torch.nn.Module
            The compiled model.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Metrics including throughput, latency, and compilation time.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        if isinstance(input_data, torch.Tensor) and input_data.device != device:
            input_data = input_data.to(device)

        model.eval()

        # Warm-up (important for torch.compile which compiles on first call)
        try:
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_data)
        except Exception as e:
            logger.warning("Model execution failed during warm-up: {}", e)
            return {
                "latency_ms": 0.0,
                "latency_std_ms": 0.0,
                "throughput_fps": 0.0,
                "peak_memory_mb": 0.0,
                "model_size_mb": 0.0,
                "error": str(e),
            }

        # Latency measurement
        n_runs = 50
        times: list[float] = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        latency_arr = np.array(times) * 1000.0  # ms
        latency_ms = float(latency_arr.mean())
        throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        # Peak GPU memory
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Model size
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = params_bytes / (1024 ** 2)

        metrics = {
            "latency_ms": round(latency_ms, 4),
            "latency_std_ms": round(float(latency_arr.std()), 4),
            "throughput_fps": round(throughput_fps, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "model_size_mb": round(model_size_mb, 2),
        }

        logger.info(
            "Compilation benchmark: latency={:.3f} ms, throughput={:.1f} fps",
            latency_ms, throughput_fps,
        )
        return metrics

    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check if torch.compile is compatible with the given adapter.

        torch.compile is broadly compatible with all PyTorch models,
        though some dynamic control flow may cause graph breaks.

        Parameters
        ----------
        adapter : BaseModelAdapter
            The model adapter to check.

        Returns
        -------
        bool
            ``True`` if the adapter's model can be compiled.
        """
        # torch.compile is compatible with all PyTorch-based models
        return True

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Validate that the model has no graph breaks (if fullgraph=True).

        Performs a trial compilation and checks for graph breaks.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        config : dict
            Compilation configuration.

        Returns
        -------
        bool
            ``True`` if validation passes.
        """
        fullgraph = config.get("fullgraph", False)

        if not self._torch_compile_available:
            logger.warning("torch.compile not available; validation skipped")
            return True

        try:
            # Create a dummy input for tracing
            device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )
            # Try to infer input shape from first Linear layer
            example_input = self._create_example_input(model, device)
            if example_input is None:
                logger.info("Could not create example input for validation")
                return True

            model.eval()
            if fullgraph:
                # Use torch._dynamo.check_graph_breaks if available
                try:
                    import torch._dynamo as dynamo  # type: ignore
                    reset = dynamo.reset()
                    with torch.no_grad():
                        compiled = torch.compile(model, mode="default", fullgraph=True)
                        _ = compiled(example_input)
                    logger.info("fullgraph validation passed — no graph breaks")
                except Exception as e:
                    logger.warning("fullgraph validation found graph breaks: {}", e)
                    return False
            else:
                # Just do a trial run
                with torch.no_grad():
                    compiled = torch.compile(model, mode="default")
                    _ = compiled(example_input)
                logger.info("Trial compilation succeeded")

            return True

        except Exception as e:
            logger.warning("Validation failed: {}", e)
            return False

    # ------------------------------------------------------------------
    # Fallback: JIT trace
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_jit_trace(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Fallback: compile via torch.jit.trace."""
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

        example = config.get("example_input")
        if example is None:
            # Try to create a dummy input
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    example = torch.randn(1, module.in_features, device=device)
                    break

        if example is None:
            logger.warning(
                "Cannot create example input for JIT trace. "
                "Returning original model."
            )
            return model

        try:
            model.eval()
            traced = torch.jit.trace(model, example)
            logger.info("torch.jit.trace succeeded as fallback")
            return traced
        except Exception as e:
            logger.warning("torch.jit.trace also failed: {}. Returning original model.", e)
            return model

    @staticmethod
    def _create_example_input(model: nn.Module, device: torch.device) -> torch.Tensor | None:
        """Create a dummy input tensor by inspecting the model's first Linear layer."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return torch.randn(2, module.in_features, device=device)
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def revert(self, model: nn.Module) -> nn.Module:
        """Revert optimization - returns model unchanged for non-reversible engines."""
        logger.info(f"{self.engine_name}: revert not applicable, returning model as-is")
        return model

    @property
    def engine_name(self) -> str:
        """Return the engine identifier."""
        return "compilation"

    @property
    def optimization_type(self) -> str:
        """Return the optimization category."""
        return "compilation"
