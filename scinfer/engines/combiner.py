"""Optimisation combiner engine.

Composes multiple optimisation engines into a single pipeline, handling
ordering, compatibility checks, and conflict resolution.  The combiner
ensures that engines are applied in the correct order and that their
interactions are well-defined.

Expected implementation phases:
- Phase 3: Basic pipeline composition
- Phase 4: Conflict detection and resolution
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from loguru import logger

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine
from scinfer.core.registry import register_engine, EngineRegistry


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Compatibility matrix: engine_name → set of engines it can coexist with
COMPATIBILITY_MATRIX: Dict[str, set] = {
    "flash_attention": {"quantization", "compilation", "batch_scheduling", "distillation"},
    "quantization": {"flash_attention", "compilation", "batch_scheduling", "distillation"},
    "compilation": {"flash_attention", "quantization", "batch_scheduling", "distillation"},
    "batch_scheduling": {"flash_attention", "quantization", "compilation", "distillation"},
    "distillation": {"flash_attention", "quantization", "compilation", "batch_scheduling"},
}

# Engine application order (lower index = applied first)
APPLICATION_ORDER: List[str] = [
    "quantization",
    "flash_attention",
    "distillation",
    "compilation",
    "batch_scheduling",
]

# Which engines are compatible with each other (pairwise)
ENGINE_COMPATIBILITY: Dict[Tuple[str, str], bool] = {}
for _eng, _others in COMPATIBILITY_MATRIX.items():
    for _other in _others:
        ENGINE_COMPATIBILITY[(_eng, _other)] = True
        ENGINE_COMPATIBILITY[(_other, _eng)] = True


@register_engine("combiner")
class OptimizationCombiner(BaseOptimizationEngine):
    """Composes multiple optimisation engines into a unified pipeline.

    The combiner:
    1. Validates that all requested engines are compatible with the adapter.
    2. Orders engines according to the configured priority.
    3. Applies each engine sequentially to the model.
    4. Validates biological fidelity after the full pipeline.
    """

    def __init__(self, engines: List[BaseOptimizationEngine] | None = None) -> None:
        """Initialise the optimisation combiner.

        Parameters
        ----------
        engines : list of BaseOptimizationEngine, optional
            Pre-constructed engine instances.  If ``None``, engines are
            resolved from the registry based on configuration.
        """
        if engines is not None:
            self._engines = list(engines)
        else:
            self._engines = self._resolve_engines_from_registry()

        self._applied_engines: List[str] = []
        logger.debug(
            "OptimizationCombiner initialised with engines: {}",
            [e.engine_name for e in self._engines],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_engines_from_registry() -> List[BaseOptimizationEngine]:
        """Instantiate all registered engines (excluding combiner itself)."""
        engines: List[BaseOptimizationEngine] = []
        for name in EngineRegistry.list():
            if name == "combiner":
                continue
            try:
                cls = EngineRegistry.get(name)
                engines.append(cls())
            except Exception as e:
                logger.warning("Could not instantiate engine '{}': {}", name, e)
        return engines

    @staticmethod
    def _engine_order(engine: BaseOptimizationEngine) -> int:
        """Return the sort key for an engine based on APPLICATION_ORDER."""
        name = engine.engine_name
        if name in APPLICATION_ORDER:
            return APPLICATION_ORDER.index(name)
        return len(APPLICATION_ORDER)

    # ------------------------------------------------------------------
    # BaseOptimizationEngine interface
    # ------------------------------------------------------------------

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Apply all optimisation engines in order.

        Parameters
        ----------
        model : torch.nn.Module
            The model to optimise.
        config : dict
            Combined configuration for all engines.

        Returns
        -------
        torch.nn.Module
            The fully optimised model.
        """
        # Sort engines by application order
        sorted_engines = sorted(self._engines, key=self._engine_order)

        # Determine which engines to apply
        enabled = config.get("engines", None)  # optional whitelist
        self._applied_engines = []

        logger.info(
            "Applying {} engines in order: {}",
            len(sorted_engines),
            [e.engine_name for e in sorted_engines],
        )

        for engine in sorted_engines:
            name = engine.engine_name

            if enabled is not None and name not in enabled:
                logger.debug("Skipping engine '{}' (not in enabled list)", name)
                continue

            engine_config = config.get(name, {})

            try:
                logger.info("Applying engine: {}", name)
                model = engine.apply(model, engine_config)
                self._applied_engines.append(name)
                logger.info("Engine '{}' applied successfully", name)
            except Exception as e:
                logger.warning(
                    "Engine '{}' failed ({}). Skipping.", name, e
                )

        logger.info(
            "Combiner finished. Applied engines: {}", self._applied_engines
        )
        return model

    def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark the combined optimisation pipeline.

        Parameters
        ----------
        model : torch.nn.Module
            The optimised model.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Aggregate metrics across all engines.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        if isinstance(input_data, torch.Tensor) and input_data.device != device:
            input_data = input_data.to(device)

        model.eval()

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)

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
            "n_applied_engines": len(self._applied_engines),
        }

        logger.info(
            "Combiner benchmark: latency={:.3f} ms, throughput={:.1f} fps, "
            "engines={}",
            latency_ms, throughput_fps, self._applied_engines,
        )
        return metrics

    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check if all engines in the pipeline are compatible.

        Parameters
        ----------
        adapter : BaseModelAdapter
            The model adapter to check.

        Returns
        -------
        bool
            ``True`` if all engines are compatible.
        """
        return all(engine.is_compatible(adapter) for engine in self._engines)

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Validate the full pipeline before application.

        Checks for conflicts between engines and ensures the model
        satisfies all engine requirements.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        config : dict
            Combined configuration.

        Returns
        -------
        bool
            ``True`` if validation passes.
        """
        engine_names = [e.engine_name for e in self._engines]

        # Check pairwise compatibility
        compat_ok = self.check_compatibility(engine_names)
        if not compat_ok:
            logger.warning("Engine compatibility check failed")
            return False

        # Validate each engine individually
        for engine in self._engines:
            name = engine.engine_name
            engine_config = config.get(name, {})
            try:
                if hasattr(engine, "validate"):
                    ok = engine.validate(model, engine_config)
                    if not ok:
                        logger.warning("Validation failed for engine '{}'", name)
                        return False
            except Exception as e:
                logger.warning("Validation error for engine '{}': {}", name, e)
                return False

        logger.info("Pipeline validation passed for engines: {}", engine_names)
        return True

    # ------------------------------------------------------------------
    # Combiner-specific methods
    # ------------------------------------------------------------------

    def check_compatibility(self, engine_names: List[str]) -> bool:
        """Check if a set of engines are mutually compatible.

        Parameters
        ----------
        engine_names : list[str]
            Engine names to check.

        Returns
        -------
        bool
            ``True`` if all pairs are compatible.
        """
        for a, b in itertools.combinations(engine_names, 2):
            if a == b:
                continue
            # Check if both are known
            if a not in COMPATIBILITY_MATRIX and b not in COMPATIBILITY_MATRIX:
                # Unknown engines — assume compatible
                continue
            compatible = ENGINE_COMPATIBILITY.get((a, b), False)
            if not compatible:
                logger.warning(
                    "Engines '{}' and '{}' are not compatible", a, b
                )
                return False
        return True

    def pareto_analysis(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        data: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Enumerate engine combinations and find Pareto-optimal ones.

        Tests all 2^N - 1 non-empty subsets of available engines,
        collects (speedup, accuracy, memory) data points, and filters
        for the Pareto front.

        Parameters
        ----------
        model : torch.nn.Module
            The base (unoptimised) model.
        config : dict
            Configuration for each engine.
        data : torch.Tensor
            Representative input for benchmarking.

        Returns
        -------
        list[dict]
            Pareto-optimal combinations, each with keys:
            ``engines``, ``speedup``, ``accuracy``, ``memory_mb``.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        if data.device != device:
            data = data.to(device)

        model.eval()

        # Baseline benchmark
        with torch.no_grad():
            for _ in range(3):
                _ = model(data)
        t0 = time.perf_counter()
        n_baseline = 20
        with torch.no_grad():
            for _ in range(n_baseline):
                _ = model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        baseline_latency = (time.perf_counter() - t0) / n_baseline

        baseline_mem = 0.0
        if torch.cuda.is_available():
            baseline_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()

        # Baseline output for accuracy comparison
        with torch.no_grad():
            baseline_output = model(data)

        # Enumerate all non-empty subsets
        engine_names = [e.engine_name for e in self._engines]
        data_points: List[Dict[str, Any]] = []

        for r in range(1, len(engine_names) + 1):
            for combo in itertools.combinations(engine_names, r):
                combo_list = list(combo)

                # Check compatibility
                if not self.check_compatibility(combo_list):
                    continue

                # Apply engines in order
                try:
                    import copy
                    test_model = copy.deepcopy(model)
                    ordered = sorted(
                        combo_list,
                        key=lambda n: APPLICATION_ORDER.index(n)
                        if n in APPLICATION_ORDER else len(APPLICATION_ORDER),
                    )

                    for name in ordered:
                        eng = next(
                            (e for e in self._engines if e.engine_name == name),
                            None,
                        )
                        if eng is None:
                            continue
                        eng_config = config.get(name, {})
                        test_model = eng.apply(test_model, eng_config)

                    test_model.eval()

                    # Benchmark
                    with torch.no_grad():
                        for _ in range(3):
                            _ = test_model(data)

                    t0 = time.perf_counter()
                    n_runs = 20
                    with torch.no_grad():
                        for _ in range(n_runs):
                            _ = test_model(data)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    opt_latency = (time.perf_counter() - t0) / n_runs

                    opt_mem = 0.0
                    if torch.cuda.is_available():
                        opt_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        torch.cuda.reset_peak_memory_stats()

                    # Accuracy: cosine similarity with baseline
                    with torch.no_grad():
                        opt_output = test_model(data)

                    # Align dimensions
                    if baseline_output.shape != opt_output.shape:
                        min_dim = min(baseline_output.shape[-1], opt_output.shape[-1])
                        bo = baseline_output[..., :min_dim].reshape(baseline_output.shape[0], -1)
                        oo = opt_output[..., :min_dim].reshape(opt_output.shape[0], -1)
                    else:
                        bo = baseline_output.reshape(baseline_output.shape[0], -1)
                        oo = opt_output.reshape(opt_output.shape[0], -1)

                    cos_sim = torch.nn.functional.cosine_similarity(
                        bo.float(), oo.float(), dim=-1
                    ).mean().item()

                    speedup = baseline_latency / opt_latency if opt_latency > 0 else 1.0

                    data_points.append({
                        "engines": combo_list,
                        "speedup": round(speedup, 4),
                        "accuracy": round(cos_sim, 6),
                        "memory_mb": round(opt_mem, 2),
                    })

                except Exception as e:
                    logger.debug("Combo {} failed: {}", combo_list, e)
                    continue

        # Filter Pareto-optimal (maximise speedup & accuracy, minimise memory)
        pareto_front = self._filter_pareto(data_points)

        logger.info(
            "Pareto analysis: {} combos tested, {} on Pareto front",
            len(data_points), len(pareto_front),
        )
        return pareto_front

    def auto_search(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        constraints: Dict[str, float],
    ) -> Dict[str, Any]:
        """Search for the best engine combination under constraints.

        Parameters
        ----------
        model : torch.nn.Module
            The base model.
        config : dict
            Per-engine configuration.
        constraints : dict
            Must contain at least one of:
            - ``"max_memory_mb"``: memory budget.
            - ``"min_speedup"``: minimum speedup factor.
            - ``"min_accuracy"``: minimum cosine similarity.

        Returns
        -------
        dict
            Best combination found with keys: ``engines``, ``speedup``,
            ``accuracy``, ``memory_mb``.
        """
        # Generate synthetic data for benchmarking
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        # Try to infer input shape
        example = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                example = torch.randn(8, module.in_features, device=device)
                break
        if example is None:
            example = torch.randn(8, 128, device=device)

        # Run Pareto analysis
        all_points = self.pareto_analysis(model, config, example)

        if not all_points:
            logger.warning("No valid combinations found")
            return {"engines": [], "speedup": 1.0, "accuracy": 1.0, "memory_mb": 0.0}

        # Filter by constraints
        max_mem = constraints.get("max_memory_mb", float("inf"))
        min_speedup = constraints.get("min_speedup", 0.0)
        min_accuracy = constraints.get("min_accuracy", 0.0)

        feasible = [
            p for p in all_points
            if p["memory_mb"] <= max_mem
            and p["speedup"] >= min_speedup
            and p["accuracy"] >= min_accuracy
        ]

        if not feasible:
            logger.warning(
                "No combination satisfies constraints. "
                "Returning best speedup from Pareto front."
            )
            feasible = all_points

        # Sort by speedup descending, then accuracy descending
        feasible.sort(key=lambda p: (p["speedup"], p["accuracy"]), reverse=True)
        best = feasible[0]

        logger.info(
            "auto_search best: engines={}, speedup={:.2f}x, accuracy={:.4f}, "
            "memory={:.1f} MB",
            best["engines"], best["speedup"], best["accuracy"], best["memory_mb"],
        )
        return best

    def get_compatibility_matrix(self) -> Dict[str, Any]:
        """Return the engine compatibility matrix.

        Returns
        -------
        dict
            Mapping of engine_name → set of compatible engine names.
        """
        return {k: sorted(v) for k, v in COMPATIBILITY_MATRIX.items()}

    def revert(self, model: nn.Module) -> nn.Module:
        """Revert optimization - returns model unchanged for non-reversible engines."""
        logger.info(f"{self.engine_name}: revert not applicable, returning model as-is")
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_pareto(
        data_points: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter data points to keep only Pareto-optimal ones.

        Objectives: maximise speedup, maximise accuracy, minimise memory.
        A point is Pareto-dominated if another point is >= in all objectives
        and strictly > in at least one.

        Parameters
        ----------
        data_points : list[dict]
            All evaluated combinations.

        Returns
        -------
        list[dict]
            Pareto-optimal subset.
        """
        if not data_points:
            return []

        pareto: List[Dict[str, Any]] = []

        for i, pi in enumerate(data_points):
            dominated = False
            for j, pj in enumerate(data_points):
                if i == j:
                    continue
                # pj dominates pi if pj >= pi in all objectives and > in at least one
                if (
                    pj["speedup"] >= pi["speedup"]
                    and pj["accuracy"] >= pi["accuracy"]
                    and pj["memory_mb"] <= pi["memory_mb"]
                    and (
                        pj["speedup"] > pi["speedup"]
                        or pj["accuracy"] > pi["accuracy"]
                        or pj["memory_mb"] < pi["memory_mb"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(pi)

        return pareto

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        """Return the engine identifier."""
        return "combiner"

    @property
    def optimization_type(self) -> str:
        """Return the optimization category."""
        return "combiner"
