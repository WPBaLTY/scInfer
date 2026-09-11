"""Benchmark runner for comparing models and optimization strategies.

Provides:
- :class:`BenchmarkResult` — container for benchmark output with export
  helpers (CSV, JSON, DataFrame).
- :class:`BenchmarkRunner` — high-level runner that orchestrates
  model × dataset benchmarking.
- :class:`BenchmarkProtocol` — unified protocol for standard dataset
  loading, measurement specification, and result persistence.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from loguru import logger

from scinfer.evaluation.metrics.inference import InferenceMetrics


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for benchmark results across models/configurations.

    Attributes
    ----------
    results : dict
        Nested mapping: ``model_name → metric_name → value``.
    config : dict
        Configuration used for the benchmark.
    metadata : dict
        Additional metadata (hardware, versions, timestamps).
    """

    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable comparison table."""
        if not self.results:
            return "(no results)"

        # Collect all metric names
        all_metrics: List[str] = []
        for metrics in self.results.values():
            for m in metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        # Column widths
        name_w = max(len(n) for n in list(self.results.keys()) + ["Model"])
        col_w = 14

        header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in all_metrics)
        sep = "-" * len(header)
        rows = [sep, header, sep]
        for model_name, metrics in self.results.items():
            row = f"{model_name:<{name_w}}"
            for m in all_metrics:
                val = metrics.get(m, float("nan"))
                if isinstance(val, float):
                    row += f"{val:>{col_w}.4f}"
                else:
                    row += f"{str(val):>{col_w}}"
            rows.append(row)
        rows.append(sep)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise results to a plain dictionary."""
        return {
            "results": {k: dict(v) for k, v in self.results.items()},
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }

    def to_json(self, path: Union[str, Path]) -> Path:
        """Export results to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {path}")
        return path

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Export results to a flat CSV file.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        Path
        """
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        all_metrics: List[str] = []
        for metrics in self.results.values():
            for m in metrics:
                if m not in all_metrics:
                    all_metrics.append(m)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + all_metrics)
            for model_name, metrics in self.results.items():
                row = [model_name] + [metrics.get(m, "") for m in all_metrics]
                writer.writerow(row)
        logger.info(f"Benchmark CSV saved to {path}")
        return path

    def to_dataframe(self) -> Any:
        """Convert results to a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Rows = models, columns = metrics.
        """
        import pandas as pd
        return pd.DataFrame.from_dict(self.results, orient="index")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, other: "BenchmarkResult") -> Dict[str, Dict[str, float]]:
        """Compare this result (baseline) against another (optimised).

        Returns
        -------
        dict
            ``{model_name: {metric: speedup_ratio}}`` where ratio > 1 means
            the *other* result is better (for latency/memory) or faster.
        """
        comparison: Dict[str, Dict[str, float]] = {}
        for model_name in self.results:
            if model_name not in other.results:
                continue
            comparison[model_name] = {}
            base_metrics = self.results[model_name]
            opt_metrics = other.results[model_name]
            for metric in base_metrics:
                if metric not in opt_metrics:
                    continue
                bval = base_metrics[metric]
                oval = opt_metrics[metric]
                if isinstance(bval, (int, float)) and isinstance(oval, (int, float)):
                    if bval != 0:
                        comparison[model_name][metric] = oval / bval
                    else:
                        comparison[model_name][metric] = float("inf") if oval > 0 else 1.0
        return comparison


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run standardised inference benchmarks.

    Parameters
    ----------
    model_names : list of str
        Models to benchmark.
    datasets : list of str, optional
        Dataset identifiers or paths.
    metrics : list of str, optional
        Metric names to compute (default: throughput, latency, memory).
    warmup_runs : int
        Number of warmup iterations before timing.
    benchmark_runs : int
        Number of timed iterations.
    """

    def __init__(
        self,
        model_names: List[str],
        datasets: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
    ) -> None:
        self.model_names = list(model_names)
        self.datasets = datasets or []
        self.metrics = metrics or ["throughput", "latency", "memory"]
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def run(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        models: Optional[Dict[str, nn.Module]] = None,
    ) -> BenchmarkResult:
        """Execute the benchmark across all configured models.

        Parameters
        ----------
        input_data : dict, optional
            Pre-loaded input tensors keyed by model name.
        models : dict, optional
            Pre-loaded models keyed by model name.  If ``None``, models
            must be registered externally (see :class:`BenchmarkProtocol`).

        Returns
        -------
        BenchmarkResult
            Aggregated benchmark results.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for name in self.model_names:
            data = (input_data or {}).get(name)
            model = (models or {}).get(name)
            if model is None or data is None:
                logger.warning(f"Skipping '{name}': model or data not provided")
                continue
            results[name] = self.run_single_model(model, data)

        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            metadata["gpu_count"] = torch.cuda.device_count()

        return BenchmarkResult(
            results=results,
            config={
                "model_names": self.model_names,
                "datasets": self.datasets,
                "metrics": self.metrics,
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
            },
            metadata=metadata,
        )

    def run_single(
        self, model_name: str, input_data: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Benchmark a single model (convenience wrapper).

        Parameters
        ----------
        model_name : str
            Model identifier.
        input_data : torch.Tensor
            Input tensor.
        model : nn.Module, optional
            The model instance.

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        if model is None:
            logger.warning(f"No model instance for '{model_name}'")
            return {}
        return self.run_single_model(model, input_data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def run_single_model(self, model: nn.Module, data: torch.Tensor) -> Dict[str, Any]:
        """Run all requested metrics on a single model."""
        out: Dict[str, Any] = {}
        batch_size = data.shape[0]

        if "throughput" in self.metrics:
            tp = InferenceMetrics.measure_throughput(
                model, data, batch_size=batch_size,
                n_warmup=self.warmup_runs, n_measure=self.benchmark_runs,
            )
            out["throughput_ips"] = tp.median
            out["throughput_std"] = tp.std

        if "latency" in self.metrics:
            lat = InferenceMetrics.measure_latency(
                model, data, n_warmup=self.warmup_runs, n_measure=self.benchmark_runs,
            )
            out["latency_median_ms"] = lat.median_ms
            out["latency_std_ms"] = lat.std_ms
            out.update({f"latency_{k}_ms": v for k, v in lat.percentiles.items()})

        if "memory" in self.metrics:
            mem = InferenceMetrics.measure_peak_memory(model, data, batch_size=batch_size)
            out["peak_gpu_memory_mb"] = mem.peak_gpu_mb
            out["cpu_memory_mb"] = mem.cpu_mb
            out["model_params_mb"] = mem.model_params_mb

        return out


# ---------------------------------------------------------------------------
# BenchmarkProtocol
# ---------------------------------------------------------------------------

class BenchmarkProtocol:
    """Unified benchmark execution protocol.

    Provides:
    - Standard dataset loading and preprocessing.
    - Unified measurement specification (warmup + measure + statistics).
    - Multi-model comparison benchmarks.
    - Result persistence and export.
    """

    # Known dataset identifiers
    _KNOWN_DATASETS = {
        "pbmc68k", "pbmc_68k",
        "pancreas_baron", "pancreas",
        "pancreas_muraro",
        "mouse_brain_1_3m", "mouse_brain",
    }

    def __init__(
        self,
        output_dir: Union[str, Path] = "./benchmark_results",
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self._models: Dict[str, nn.Module] = {}
        self._datasets: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(self, name: str, model: nn.Module) -> None:
        """Register a model for benchmarking.

        Parameters
        ----------
        name : str
            Model identifier.
        model : nn.Module
            Model instance.
        """
        self._models[name] = model
        logger.info(f"Registered model: {name}")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_benchmark_dataset(self, name: str) -> Any:
        """Load a standard benchmark dataset.

        Supported names: ``pbmc68k``, ``pancreas_baron``, ``pancreas_muraro``,
        ``mouse_brain_1_3m``.

        Falls back to generating a synthetic dataset if the real dataset
        cannot be downloaded.

        Parameters
        ----------
        name : str
            Dataset identifier.

        Returns
        -------
        anndata.AnnData
            Standardised AnnData object.
        """
        import scanpy as sc
        import anndata as ad

        name_lower = name.lower().replace("-", "_").replace(" ", "_")

        if name_lower in ("pbmc68k", "pbmc_68k"):
            try:
                adata = sc.datasets.pbmc68k_reduced()
                logger.info(f"Loaded PBMC 68K dataset: {adata.shape}")
                return adata
            except Exception as e:
                logger.warning(f"Could not load PBMC 68K: {e}. Generating synthetic data.")
                return self._synthetic_dataset(5000, 2000, "pbmc68k")

        if name_lower in ("pancreas_baron", "pancreas"):
            try:
                adata = sc.datasets.pancreas()
                logger.info(f"Loaded Pancreas dataset: {adata.shape}")
                return adata
            except Exception as e:
                logger.warning(f"Could not load Pancreas: {e}. Generating synthetic data.")
                return self._synthetic_dataset(8000, 3000, "pancreas")

        if name_lower == "pancreas_muraro":
            try:
                adata = sc.datasets.pancreas()
                logger.info(f"Loaded Pancreas (Muraro) dataset: {adata.shape}")
                return adata
            except Exception as e:
                logger.warning(f"Could not load Pancreas Muraro: {e}. Generating synthetic data.")
                return self._synthetic_dataset(8000, 3000, "pancreas_muraro")

        if name_lower in ("mouse_brain_1_3m", "mouse_brain"):
            try:
                adata = sc.datasets.pbmc3k()  # fallback to smaller dataset
                logger.info(f"Loaded Mouse Brain (fallback pbmc3k) dataset: {adata.shape}")
                return adata
            except Exception as e:
                logger.warning(f"Could not load Mouse Brain: {e}. Generating synthetic data.")
                return self._synthetic_dataset(10000, 5000, "mouse_brain")

        # Unknown dataset — try as file path
        path = Path(name)
        if path.exists():
            import scanpy as sc
            adata = sc.read_h5ad(path)
            logger.info(f"Loaded dataset from {path}: {adata.shape}")
            return adata

        logger.warning(f"Unknown dataset '{name}'. Generating synthetic data.")
        return self._synthetic_dataset(5000, 2000, name)

    @staticmethod
    def _synthetic_dataset(n_cells: int, n_genes: int, name: str) -> Any:
        """Generate a synthetic AnnData for testing."""
        import anndata as ad
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["cell_type"] = rng.choice(
            [f"type_{i}" for i in range(10)], size=n_cells
        )
        adata.obs["batch"] = rng.choice([f"batch_{i}" for i in range(3)], size=n_cells)
        logger.info(f"Generated synthetic dataset '{name}': {adata.shape}")
        return adata

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        model_names: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        input_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> BenchmarkResult:
        """Run a full benchmark across model × dataset combinations.

        Parameters
        ----------
        model_names : list of str, optional
            Models to benchmark (defaults to registered models).
        datasets : list of str, optional
            Dataset identifiers.
        metrics : list of str, optional
            Metrics to compute.
        config : dict, optional
            Extra configuration.
        input_tensors : dict, optional
            Pre-computed input tensors keyed by model name.

        Returns
        -------
        BenchmarkResult
            Aggregated results.
        """
        model_names = model_names or list(self._models.keys())
        datasets = datasets or []
        metrics = metrics or ["throughput", "latency", "memory"]

        runner = BenchmarkRunner(
            model_names=model_names,
            datasets=datasets,
            metrics=metrics,
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
        )

        # Build input data dict
        input_data = input_tensors or {}

        result = runner.run(
            input_data=input_data,
            models=self._models,
        )

        # Attach extra config
        if config:
            result.config.update(config)

        # Persist
        result.to_json(self.output_dir / "benchmark_results.json")
        result.to_csv(self.output_dir / "benchmark_results.csv")

        return result

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_comparison_table(
        results: BenchmarkResult,
        metric_name: str,
        fmt: str = "markdown",
    ) -> str:
        """Generate a model comparison table for a specific metric.

        Parameters
        ----------
        results : BenchmarkResult
            Benchmark results to tabulate.
        metric_name : str
            Metric to display.
        fmt : str
            ``"markdown"`` or ``"latex"``.

        Returns
        -------
        str
            Formatted table.
        """
        rows: List[Tuple[str, float]] = []
        for model_name, metrics in results.results.items():
            val = metrics.get(metric_name, float("nan"))
            if isinstance(val, (int, float)):
                rows.append((model_name, float(val)))

        rows.sort(key=lambda x: x[1], reverse=True)

        if fmt == "latex":
            lines = [
                r"\begin{tabular}{lr}",
                r"\toprule",
                r"Model & " + metric_name.replace("_", r"\_") + r" \\",
                r"\midrule",
            ]
            for name, val in rows:
                lines.append(f"{name} & {val:.4f} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}"]
            return "\n".join(lines)
        else:
            # Markdown
            lines = [
                f"| Model | {metric_name} |",
                "|-------|-------------|",
            ]
            for name, val in rows:
                lines.append(f"| {name} | {val:.4f} |")
            return "\n".join(lines)
