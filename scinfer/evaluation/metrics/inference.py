"""Inference performance metrics.

Provides the :class:`InferenceMetrics` class for measuring throughput,
latency, memory consumption, and per-step timing of model inference.

Also exposes convenience functions :func:`compute_throughput`,
:func:`compute_latency`, and :func:`compute_memory_usage` for quick
one-shot measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from loguru import logger


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_device(model: nn.Module) -> torch.device:
    """Return the device of the first model parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync_device(device: torch.device) -> None:
    """Synchronise CUDA device if available."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_numpy(val: Union[torch.Tensor, np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return val


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class ThroughputResult:
    """Result of a throughput measurement."""
    median: float  # items/second
    std: float
    all_measurements: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ThroughputResult(median={self.median:.2f} ± {self.std:.2f} items/s)"


@dataclass
class LatencyResult:
    """Result of a latency measurement."""
    median_ms: float
    std_ms: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    all_measurements_ms: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"LatencyResult(median={self.median_ms:.3f} ± {self.std_ms:.3f} ms)"


@dataclass
class MemoryResult:
    """Result of a memory measurement."""
    peak_gpu_mb: float = 0.0
    cpu_mb: float = 0.0
    model_params_mb: float = 0.0

    def __repr__(self) -> str:
        parts = [f"peak_gpu={self.peak_gpu_mb:.1f} MB"]
        if self.cpu_mb > 0:
            parts.append(f"cpu={self.cpu_mb:.1f} MB")
        parts.append(f"params={self.model_params_mb:.1f} MB")
        return f"MemoryResult({', '.join(parts)})"


@dataclass
class StepTimingResult:
    """Per-step timing breakdown."""
    step_times_ms: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.2f}ms" for k, v in self.step_times_ms.items())
        return f"StepTimingResult(total={self.total_ms:.2f}ms, {items})"


# ---------------------------------------------------------------------------
# InferenceMetrics — main class
# ---------------------------------------------------------------------------

class InferenceMetrics:
    """Measure model inference performance metrics.

    Metrics include:
    - **throughput**: inference throughput (items/second)
    - **latency**: per-cell inference latency (ms/cell)
    - **peak_memory**: peak GPU memory usage (MB)
    - **cpu_memory**: peak CPU memory usage (MB)
    - **time_per_step**: per-analysis-step timing breakdown
    """

    # ------------------------------------------------------------------
    # Throughput
    # ------------------------------------------------------------------

    @staticmethod
    def measure_throughput(
        model: nn.Module,
        data: torch.Tensor,
        batch_size: int = 32,
        n_warmup: int = 10,
        n_measure: int = 50,
    ) -> ThroughputResult:
        """Measure inference throughput.

        Parameters
        ----------
        model : nn.Module
            Model under test (should already be on the target device).
        data : torch.Tensor
            Input tensor whose first dimension is the sample axis.
        batch_size : int
            Mini-batch size for each forward pass.
        n_warmup : int
            Number of warmup iterations (discarded).
        n_measure : int
            Number of timed iterations.

        Returns
        -------
        ThroughputResult
            Median ± std throughput in items/second.
        """
        device = _get_device(model)
        model.eval()

        n_samples = data.shape[0]
        # Ensure data is on the correct device
        if isinstance(data, torch.Tensor) and data.device != device:
            data = data.to(device)

        measurements: List[float] = []

        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                idx = torch.randint(0, n_samples, (min(batch_size, n_samples),))
                batch = data[idx]
                _ = model(batch)
            _sync_device(device)

            # Measure
            for _ in range(n_measure):
                idx = torch.randint(0, n_samples, (min(batch_size, n_samples),))
                batch = data[idx]

                _sync_device(device)
                t0 = time.perf_counter()
                _ = model(batch)
                _sync_device(device)
                t1 = time.perf_counter()

                elapsed = t1 - t0
                throughput = batch.shape[0] / elapsed
                measurements.append(throughput)

        arr = np.array(measurements, dtype=np.float64)
        result = ThroughputResult(
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            all_measurements=measurements,
        )
        logger.debug(f"Throughput: {result.median:.2f} ± {result.std:.2f} items/s")
        return result

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    @staticmethod
    def measure_latency(
        model: nn.Module,
        data: torch.Tensor,
        n_warmup: int = 10,
        n_measure: int = 100,
    ) -> LatencyResult:
        """Measure per-cell inference latency.

        Feeds one sample at a time and records the wall-clock time for
        each forward pass.

        Parameters
        ----------
        model : nn.Module
            Model under test.
        data : torch.Tensor
            Input tensor (N, …).  Each row is treated as a single cell.
        n_warmup : int
            Warmup iterations.
        n_measure : int
            Timed iterations.

        Returns
        -------
        LatencyResult
            Latency statistics in milliseconds.
        """
        device = _get_device(model)
        model.eval()

        if isinstance(data, torch.Tensor) and data.device != device:
            data = data.to(device)

        measurements_ms: List[float] = []

        with torch.no_grad():
            for i in range(n_warmup + n_measure):
                sample = data[i % data.shape[0]].unsqueeze(0)
                _sync_device(device)
                t0 = time.perf_counter()
                _ = model(sample)
                _sync_device(device)
                t1 = time.perf_counter()
                if i >= n_warmup:
                    measurements_ms.append((t1 - t0) * 1000.0)

        arr = np.array(measurements_ms, dtype=np.float64)
        percentiles = {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
        result = LatencyResult(
            median_ms=float(np.median(arr)),
            std_ms=float(np.std(arr)),
            percentiles=percentiles,
            all_measurements_ms=measurements_ms,
        )
        logger.debug(f"Latency: {result.median_ms:.3f} ± {result.std_ms:.3f} ms/cell")
        return result

    # ------------------------------------------------------------------
    # Peak memory
    # ------------------------------------------------------------------

    @staticmethod
    def measure_peak_memory(
        model: nn.Module,
        data: torch.Tensor,
        batch_size: int = 32,
    ) -> MemoryResult:
        """Measure peak GPU and CPU memory during inference.

        Parameters
        ----------
        model : nn.Module
            Model under test.
        data : torch.Tensor
            Input tensor.
        batch_size : int
            Batch size for the forward pass.

        Returns
        -------
        MemoryResult
            Peak GPU memory (MB), CPU memory (MB), and model parameter memory (MB).
        """
        device = _get_device(model)
        model.eval()

        if isinstance(data, torch.Tensor) and data.device != device:
            data = data.to(device)

        # Model parameter memory
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        params_mb = params_bytes / (1024 ** 2)

        # CPU memory via psutil (best-effort)
        cpu_before_mb = 0.0
        cpu_after_mb = 0.0
        try:
            import psutil
            import os
            proc = psutil.Process(os.getpid())
            cpu_before_mb = proc.memory_info().rss / (1024 ** 2)
        except ImportError:
            logger.debug("psutil not available; CPU memory measurement skipped")

        peak_gpu_mb = 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for start in range(0, data.shape[0], batch_size):
                    batch = data[start : start + batch_size]
                    _ = model(batch)
                _sync_device(device)
            peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            # CPU-only fallback — estimate from parameter + input sizes
            with torch.no_grad():
                for start in range(0, data.shape[0], batch_size):
                    batch = data[start : start + batch_size]
                    _ = model(batch)
            peak_gpu_mb = 0.0  # No GPU memory

        try:
            import psutil
            import os
            proc = psutil.Process(os.getpid())
            cpu_after_mb = proc.memory_info().rss / (1024 ** 2)
        except ImportError:
            pass

        cpu_delta = max(cpu_after_mb - cpu_before_mb, 0.0)

        result = MemoryResult(
            peak_gpu_mb=peak_gpu_mb,
            cpu_mb=cpu_delta,
            model_params_mb=params_mb,
        )
        logger.debug(f"Memory: GPU peak={peak_gpu_mb:.1f} MB, params={params_mb:.1f} MB")
        return result

    # ------------------------------------------------------------------
    # Time per step (PyTorch Profiler breakdown)
    # ------------------------------------------------------------------

    @staticmethod
    def measure_time_per_step(
        model: nn.Module,
        data: torch.Tensor,
        batch_size: int = 32,
        n_warmup: int = 3,
    ) -> StepTimingResult:
        """Decompose inference time into functional categories.

        Uses :mod:`torch.profiler` to attribute time to:
        ``attention``, ``ffn``, ``normalization``, ``embedding``, ``other``.

        Parameters
        ----------
        model : nn.Module
            Model under test.
        data : torch.Tensor
            Input tensor.
        batch_size : int
            Batch size.
        n_warmup : int
            Warmup iterations before profiling.

        Returns
        -------
        StepTimingResult
            Mapping of category → time in ms.
        """
        device = _get_device(model)
        model.eval()

        if isinstance(data, torch.Tensor) and data.device != device:
            data = data.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                batch = data[:batch_size]
                _ = model(batch)
            _sync_device(device)

        # Category mapping based on kernel / operator names
        _CATEGORY_MAP = {
            "attention": ["aten::bmm", "aten::scaled_dot_product", "softmax", "matmul",
                          "aten::baddbmm", "flash", "xformers", "multihead"],
            "ffn": ["aten::linear", "aten::addmm", "aten::mm", "gelu", "relu", "silu",
                    "aten::add_", "aten::mul_"],
            "normalization": ["layer_norm", "batch_norm", "group_norm", "rms_norm",
                              "aten::native_layer_norm"],
            "embedding": ["aten::embedding", "embedding_bag"],
        }

        def _categorise(name: str) -> str:
            name_lower = name.lower()
            for cat, keywords in _CATEGORY_MAP.items():
                if any(kw in name_lower for kw in keywords):
                    return cat
            return "other"

        step_times: Dict[str, float] = {
            "attention": 0.0,
            "ffn": 0.0,
            "normalization": 0.0,
            "embedding": 0.0,
            "other": 0.0,
        }

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                batch = data[:batch_size]
                _ = model(batch)
            _sync_device(device)

        # Parse profiler events
        total_time_us = 0.0
        for evt in prof.key_averages():
            cat = _categorise(evt.key)
            cuda_time = evt.cuda_time_total if hasattr(evt, "cuda_time_total") else 0.0
            cpu_time = evt.cpu_time_total if hasattr(evt, "cpu_time_total") else 0.0
            # Prefer CUDA time when available
            t_us = cuda_time if cuda_time > 0 else cpu_time
            step_times[cat] += t_us
            total_time_us += t_us

        # Convert to ms
        step_times_ms = {k: v / 1000.0 for k, v in step_times.items()}
        total_ms = sum(step_times_ms.values())

        result = StepTimingResult(step_times_ms=step_times_ms, total_ms=total_ms)
        logger.debug(f"Step timing: total={total_ms:.2f} ms")
        return result

    # ------------------------------------------------------------------
    # Memory breakdown (per-layer)
    # ------------------------------------------------------------------

    @staticmethod
    def profile_memory_breakdown(model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Per-layer parameter, gradient, and buffer memory statistics.

        Parameters
        ----------
        model : nn.Module
            Model to analyse.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{layer_name: {"params_mb": ..., "grad_mb": ..., "buffers_mb": ...}}``
        """
        breakdown: Dict[str, Dict[str, float]] = {}

        for name, module in model.named_modules():
            params_bytes = sum(
                p.numel() * p.element_size() for p in module.parameters(recurse=False)
            )
            grad_bytes = sum(
                p.grad.numel() * p.grad.element_size()
                for p in module.parameters(recurse=False)
                if p.grad is not None
            )
            buf_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))

            if params_bytes > 0 or grad_bytes > 0 or buf_bytes > 0:
                breakdown[name if name else "(root)"] = {
                    "params_mb": params_bytes / (1024 ** 2),
                    "grad_mb": grad_bytes / (1024 ** 2),
                    "buffers_mb": buf_bytes / (1024 ** 2),
                }

        return breakdown


# ---------------------------------------------------------------------------
# Convenience functions (backward-compatible with placeholder signatures)
# ---------------------------------------------------------------------------

def compute_throughput(
    model: nn.Module,
    input_data: torch.Tensor,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
) -> float:
    """Compute inference throughput in samples per second.

    This is a convenience wrapper around :meth:`InferenceMetrics.measure_throughput`.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.
    input_data : torch.Tensor
        Input tensor (batch_size, …).
    warmup_runs : int
        Number of warmup iterations.
    benchmark_runs : int
        Number of timed iterations.

    Returns
    -------
    float
        Median throughput in samples/second.
    """
    batch_size = input_data.shape[0]
    result = InferenceMetrics.measure_throughput(
        model, input_data, batch_size=batch_size,
        n_warmup=warmup_runs, n_measure=benchmark_runs,
    )
    return result.median


def compute_latency(
    model: nn.Module,
    input_data: torch.Tensor,
    warmup_runs: int = 5,
    benchmark_runs: int = 20,
    percentiles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute inference latency percentiles in milliseconds.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.
    input_data : torch.Tensor
        Input tensor (batch_size, …).
    warmup_runs : int
        Number of warmup iterations.
    benchmark_runs : int
        Number of timed iterations.
    percentiles : list of float, optional
        Percentiles to compute (default: [50, 90, 95, 99]).

    Returns
    -------
    dict[str, float]
        Mapping of percentile name → latency in ms.
    """
    result = InferenceMetrics.measure_latency(
        model, input_data, n_warmup=warmup_runs, n_measure=benchmark_runs,
    )
    return result.percentiles


def compute_memory_usage(
    model: nn.Module,
    input_data: torch.Tensor,
) -> Dict[str, float]:
    """Compute GPU memory usage during inference.

    Parameters
    ----------
    model : torch.nn.Module
        The model to measure.
    input_data : torch.Tensor
        Input tensor.

    Returns
    -------
    dict[str, float]
        Memory metrics in MB: ``peak_allocated``, ``peak_reserved``,
        ``model_params_mb``.
    """
    result = InferenceMetrics.measure_peak_memory(model, input_data)
    out: Dict[str, float] = {
        "model_params_mb": result.model_params_mb,
    }
    device = _get_device(model)
    if device.type == "cuda":
        out["peak_allocated"] = result.peak_gpu_mb
        out["peak_reserved"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        out["peak_allocated"] = 0.0
        out["peak_reserved"] = 0.0
    return out
