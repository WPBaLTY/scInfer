"""Inference profiler for single-cell foundation models.

Identifies performance bottlenecks by instrumenting model layers and
measuring per-layer latency, memory consumption, and compute utilisation.

Provides:
- :class:`ProfileResult` — container for profiling output.
- :class:`InferenceProfiler` — deep profiling engine with time/memory
  decomposition, FLOPs estimation, hardware comparison, and Chrome trace
  export.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from loguru import logger


# ---------------------------------------------------------------------------
# Category mapping for operator → functional block
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "attention": [
        "aten::bmm", "aten::scaled_dot_product", "softmax", "multihead",
        "aten::baddbmm", "flash", "xformers", "attn",
    ],
    "ffn": [
        "aten::linear", "aten::addmm", "aten::mm", "gelu", "relu", "silu",
        "aten::add_", "aten::mul_", "aten::add",
    ],
    "normalization": [
        "layer_norm", "batch_norm", "group_norm", "rms_norm",
        "aten::native_layer_norm",
    ],
    "embedding": [
        "aten::embedding", "embedding_bag",
    ],
    "data_loading": [
        "aten::copy_", "aten::to", "cudaMemcpy", "Memcpy",
    ],
}


def _categorise_op(name: str) -> str:
    """Map an operator name to a functional category."""
    name_lower = name.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# ProfileResult
# ---------------------------------------------------------------------------

@dataclass
class ProfileResult:
    """Container for inference profiling results.

    Attributes
    ----------
    model_name : str
        Name of the profiled model.
    total_latency_ms : float
        Total inference latency in milliseconds.
    per_layer_latency_ms : dict
        Mapping of layer name → latency in ms.
    peak_memory_mb : float
        Peak GPU memory usage in megabytes.
    bottleneck_layers : list of str
        Layers identified as performance bottlenecks.
    raw_profile : dict
        Raw profiling data for advanced analysis.
    time_breakdown : dict
        Time decomposition by functional category (ms).
    memory_breakdown : dict
        Memory decomposition by layer (MB).
    flops : dict
        FLOPs estimation results.
    """

    model_name: str
    total_latency_ms: float = 0.0
    per_layer_latency_ms: Dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    bottleneck_layers: List[str] = field(default_factory=list)
    raw_profile: Dict[str, Any] = field(default_factory=dict)
    time_breakdown: Dict[str, float] = field(default_factory=dict)
    memory_breakdown: Dict[str, float] = field(default_factory=dict)
    flops: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the profiling results."""
        lines = [
            f"=== Profile: {self.model_name} ===",
            f"Total latency : {self.total_latency_ms:.3f} ms",
            f"Peak GPU mem  : {self.peak_memory_mb:.1f} MB",
        ]
        if self.time_breakdown:
            lines.append("Time breakdown:")
            for cat, ms in sorted(self.time_breakdown.items(), key=lambda x: -x[1]):
                pct = ms / self.total_latency_ms * 100 if self.total_latency_ms > 0 else 0
                lines.append(f"  {cat:15s}: {ms:8.2f} ms ({pct:5.1f}%)")
        if self.bottleneck_layers:
            lines.append(f"Bottleneck layers: {', '.join(self.bottleneck_layers)}")
        if self.flops:
            lines.append(f"Total FLOPs: {self.flops.get('total', 0):.2e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InferenceProfiler
# ---------------------------------------------------------------------------

class InferenceProfiler:
    """Profile inference performance of single-cell foundation models.

    Uses PyTorch's built-in profiler and custom hooks to measure:
    - Per-layer latency breakdown.
    - GPU memory consumption over time.
    - Compute utilisation (FLOPs, compute density).
    - Bottleneck identification (layers consuming > threshold % of total time).

    Parameters
    ----------
    model_name : str
        Name of the model to profile.
    warmup_runs : int
        Number of warmup forward passes before profiling.
    profile_runs : int
        Number of profiled forward passes.
    bottleneck_threshold : float
        Fraction of total time above which a layer is flagged as bottleneck.
    """

    def __init__(
        self,
        model_name: str,
        warmup_runs: int = 3,
        profile_runs: int = 10,
        bottleneck_threshold: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        self.bottleneck_threshold = bottleneck_threshold
        logger.debug(
            f"InferenceProfiler initialised for '{model_name}' "
            f"(warmup={warmup_runs}, runs={profile_runs})"
        )

    # ------------------------------------------------------------------
    # Main profiling entry point
    # ------------------------------------------------------------------

    def profile(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        input_lengths: Optional[List[int]] = None,
    ) -> ProfileResult:
        """Run profiling on the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to profile.
        input_data : torch.Tensor
            Representative input tensor.
        input_lengths : list of int, optional
            Sequence lengths to profile at (for variable-length analysis).

        Returns
        -------
        ProfileResult
            Profiling results with per-layer breakdown.
        """
        model.eval()
        device = self._get_device(model)

        if input_data.device != device:
            input_data = input_data.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_data)
            self._sync(device)

        # ---- Time profiling with hooks ----
        per_layer_times: Dict[str, List[float]] = defaultdict(list)
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def _make_pre_hook(name: str):
            def hook_fn(module, inputs):
                self._sync(device)
                module._scinfer_t0 = time.perf_counter()
            return hook_fn

        def _make_post_hook(name: str):
            def hook_fn(module, inputs, output):
                self._sync(device)
                t = (time.perf_counter() - module._scinfer_t0) * 1000.0
                per_layer_times[name].append(t)
            return hook_fn

        for name, mod in model.named_modules():
            if name == "":
                continue
            hooks.append(mod.register_forward_pre_hook(_make_pre_hook(name)))
            hooks.append(mod.register_forward_hook(_make_post_hook(name)))

        # Profiled runs
        total_times: List[float] = []
        with torch.no_grad():
            for _ in range(self.profile_runs):
                self._sync(device)
                t0 = time.perf_counter()
                _ = model(input_data)
                self._sync(device)
                total_times.append((time.perf_counter() - t0) * 1000.0)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Aggregate per-layer times
        per_layer_avg = {
            name: float(np.mean(times))
            for name, times in per_layer_times.items()
            if times
        }

        # Total latency
        total_ms = float(np.mean(total_times))

        # Identify bottlenecks
        bottlenecks = [
            name for name, t in per_layer_avg.items()
            if t / total_ms > self.bottleneck_threshold
        ] if total_ms > 0 else []

        # ---- Time breakdown by category ----
        time_breakdown = self._categorise_times(per_layer_avg)

        # ---- Memory profiling ----
        peak_mem = self._measure_peak_memory(model, input_data, device)
        mem_breakdown = self._memory_breakdown(model)

        # ---- Raw profile data ----
        raw = {
            "per_layer_times_ms": per_layer_avg,
            "total_times_ms": total_times,
            "n_runs": self.profile_runs,
        }

        result = ProfileResult(
            model_name=self.model_name,
            total_latency_ms=total_ms,
            per_layer_latency_ms=per_layer_avg,
            peak_memory_mb=peak_mem,
            bottleneck_layers=bottlenecks,
            raw_profile=raw,
            time_breakdown=time_breakdown,
            memory_breakdown=mem_breakdown,
        )
        logger.info(f"Profiling complete: {total_ms:.2f} ms total, "
                     f"{peak_mem:.1f} MB peak GPU mem")
        return result

    # ------------------------------------------------------------------
    # profile_time — torch.profiler based
    # ------------------------------------------------------------------

    def profile_time(
        self,
        model: nn.Module,
        adapter: Any,
        input_data: torch.Tensor,
    ) -> Dict[str, Any]:
        """Fine-grained time profiling using ``torch.profiler``.

        Parameters
        ----------
        model : nn.Module
            Model to profile.
        adapter : Any
            Adapter instance (unused for now, reserved for future use).
        input_data : torch.Tensor
            Input tensor.

        Returns
        -------
        dict
            ``{"breakdown_ms": {category: ms}, "percent": {category: pct},
            "total_ms": float, "events": [...]}``
        """
        model.eval()
        device = self._get_device(model)
        if input_data.device != device:
            input_data = input_data.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_data)
            self._sync(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_data)
            self._sync(device)

        # Aggregate by category
        cat_times_us: Dict[str, float] = defaultdict(float)
        events_list: List[Dict[str, Any]] = []

        for evt in prof.key_averages():
            cuda_us = getattr(evt, "cuda_time_total", 0.0)
            cpu_us = getattr(evt, "cpu_time_total", 0.0)
            t_us = cuda_us if cuda_us > 0 else cpu_us
            cat = _categorise_op(evt.key)
            cat_times_us[cat] += t_us
            events_list.append({
                "name": evt.key,
                "category": cat,
                "cpu_time_us": cpu_us,
                "cuda_time_us": cuda_us,
                "count": evt.count,
            })

        total_us = sum(cat_times_us.values())
        breakdown_ms = {k: v / 1000.0 for k, v in cat_times_us.items()}
        percent = {k: (v / total_us * 100) if total_us > 0 else 0.0
                   for k, v in cat_times_us.items()}

        return {
            "breakdown_ms": breakdown_ms,
            "percent": percent,
            "total_ms": total_us / 1000.0,
            "events": events_list,
        }

    # ------------------------------------------------------------------
    # profile_memory
    # ------------------------------------------------------------------

    def profile_memory(
        self,
        model: nn.Module,
        adapter: Any,
    ) -> Dict[str, Any]:
        """Profile per-layer parameter and buffer memory.

        Parameters
        ----------
        model : nn.Module
            Model to profile.
        adapter : Any
            Adapter instance (reserved).

        Returns
        -------
        dict
            ``{"per_layer": {name: {params_mb, buffers_mb}}, "total_params_mb": float,
            "total_buffers_mb": float}``
        """
        per_layer: Dict[str, Dict[str, float]] = {}
        total_params = 0.0
        total_buffers = 0.0

        for name, mod in model.named_modules():
            p_bytes = sum(
                p.numel() * p.element_size()
                for p in mod.parameters(recurse=False)
            )
            b_bytes = sum(
                buf.numel() * buf.element_size()
                for buf in mod.buffers(recurse=False)
            )
            p_mb = p_bytes / (1024 ** 2)
            b_mb = b_bytes / (1024 ** 2)
            total_params += p_mb
            total_buffers += b_mb
            if p_bytes > 0 or b_bytes > 0:
                per_layer[name if name else "(root)"] = {
                    "params_mb": p_mb,
                    "buffers_mb": b_mb,
                }

        return {
            "per_layer": per_layer,
            "total_params_mb": total_params,
            "total_buffers_mb": total_buffers,
        }

    # ------------------------------------------------------------------
    # estimate_flops
    # ------------------------------------------------------------------

    def estimate_flops(
        self,
        model: nn.Module,
        adapter: Any,
        input_shape: Tuple[int, ...],
    ) -> Dict[str, float]:
        """Estimate FLOPs for a single forward pass.

        Uses a simplified analytical approach for common layer types:
        - Linear: 2 * in * out * batch
        - Attention (QKV): ~12 * d_model^2 * seq_len + 4 * seq_len * d_model
        - LayerNorm: ~5 * hidden * seq_len

        Parameters
        ----------
        model : nn.Module
            Model to analyse.
        adapter : Any
            Adapter instance (reserved).
        input_shape : tuple of int
            Shape of a single input (batch, seq_len, hidden) or (batch, hidden).

        Returns
        -------
        dict
            ``{"attention_flops": float, "ffn_flops": float, "total": float,
            "compute_density": float}``
        """
        batch = input_shape[0] if len(input_shape) >= 1 else 1
        seq_len = input_shape[1] if len(input_shape) >= 3 else 1
        hidden = input_shape[-1] if len(input_shape) >= 2 else 512

        attention_flops = 0.0
        ffn_flops = 0.0
        total_params_bytes = 0

        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                # FLOPs = 2 * batch * seq_len * in_features * out_features
                flops = 2.0 * batch * seq_len * mod.in_features * mod.out_features
                param_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())
                total_params_bytes += param_bytes

                # Heuristic: if out_features looks like attention projection
                if any(kw in name.lower() for kw in ["attn", "attention", "query", "key", "value", "qkv"]):
                    attention_flops += flops
                else:
                    ffn_flops += flops

            elif isinstance(mod, nn.MultiheadAttention):
                # Approximate: 4 * batch * seq_len * embed_dim^2 (QKV + output proj)
                # + 2 * batch * seq_len^2 * head_dim (attention scores)
                embed_dim = mod.embed_dim
                flops = (4.0 * batch * seq_len * embed_dim ** 2
                         + 2.0 * batch * seq_len ** 2 * (embed_dim // max(mod.num_heads, 1)))
                attention_flops += flops
                param_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())
                total_params_bytes += param_bytes

        total_flops = attention_flops + ffn_flops
        # Compute density: FLOPs per byte of memory accessed
        compute_density = total_flops / max(total_params_bytes, 1)

        result = {
            "attention_flops": attention_flops,
            "ffn_flops": ffn_flops,
            "total": total_flops,
            "compute_density": compute_density,
        }
        logger.debug(f"FLOPs estimate: total={total_flops:.2e}, "
                     f"attn={attention_flops:.2e}, ffn={ffn_flops:.2e}")
        return result

    # ------------------------------------------------------------------
    # compare_hardware
    # ------------------------------------------------------------------

    def compare_hardware(
        self,
        model: nn.Module,
        adapter: Any,
        gpu_list: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare inference performance across multiple GPUs.

        Parameters
        ----------
        model : nn.Module
            Model to benchmark.
        adapter : Any
            Adapter instance (reserved).
        gpu_list : list of int, optional
            CUDA device IDs.  Defaults to all available GPUs.

        Returns
        -------
        dict
            ``{gpu_name: {"latency_ms": float, "peak_mem_mb": float,
            "throughput_ips": float}}``
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available — compare_hardware returns empty results")
            return {}

        if gpu_list is None:
            gpu_list = list(range(torch.cuda.device_count()))

        results: Dict[str, Dict[str, float]] = {}
        original_device = next(model.parameters()).device

        for gpu_id in gpu_list:
            device = torch.device(f"cuda:{gpu_id}")
            gpu_name = torch.cuda.get_device_name(gpu_id)
            model = model.to(device)

            # Create dummy input on device
            param = next(model.parameters())
            dummy = torch.randn(8, param.shape[-1] if param.dim() >= 2 else 32, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(self.warmup_runs):
                    _ = model(dummy)
                torch.cuda.synchronize(device)

            # Measure latency
            times: List[float] = []
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(self.profile_runs):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    _ = model(dummy)
                    torch.cuda.synchronize(device)
                    times.append((time.perf_counter() - t0) * 1000.0)

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            median_lat = float(np.median(times))
            throughput = dummy.shape[0] / (median_lat / 1000.0)

            results[gpu_name] = {
                "latency_ms": median_lat,
                "peak_mem_mb": peak_mem,
                "throughput_ips": throughput,
            }
            logger.info(f"GPU {gpu_name}: {median_lat:.2f} ms, {peak_mem:.1f} MB")

        # Restore original device
        model.to(original_device)
        return results

    # ------------------------------------------------------------------
    # export_chrome_trace
    # ------------------------------------------------------------------

    def export_chrome_trace(
        self,
        model: nn.Module,
        adapter: Any,
        output_path: Union[str, Path],
        input_data: Optional[torch.Tensor] = None,
    ) -> Path:
        """Export a Chrome trace file for visualisation in ``chrome://tracing``.

        Parameters
        ----------
        model : nn.Module
            Model to trace.
        adapter : Any
            Adapter instance (reserved).
        output_path : str or Path
            Output file path (e.g. ``"trace.json.gz"``).
        input_data : torch.Tensor, optional
            Input tensor.  If ``None``, a random tensor is generated.

        Returns
        -------
        Path
            Path to the written trace file.
        """
        model.eval()
        device = self._get_device(model)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_data is None:
            param = next(model.parameters(), None)
            if param is not None:
                last_dim = param.shape[-1]
                input_data = torch.randn(4, 32, last_dim, device=device)
            else:
                input_data = torch.randn(4, 32, device=device)

        if input_data.device != device:
            input_data = input_data.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_data)
            self._sync(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_data)
            self._sync(device)

        prof.export_chrome_trace(str(output_path))
        logger.info(f"Chrome trace exported to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _categorise_times(self, per_layer: Dict[str, float]) -> Dict[str, float]:
        """Aggregate per-layer times into functional categories."""
        cat_times: Dict[str, float] = defaultdict(float)
        for name, t_ms in per_layer.items():
            cat = self._layer_category(name)
            cat_times[cat] += t_ms
        return dict(cat_times)

    @staticmethod
    def _layer_category(name: str) -> str:
        """Map a layer name to a functional category."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["attn", "attention", "query", "key", "value"]):
            return "attention"
        if any(kw in name_lower for kw in ["norm", "layernorm", "batchnorm"]):
            return "normalization"
        if any(kw in name_lower for kw in ["embed", "token_embed", "pos_embed"]):
            return "embedding"
        if any(kw in name_lower for kw in ["fc", "linear", "mlp", "ffn"]):
            return "ffn"
        return "other"

    @staticmethod
    def _get_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _measure_peak_memory(
        self, model: nn.Module, input_data: torch.Tensor, device: torch.device
    ) -> float:
        if device.type != "cuda":
            return 0.0
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(input_data)
        self._sync(device)
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    @staticmethod
    def _memory_breakdown(model: nn.Module) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        for name, mod in model.named_modules():
            p_bytes = sum(
                p.numel() * p.element_size()
                for p in mod.parameters(recurse=False)
            )
            if p_bytes > 0:
                breakdown[name if name else "(root)"] = p_bytes / (1024 ** 2)
        return breakdown
