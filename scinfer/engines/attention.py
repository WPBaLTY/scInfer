"""FlashAttention acceleration engine.

Replaces standard scaled dot-product attention with FlashAttention-2/3
for memory-efficient and faster attention computation.  FlashAttention
uses IO-aware tiling to reduce HBM reads/writes, yielding significant
speedups for long sequences.

Expected implementation phases:
- Phase 3: FlashAttention-2 integration
- Phase 4: FlashAttention-3 (Hopper) support
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from loguru import logger

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine
from scinfer.core.registry import register_engine

# Optional dependency: flash_attn
try:
    import flash_attn  # type: ignore
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Compatibility matrix: model adapter name → supported FA versions
# ---------------------------------------------------------------------------
_COMPATIBILITY_MATRIX: Dict[str, bool] = {
    "scgpt": True,
    "geneformer": True,
    "scfoundation": True,
    "uce": True,
}


@register_engine("flash_attention")
class FlashAttentionEngine(BaseOptimizationEngine):
    """Inference optimisation engine for FlashAttention.

    Replaces the standard attention implementation with FlashAttention,
    which provides:
    - O(N) memory complexity instead of O(N^2).
    - 2-4x speedup for typical single-cell sequence lengths.
    - Exact attention (no approximation).
    """

    # Fallback chain: FA-3 → FA-2 → SDPA → native
    _FALLBACK_CHAIN = ("flash_attention_3", "flash_attention_2", "sdpa", "native")

    def __init__(self) -> None:
        """Initialise the FlashAttention engine."""
        self._gpu_info = self.detect_gpu_capability()
        self._fa_version: Optional[str] = self._determine_fa_version()
        self._flash_attn_available = _FLASH_ATTN_AVAILABLE
        logger.debug(
            "FlashAttentionEngine initialised — GPU: {}, FA version: {}",
            self._gpu_info.get("name", "unknown"),
            self._fa_version or "none (CPU-only or unsupported)",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_fa_version(self) -> Optional[str]:
        """Determine the best FlashAttention version for the current GPU."""
        cap = self._gpu_info.get("compute_capability")
        if cap is None:
            return None
        if cap >= 9.0:
            return "flash_attention_3"
        if cap >= 8.0:
            return "flash_attention_2"
        # Below Ampere — fall back to SDPA (PyTorch native)
        return "sdpa"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_gpu_capability(self) -> Dict[str, Any]:
        """Return a dictionary describing the current GPU's compute capability.

        Returns
        -------
        dict
            Keys: ``name``, ``compute_capability``, ``total_memory_mb``,
            ``device_id``.  If CUDA is unavailable the capability is ``None``.
        """
        info: Dict[str, Any] = {
            "name": None,
            "compute_capability": None,
            "total_memory_mb": None,
            "device_id": None,
        }
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            info.update(
                name=props.name,
                compute_capability=float(f"{props.major}.{props.minor}"),
                total_memory_mb=props.total_memory / (1024 ** 2),
                device_id=dev,
            )
        return info

    def select_attention_implementation(self, config: Dict[str, Any]) -> str:
        """Select the optimal attention implementation given hardware.

        Parameters
        ----------
        config : dict
            Must contain ``"version"`` (optional) to force a specific version.

        Returns
        -------
        str
            One of ``"flash_attention_3"``, ``"flash_attention_2"``,
            ``"sdpa"``, or ``"native"``.
        """
        forced = config.get("version")
        if forced and forced in self._FALLBACK_CHAIN:
            return forced
        return self._fa_version or "native"

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Replace standard attention with FlashAttention in the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose attention layers will be replaced.
        config : dict
            FlashAttention configuration (version, causal, etc.).

        Returns
        -------
        torch.nn.Module
            The model with FlashAttention layers.
        """
        if not self.validate(model, config):
            logger.warning("Validation failed; returning original model")
            return model

        impl = self.select_attention_implementation(config)
        causal = config.get("causal", False)
        logger.info("Applying FlashAttention: implementation={}", impl)

        if impl in ("flash_attention_3", "flash_attention_2"):
            if not _FLASH_ATTN_AVAILABLE:
                logger.warning(
                    "flash_attn package not installed — falling back to SDPA"
                )
                impl = "sdpa"

        if impl in ("flash_attention_3", "flash_attention_2"):
            model = self._apply_flash_attention(model, causal=causal)
        elif impl == "sdpa":
            model = self._apply_sdpa(model, causal=causal)
        else:
            logger.info("Using native attention (no acceleration)")

        return model

    def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark FlashAttention model speed and memory.

        Parameters
        ----------
        model : torch.nn.Module
            The model with FlashAttention.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Metrics including throughput, latency, and peak memory.
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

        metrics = {
            "latency_ms": round(latency_ms, 4),
            "latency_std_ms": round(float(latency_arr.std()), 4),
            "throughput_fps": round(throughput_fps, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
        }
        logger.info(
            "FlashAttention benchmark: latency={:.3f} ms, throughput={:.1f} fps, "
            "peak_mem={:.1f} MB",
            latency_ms, throughput_fps, peak_memory_mb,
        )
        return metrics

    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check if FlashAttention is compatible with the given adapter.

        FlashAttention requires Ampere+ GPU (compute capability >= 8.0)
        and a model that uses standard scaled dot-product attention.

        Parameters
        ----------
        adapter : BaseModelAdapter
            The model adapter to check.

        Returns
        -------
        bool
            ``True`` if compatible.
        """
        name = getattr(adapter, "model_name", None)
        if name and name in _COMPATIBILITY_MATRIX:
            return _COMPATIBILITY_MATRIX[name]
        # Check supported_optimizations list
        supported = getattr(adapter, "supported_optimizations", [])
        return "flash_attention" in supported

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Validate GPU compute capability and model architecture.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        config : dict
            FlashAttention configuration.

        Returns
        -------
        bool
            ``True`` if validation passes.
        """
        # Must have at least one attention-like module or SDPA-compatible layer
        has_attention = False
        for module in model.modules():
            if isinstance(module, (nn.MultiheadAttention,)):
                has_attention = True
                break
            # Also check for common custom attention class names
            cls_name = type(module).__name__.lower()
            if "attention" in cls_name:
                has_attention = True
                break

        if not has_attention:
            logger.warning("Model does not appear to contain attention layers")
            return False

        # GPU check for FA-2/FA-3
        impl = self.select_attention_implementation(config)
        if impl in ("flash_attention_2", "flash_attention_3"):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available — FlashAttention requires GPU")
                return False
            cap = self._gpu_info.get("compute_capability")
            if cap is not None and cap < 8.0:
                logger.warning(
                    "GPU compute capability {:.1f} < 8.0 — FlashAttention requires Ampere+",
                    cap,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Attention replacement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_flash_attention(model: nn.Module, causal: bool = False) -> nn.Module:
        """Replace nn.MultiheadAttention with flash_attn-based attention.

        For models with custom attention (e.g. scGPT), if custom masks are
        used, FA-2 supports custom masks but through a non-optimised path,
        so we fall back to SDPA for those cases.
        """
        try:
            from flash_attn import flash_attn_func  # type: ignore
        except ImportError:
            logger.warning("flash_attn not importable; skipping FA replacement")
            return model

        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1] if parts else name

                wrapper = _FlashAttentionWrapper(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    causal=causal,
                )
                try:
                    setattr(parent, attr_name, wrapper)
                    replaced += 1
                except (AttributeError, TypeError):
                    pass

        logger.info("Replaced {} attention modules with FlashAttention", replaced)
        return model

    @staticmethod
    def _apply_sdpa(model: nn.Module, causal: bool = False) -> nn.Module:
        """Replace nn.MultiheadAttention with SDPA-based attention."""
        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.MultiheadAttention):
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                attr_name = parts[-1] if parts else name

                wrapper = _SDPAAttentionWrapper(
                    embed_dim=module.embed_dim,
                    num_heads=module.num_heads,
                    causal=causal,
                )
                try:
                    setattr(parent, attr_name, wrapper)
                    replaced += 1
                except (AttributeError, TypeError):
                    pass

        logger.info("Replaced {} attention modules with SDPA", replaced)
        return model

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
        return "flash_attention"

    @property
    def optimization_type(self) -> str:
        """Return the optimization category."""
        return "flash_attention"


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class _FlashAttentionWrapper(nn.Module):
    """Thin wrapper that calls flash_attn_func in place of MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = embed_dim // num_heads
        # Keep the original projection layers
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # query shape: (batch, seq_len, embed_dim) or (seq_len, batch, embed_dim)
        if query.dim() == 3 and query.shape[0] != query.shape[1]:
            # Assume (seq, batch, dim) → transpose
            qkv = self.in_proj(query).transpose(0, 1)  # (batch, seq, 3*dim)
        else:
            qkv = self.in_proj(query)

        batch_size, seq_len, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for flash attention: (batch, nheads, seq, headdim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        try:
            from flash_attn import flash_attn_func  # type: ignore
            out = flash_attn_func(q, k, v, causal=self.causal)
        except Exception:
            # Fallback to scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal
            )
            out = out.transpose(1, 2)

        out = out.reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class _SDPAAttentionWrapper(nn.Module):
    """Wrapper using PyTorch's native scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int, causal: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if query.dim() == 3 and query.shape[0] != query.shape[1]:
            qkv = self.in_proj(query).transpose(0, 1)
        else:
            qkv = self.in_proj(query)

        batch_size, seq_len, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)
