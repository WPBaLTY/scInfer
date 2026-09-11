"""Batch scheduling optimisation engine.

Implements intelligent batch scheduling strategies for inference,
including dynamic batching, padding optimisation, and sequence-length
bucketing to maximise GPU utilisation when processing large numbers
of cells.

Expected implementation phases:
- Phase 3: Dynamic batching
- Phase 4: Sequence-length bucketing
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Generator, Optional

import numpy as np
import torch
from torch import nn

from loguru import logger

from scinfer.adapters.base import BaseModelAdapter
from scinfer.engines.base import BaseOptimizationEngine
from scinfer.core.registry import register_engine


# ---------------------------------------------------------------------------
# ScheduledModel wrapper
# ---------------------------------------------------------------------------

class ScheduledModel(nn.Module):
    """Wrapper that applies dynamic batching logic during forward pass.

    Parameters
    ----------
    model : nn.Module
        The underlying model.
    max_batch_size : int
        Maximum batch size allowed.
    strategy : str
        Batching strategy name.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 256,
        strategy: str = "dynamic",
    ) -> None:
        super().__init__()
        self.model = model
        self.max_batch_size = max_batch_size
        self.strategy = strategy

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with automatic batch splitting if input exceeds max_batch_size.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose first dimension is the batch dimension.

        Returns
        -------
        torch.Tensor
            Model output.
        """
        batch_size = x.shape[0]

        if batch_size <= self.max_batch_size:
            return self.model(x, *args, **kwargs)

        # Split into sub-batches
        outputs = []
        for start in range(0, batch_size, self.max_batch_size):
            end = min(start + self.max_batch_size, batch_size)
            chunk = x[start:end]
            out = self.model(chunk, *args, **kwargs)
            outputs.append(out)

        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# BatchSchedulingEngine
# ---------------------------------------------------------------------------

@register_engine("batch_scheduling")
class BatchSchedulingEngine(BaseOptimizationEngine):
    """Inference optimisation engine for batch scheduling.

    Optimises how cells are batched for inference:
    - Dynamic batching: adjust batch size based on available memory.
    - Sequence-length bucketing: group cells by gene count to minimise padding.
    - Prefetching: overlap data loading with GPU computation.
    """

    def __init__(self) -> None:
        """Initialise the batch scheduling engine."""
        self._n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._gpu_memory_mb: list[float] = []
        if torch.cuda.is_available():
            for i in range(self._n_gpus):
                props = torch.cuda.get_device_properties(i)
                self._gpu_memory_mb.append(props.total_memory / (1024 ** 2))

        self._scheduled_models: Dict[str, nn.Module] = {}
        logger.debug(
            "BatchSchedulingEngine initialised — GPUs: {}, memory: {} MB",
            self._n_gpus,
            self._gpu_memory_mb or ["N/A"],
        )

    # ------------------------------------------------------------------
    # BaseOptimizationEngine interface
    # ------------------------------------------------------------------

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Wrap the model with batch scheduling logic.

        Parameters
        ----------
        model : torch.nn.Module
            The model to wrap.
        config : dict
            Scheduling configuration (strategy, bucket sizes, etc.).

        Returns
        -------
        torch.nn.Module
            The model wrapped with batch scheduling.
        """
        strategy = config.get("strategy", "dynamic")
        max_batch_size = config.get("max_batch_size", 256)
        budget_mb = config.get("budget_mb")

        # Auto-compute batch size from memory budget if provided
        if budget_mb is not None:
            max_batch_size = self.auto_batch_size(model, budget_mb)
            logger.info("Auto batch size for {} MB budget: {}", budget_mb, max_batch_size)

        scheduled = ScheduledModel(
            model=model,
            max_batch_size=max_batch_size,
            strategy=strategy,
        )

        model_id = f"sched_{strategy}_{id(model) % 10000}"
        self._scheduled_models[model_id] = scheduled

        logger.info(
            "Applied batch scheduling: strategy={}, max_batch_size={}",
            strategy, max_batch_size,
        )
        return scheduled

    def benchmark(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, float]:
        """Benchmark batch scheduling performance.

        Tests different batch sizes and reports throughput for each.

        Parameters
        ----------
        model : torch.nn.Module
            The model with batch scheduling.
        input_data : torch.Tensor
            Representative input tensor.

        Returns
        -------
        dict[str, float]
            Metrics including throughput, GPU utilisation, and padding waste.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        if isinstance(input_data, torch.Tensor) and input_data.device != device:
            input_data = input_data.to(device)

        model.eval()

        # Test a range of batch sizes
        batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
        batch_sizes = [b for b in batch_sizes if b <= input_data.shape[0]]

        best_throughput = 0.0
        best_batch_size = 1
        results: Dict[str, float] = {}

        for bs in batch_sizes:
            data = input_data[:bs]
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(data)

            # Measure
            n_runs = 20
            times: list[float] = []
            with torch.no_grad():
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = model(data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)

            latency_ms = float(np.mean(times)) * 1000.0
            throughput = bs / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            results[f"batch_{bs}_latency_ms"] = round(latency_ms, 4)
            results[f"batch_{bs}_throughput_fps"] = round(throughput, 2)

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = bs

        # Peak memory
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results.update(
            best_batch_size=best_batch_size,
            best_throughput_fps=round(best_throughput, 2),
            peak_memory_mb=round(peak_memory_mb, 2),
        )

        logger.info(
            "Batch scheduling benchmark: best_bs={}, throughput={:.1f} fps",
            best_batch_size, best_throughput,
        )
        return results

    def is_compatible(self, adapter: BaseModelAdapter) -> bool:
        """Check if batch scheduling is compatible with the given adapter.

        Batch scheduling is a wrapper-level optimisation and is compatible
        with all models.

        Parameters
        ----------
        adapter : BaseModelAdapter
            The model adapter to check.

        Returns
        -------
        bool
            ``True`` if compatible.
        """
        return True

    # ------------------------------------------------------------------
    # Scheduling-specific methods
    # ------------------------------------------------------------------

    def auto_batch_size(self, model: nn.Module, budget_mb: float) -> int:
        """Compute optimal batch size given a GPU memory budget.

        Parameters
        ----------
        model : torch.nn.Module
            The model to estimate memory for.
        budget_mb : float
            Available GPU memory budget in MB.

        Returns
        -------
        int
            Recommended batch size (>= 1).
        """
        # Estimate model parameter memory
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_mem_mb = params_bytes / (1024 ** 2)

        # Estimate per-sample activation memory (rough: 2x model size for activations)
        activation_per_sample_mb = model_mem_mb * 2.0 / max(1, 32)  # rough estimate

        available_mb = budget_mb - model_mem_mb
        if available_mb <= 0:
            logger.warning(
                "Model size ({:.1f} MB) exceeds budget ({:.1f} MB). "
                "Using batch_size=1.",
                model_mem_mb, budget_mb,
            )
            return 1

        batch_size = max(1, int(available_mb / max(activation_per_sample_mb, 0.1)))

        # Clamp to reasonable range
        batch_size = min(batch_size, 4096)

        logger.debug(
            "auto_batch_size: model={:.1f} MB, budget={:.1f} MB → batch_size={}",
            model_mem_mb, budget_mb, batch_size,
        )
        return batch_size

    def dynamic_batching(
        self,
        model: nn.Module,
        data_loader: Any,
        budget_mb: float = 4096.0,
    ) -> list[torch.Tensor]:
        """Run inference with dynamic batch size adjustment.

        Starts with a large batch and reduces if OOM is encountered.

        Parameters
        ----------
        model : torch.nn.Module
            The model to run inference on.
        data_loader : iterable
            An iterable yielding batches of data.
        budget_mb : float
            GPU memory budget in MB.

        Returns
        -------
        list[torch.Tensor]
            List of model outputs for each batch.
        """
        device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )
        model.eval()

        current_batch_size = self.auto_batch_size(model, budget_mb)
        outputs: list[torch.Tensor] = []

        for batch in data_loader:
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch

            # Try forward pass; reduce batch size on OOM
            actual_bs = min(current_batch_size, batch.shape[0])
            success = False

            while actual_bs >= 1 and not success:
                try:
                    sub_batch = batch[:actual_bs] if isinstance(batch, torch.Tensor) else batch
                    with torch.no_grad():
                        out = model(sub_batch)
                    outputs.append(out)
                    success = True
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        actual_bs = max(1, actual_bs // 2)
                        logger.warning(
                            "OOM at batch_size={}; retrying with {}",
                            actual_bs * 2, actual_bs,
                        )
                    else:
                        raise

            if not success:
                logger.error("Failed to process batch even with batch_size=1")

        return outputs

    def multi_gpu_inference(
        self,
        model: nn.Module,
        data: torch.Tensor,
        device_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Run inference across multiple GPUs using DataParallel.

        Parameters
        ----------
        model : torch.nn.Module
            The model to parallelise.
        data : torch.Tensor
            Input data tensor.
        device_ids : list[int], optional
            GPU device IDs to use. Defaults to all available GPUs.

        Returns
        -------
        torch.Tensor
            Model output gathered from all GPUs.
        """
        if not torch.cuda.is_available() or self._n_gpus < 2:
            logger.warning(
                "Multi-GPU inference requires >= 2 CUDA devices. "
                "Falling back to single-GPU."
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            data = data.to(device)
            model.eval()
            with torch.no_grad():
                return model(data)

        if device_ids is None:
            device_ids = list(range(self._n_gpus))

        logger.info("Multi-GPU inference on devices: {}", device_ids)

        model = model.to(f"cuda:{device_ids[0]}")
        data = data.to(f"cuda:{device_ids[0]}")
        model.eval()

        parallel_model = nn.DataParallel(model, device_ids=device_ids)

        with torch.no_grad():
            output = parallel_model(data)

        return output

    @contextlib.contextmanager
    def memory_budget(self, model: nn.Module, budget_mb: float) -> Generator[int, None, None]:
        """Context manager that yields the optimal batch size for a memory budget.

        Parameters
        ----------
        model : torch.nn.Module
            The model to estimate for.
        budget_mb : float
            GPU memory budget in MB.

        Yields
        ------
        int
            Recommended batch size.
        """
        batch_size = self.auto_batch_size(model, budget_mb)

        # Reset peak memory stats if CUDA
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.debug("memory_budget context: budget={:.0f} MB → batch_size={}", budget_mb, batch_size)

        try:
            yield batch_size
        finally:
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                if peak > budget_mb:
                    logger.warning(
                        "Peak memory ({:.0f} MB) exceeded budget ({:.0f} MB)",
                        peak, budget_mb,
                    )

    def validate(self, model: nn.Module, config: Dict[str, Any]) -> bool:
        """Validate GPU availability and batch size configuration.

        Parameters
        ----------
        model : torch.nn.Module
            The model to validate.
        config : dict
            Scheduling configuration.

        Returns
        -------
        bool
            ``True`` if validation passes.
        """
        max_batch_size = config.get("max_batch_size", 256)

        # Check batch size is reasonable
        if max_batch_size < 1:
            logger.warning("max_batch_size must be >= 1, got {}", max_batch_size)
            return False
        if max_batch_size > 65536:
            logger.warning("max_batch_size {} is unreasonably large", max_batch_size)
            return False

        # GPU check: if GPU available, check memory
        if torch.cuda.is_available():
            total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            # Estimate model memory
            params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            model_mem_mb = params_bytes / (1024 ** 2)
            if model_mem_mb > total_mem_mb * 0.9:
                logger.warning(
                    "Model size ({:.0f} MB) exceeds 90% of GPU memory ({:.0f} MB)",
                    model_mem_mb, total_mem_mb,
                )
                return False
        else:
            logger.info("No GPU available; batch scheduling will run on CPU")

        return True

    def revert(self, model: nn.Module) -> nn.Module:
        """Revert optimization - returns model unchanged for non-reversible engines."""
        logger.info(f"{self.engine_name}: revert not applicable, returning model as-is")
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        """Return the engine identifier."""
        return "batch_scheduling"

    @property
    def optimization_type(self) -> str:
        """Return the optimization category."""
        return "batch_scheduling"
