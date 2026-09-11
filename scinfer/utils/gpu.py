"""GPU detection and management utilities.

Provides functions for detecting available GPUs, querying device
properties, and selecting the optimal compute device for inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from loguru import logger


def get_device(device: str = "auto") -> torch.device:
    """Resolve and return the compute device.

    Parameters
    ----------
    device : str
        Device specification: ``"auto"`` (select best available),
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Returns
    -------
    torch.device
        The resolved PyTorch device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
            logger.debug("Auto-selected device: {}", resolved)
        else:
            resolved = torch.device("cpu")
            logger.debug("Auto-selected device: {} (CUDA not available)", resolved)
        return resolved

    # Handle "cuda:N" parsing
    if device.startswith("cuda:"):
        try:
            index = int(device.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid CUDA device specification: {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device!r} but CUDA is not available")
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device!r} but only {torch.cuda.device_count()} CUDA device(s) available"
            )
        return torch.device(device)

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested 'cuda' but CUDA is not available")
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    # Fallback: let torch validate the string
    return torch.device(device)


def get_gpu_info(device_index: int = 0) -> Dict[str, Any]:
    """Query GPU properties for the specified device.

    Parameters
    ----------
    device_index : int
        CUDA device index.

    Returns
    -------
    dict[str, Any]
        GPU information including name, total memory, compute capability,
        and current memory usage.  Returns an empty dict if no GPU is
        available.
    """
    if not torch.cuda.is_available():
        return {}

    if device_index >= torch.cuda.device_count():
        logger.warning(
            "Requested GPU index {} but only {} device(s) available",
            device_index,
            torch.cuda.device_count(),
        )
        return {}

    props = torch.cuda.get_device_properties(device_index)

    total_memory_mb = props.total_memory / (1024 ** 2)
    used_memory_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    free_memory_mb = total_memory_mb - used_memory_mb

    return {
        "name": props.name,
        "total_memory_mb": round(total_memory_mb, 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "used_memory_mb": round(used_memory_mb, 1),
        "free_memory_mb": round(free_memory_mb, 1),
    }


def list_gpus() -> List[Dict[str, Any]]:
    """List all available CUDA GPUs with basic info.

    Returns
    -------
    list of dict
        One dict per GPU with keys: ``index``, ``name``, ``total_memory_mb``,
        ``compute_capability``.
    """
    if not torch.cuda.is_available():
        return []

    gpus: List[Dict[str, Any]] = []
    for idx in range(torch.cuda.device_count()):
        info = get_gpu_info(idx)
        if info:
            info["index"] = idx
            gpus.append(info)

    logger.debug("Found {} CUDA GPU(s)", len(gpus))
    return gpus


def has_flash_attention_support(device_index: int = 0) -> bool:
    """Check if the GPU supports FlashAttention (compute capability >= 8.0).

    Parameters
    ----------
    device_index : int
        CUDA device index.

    Returns
    -------
    bool
        ``True`` if FlashAttention is supported.
    """
    if not torch.cuda.is_available():
        return False

    if device_index >= torch.cuda.device_count():
        return False

    props = torch.cuda.get_device_properties(device_index)
    supported = (props.major, props.minor) >= (8, 0)
    logger.debug(
        "GPU {} ({}): compute capability {}.{}, FlashAttention supported: {}",
        device_index,
        props.name,
        props.major,
        props.minor,
        supported,
    )
    return supported
