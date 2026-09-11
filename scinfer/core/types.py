"""Type definitions and enumerations for the scInfer framework.

Centralises all enum types so that model names, optimisation strategies,
and other categorical values are consistent across the codebase.
"""

from __future__ import annotations

from enum import Enum
from typing import List


class ModelType(str, Enum):
    """Architecture family of a single-cell foundation model."""

    ENCODER_ONLY = "encoder_only"
    ENCODER_DECODER = "encoder_decoder"
    DECODER_ONLY = "decoder_only"
    HYBRID = "hybrid"


class OptimizationType(str, Enum):
    """Category of inference optimisation."""

    QUANTIZATION = "quantization"
    FLASH_ATTENTION = "flash_attention"
    COMPILATION = "compilation"
    BATCH_SCHEDULING = "batch_scheduling"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    PRUNING = "pruning"
    KV_CACHE = "kv_cache"
    SPECULATIVE_DECODING = "speculative_decoding"


class DeviceType(str, Enum):
    """Supported compute devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Dtype(str, Enum):
    """Supported tensor dtypes."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT4 = "int4"


class MetricCategory(str, Enum):
    """Grouping for evaluation metrics."""

    INFERENCE = "inference"
    BIOLOGICAL = "biological"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: List[str] = ["scgpt", "geneformer", "scfoundation", "uce"]

SUPPORTED_OPTIMIZATIONS: List[str] = [e.value for e in OptimizationType]
