"""scinfer.engines — Modular inference optimisation engines.

Each engine implements a specific optimisation strategy (quantization,
attention acceleration, compilation, batch scheduling) behind the common
:class:`BaseOptimizationEngine` protocol.
"""

from loguru import logger

from scinfer.engines.base import BaseOptimizationEngine

try:
    from scinfer.engines.quantization import QuantizationEngine
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    QuantizationEngine = None  # type: ignore[assignment,misc]

try:
    from scinfer.engines.attention import FlashAttentionEngine
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    FlashAttentionEngine = None  # type: ignore[assignment,misc]

try:
    from scinfer.engines.compilation import CompilationEngine
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    CompilationEngine = None  # type: ignore[assignment,misc]

try:
    from scinfer.engines.scheduling import BatchSchedulingEngine
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    BatchSchedulingEngine = None  # type: ignore[assignment,misc]

try:
    from scinfer.engines.combiner import OptimizationCombiner
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    OptimizationCombiner = None  # type: ignore[assignment,misc]

try:
    from scinfer.engines.distillation import DistillationEngine
except ImportError as e:  # pragma: no cover
    logger.warning(f"Engine module not available: {e}")
    DistillationEngine = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseOptimizationEngine",
    "QuantizationEngine",
    "FlashAttentionEngine",
    "CompilationEngine",
    "BatchSchedulingEngine",
    "OptimizationCombiner",
    "DistillationEngine",
]
