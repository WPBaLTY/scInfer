"""scinfer.evaluation.metrics — Inference and biological quality metrics.

Provides standardised metric computations for evaluating both the
performance and biological fidelity of optimised models.
"""

from scinfer.evaluation.metrics.inference import (
    InferenceMetrics,
    LatencyResult,
    MemoryResult,
    StepTimingResult,
    ThroughputResult,
    compute_latency,
    compute_memory_usage,
    compute_throughput,
)
from scinfer.evaluation.metrics.biological import (
    BatchCorrectionResult,
    BiologicalMetrics,
    CellTypeResult,
    DEResult,
    EmbeddingFidelityResult,
    GeneNetworkResult,
    PerturbationResult,
    compute_ari,
    compute_f1,
    compute_nmi,
    compute_pcc,
)

__all__ = [
    # Classes
    "InferenceMetrics",
    "BiologicalMetrics",
    # Result dataclasses — inference
    "ThroughputResult",
    "LatencyResult",
    "MemoryResult",
    "StepTimingResult",
    # Result dataclasses — biological
    "CellTypeResult",
    "EmbeddingFidelityResult",
    "PerturbationResult",
    "GeneNetworkResult",
    "BatchCorrectionResult",
    "DEResult",
    # Convenience functions
    "compute_throughput",
    "compute_latency",
    "compute_memory_usage",
    "compute_ari",
    "compute_nmi",
    "compute_f1",
    "compute_pcc",
]
