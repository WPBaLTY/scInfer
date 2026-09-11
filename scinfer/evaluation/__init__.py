"""scinfer.evaluation — Profiling, benchmarking, and metrics for inference.

This module provides tools for:
- Profiling model inference to identify bottlenecks.
- Running standardised benchmarks across models.
- Computing both inference metrics (throughput, latency, memory) and
  biological metrics (ARI, NMI, F1, PCC).
- Generating human-readable reports.
"""

from scinfer.evaluation.benchmark import (
    BenchmarkProtocol,
    BenchmarkResult,
    BenchmarkRunner,
)
from scinfer.evaluation.bio_validation import (
    BioValidationPipeline,
    BioValidationResult,
)
from scinfer.evaluation.metrics.biological import BiologicalMetrics
from scinfer.evaluation.metrics.inference import InferenceMetrics
from scinfer.evaluation.profiler import InferenceProfiler, ProfileResult
from scinfer.evaluation.report import ReportGenerator

__all__ = [
    "InferenceProfiler",
    "ProfileResult",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkProtocol",
    "ReportGenerator",
    "InferenceMetrics",
    "BiologicalMetrics",
    "BioValidationPipeline",
    "BioValidationResult",
]
