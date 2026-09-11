"""scInfer — Systematic inference optimization for single-cell foundation models.

scInfer provides a unified framework for accelerating inference of
single-cell foundation models (scGPT, Geneformer, scFoundation, UCE)
using LLM inference optimization techniques.

Quick Start
-----------
>>> import scinfer
>>> adata = scinfer.encode(adata, model="scgpt", optimize=True)

Top-level API
-------------
- :func:`encode` — One-line cell embedding generation.
- :func:`optimize` — Apply optimisation strategies to a model.
- :func:`benchmark` — Run cross-model benchmarks.
- :func:`profile` — Inference profiling and bottleneck analysis.
"""

__version__ = "1.0.0"

from scinfer.api import OptimizedModel, benchmark, encode, optimize, profile
from scinfer.core import Config, EngineRegistry, ModelRegistry

__all__ = [
    "__version__",
    # Top-level API
    "encode",
    "optimize",
    "benchmark",
    "profile",
    # Data classes
    "OptimizedModel",
    # Core
    "Config",
    "ModelRegistry",
    "EngineRegistry",
]
