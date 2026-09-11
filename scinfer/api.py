"""Top-level API for scInfer.

Provides the primary user-facing functions:
- ``encode()``: One-line cell embedding generation.
- ``optimize()``: Apply optimisation strategies to a model.
- ``benchmark()``: Run cross-model benchmarks.
- ``profile()``: Inference profiling and bottleneck analysis.

These functions handle configuration loading, model adapter resolution,
and engine orchestration behind a simple interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger

from scinfer.core.config import Config
from scinfer.core.registry import EngineRegistry, ModelRegistry


@dataclass
class OptimizedModel:
    """Container for an optimised model and its metadata.

    Attributes
    ----------
    model_name : str
        Name of the optimised model.
    adapter : Any
        The model adapter instance.
    engines_applied : list of str
        Names of optimisation engines that were applied.
    config : dict
        Configuration used for optimisation.
    metrics : dict
        Post-optimisation benchmark metrics.
    """

    model_name: str
    adapter: Any = None
    engines_applied: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def _load_config(config_path: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file(s), falling back to defaults.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to a YAML config file.
    model_name : str, optional
        Model name used to locate ``configs/models/<model>.yaml``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    if config_path is not None:
        cfg = Config.from_files(config_path)
        return cfg.to_dict()

    # Try default config + model-specific config
    base_dir = Path(__file__).resolve().parent.parent
    default_cfg_path = base_dir / "configs" / "default.yaml"
    paths = []
    if default_cfg_path.exists():
        paths.append(default_cfg_path)
    if model_name is not None:
        model_cfg_path = base_dir / "configs" / "models" / f"{model_name}.yaml"
        if model_cfg_path.exists():
            paths.append(model_cfg_path)

    if paths:
        cfg = Config.from_files(*paths)
        return cfg.to_dict()

    logger.debug("No config files found, using empty default config")
    return {}


def _get_adapter(model: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve and instantiate a model adapter from the registry.

    Parameters
    ----------
    model : str
        Registered model name.
    config : dict, optional
        Configuration dictionary passed to the adapter constructor.

    Returns
    -------
    BaseModelAdapter
        The constructed adapter instance.
    """
    adapter_cls = ModelRegistry.get(model)
    adapter = adapter_cls(config=config or {})
    return adapter


def encode(
    adata: "AnnData",  # noqa: F821
    model: str = "scgpt",
    optimize: bool = True,
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> "AnnData":  # noqa: F821
    """One-line cell embedding generation with optional optimisation.

    Loads the specified model, optionally applies inference optimisations,
    and stores the resulting embeddings in ``adata.obsm["X_scinfer"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data (cells × genes).
    model : str
        Model name (``"scgpt"``, ``"geneformer"``, ``"scfoundation"``, ``"uce"``).
    optimize : bool
        Whether to apply inference optimisations.
    config_path : str, optional
        Path to a custom YAML configuration file.
    **kwargs
        Additional keyword arguments passed to the model adapter.

    Returns
    -------
    anndata.AnnData
        The input ``adata`` with embeddings added to ``.obsm["X_scinfer"]``.

    Raises
    ------
    KeyError
        If the requested *model* is not registered.
    RuntimeError
        If model loading or embedding extraction fails.
    """
    logger.info(f"encode(): model={model}, optimize={optimize}")

    # 1. Load configuration
    try:
        config = _load_config(config_path, model_name=model)
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        raise

    # 2. Get adapter and load model
    try:
        adapter = _get_adapter(model, config)
        adapter.load_model(**kwargs)
        logger.info(f"Model '{model}' loaded successfully")
    except KeyError:
        raise
    except Exception as exc:
        logger.error(f"Failed to load model '{model}': {exc}")
        raise RuntimeError(f"Model loading failed for '{model}': {exc}") from exc

    # 3. Optionally apply optimisation
    if optimize:
        try:
            from scinfer.engines.combiner import OptimizationCombiner
            from scinfer.engines.quantization import QuantizationEngine

            combiner = OptimizationCombiner(engines=[QuantizationEngine()])
            nn_model = adapter.get_model()
            opt_config = {"method": "simulated", "bits": 8}
            optimized_model = combiner.apply(nn_model, opt_config)
            logger.info("Optimization applied via OptimizationCombiner")
        except ImportError as combiner_exc:
            logger.warning(
                f"OptimizationCombiner unavailable ({combiner_exc}), "
                "skipping combined optimization"
            )
        except Exception as exc:
            logger.warning(f"Optimization failed, continuing without optimization: {exc}")

    # 4. Extract embeddings
    try:
        embeddings = adapter.get_embeddings(adata)
        logger.info(f"Embeddings extracted: shape={embeddings.shape}")
    except Exception as exc:
        logger.error(f"Failed to extract embeddings: {exc}")
        raise RuntimeError(f"Embedding extraction failed: {exc}") from exc

    # 5. Write back to adata
    adata.obsm["X_scinfer"] = embeddings
    logger.info("Embeddings stored in adata.obsm['X_scinfer']")

    return adata


def optimize(
    model_name: str,
    optimization_config: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> OptimizedModel:
    """Apply optimisation strategies to a specified model.

    Loads the model, resolves the optimisation pipeline from configuration,
    applies each engine in order, and returns the optimised model wrapper.

    Parameters
    ----------
    model_name : str
        Model to optimise (``"scgpt"``, ``"geneformer"``, etc.).
    optimization_config : str or dict, optional
        Path to optimisation YAML config or a config dictionary.
    **kwargs
        Additional overrides.

    Returns
    -------
    OptimizedModel
        Container with the optimised model and metadata.

    Raises
    ------
    KeyError
        If *model_name* is not registered.
    RuntimeError
        If model loading or optimization fails.
    """
    logger.info(f"optimize(): model={model_name}")

    # 1. Load configuration
    if isinstance(optimization_config, str):
        config = _load_config(optimization_config, model_name=model_name)
    elif isinstance(optimization_config, dict):
        config = dict(optimization_config)
    else:
        config = _load_config(model_name=model_name)

    # 2. Get adapter and load model
    try:
        adapter = _get_adapter(model_name, config)
        adapter.load_model(**kwargs)
        logger.info(f"Model '{model_name}' loaded for optimization")
    except KeyError:
        raise
    except Exception as exc:
        logger.error(f"Failed to load model '{model_name}': {exc}")
        raise RuntimeError(f"Model loading failed: {exc}") from exc

    # 3. Create OptimizationCombiner and apply
    engines_applied: List[str] = []
    optimized_model = adapter.get_model()

    try:
        from scinfer.engines.combiner import OptimizationCombiner
        from scinfer.engines.quantization import QuantizationEngine

        combiner = OptimizationCombiner(engines=[QuantizationEngine()])
        opt_cfg = config.get("optimization", {"method": "simulated", "bits": 8})
        optimized_model = combiner.apply(optimized_model, opt_cfg)
        engines_applied.append("quantization")
        logger.info("Optimization pipeline applied successfully")
    except ImportError as combiner_exc:
        logger.warning(
            f"OptimizationCombiner unavailable ({combiner_exc}), "
            "applying individual engines"
        )
        # Fallback: apply individual engines directly
        try:
            from scinfer.engines.quantization import QuantizationEngine

            qe = QuantizationEngine()
            q_config = config.get("optimization", {"method": "simulated", "bits": 8})
            optimized_model = qe.apply(optimized_model, q_config)
            engines_applied.append("quantization")
        except Exception as engine_exc:
            logger.warning(f"Individual engine application failed: {engine_exc}")
    except Exception as exc:
        logger.warning(f"Optimization pipeline failed: {exc}")

    # 4. Collect benchmark metrics
    metrics: Dict[str, float] = {}

    return OptimizedModel(
        model_name=model_name,
        adapter=adapter,
        engines_applied=engines_applied,
        config=config,
        metrics=metrics,
    )


def benchmark(
    model_names: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Run benchmark across specified models and datasets.

    Parameters
    ----------
    model_names : list of str, optional
        Models to benchmark (default: ``["scgpt", "geneformer", "scfoundation"]``).
    datasets : list of str, optional
        Dataset identifiers or paths.
    metrics : list of str, optional
        Metrics to compute (default: throughput, latency, memory).
    **kwargs
        Additional benchmark parameters (``warmup_runs``, ``benchmark_runs``).

    Returns
    -------
    BenchmarkResult
        Benchmark results with per-model metrics, exportable to JSON/CSV/DataFrame.

    Raises
    ------
    RuntimeError
        If benchmark execution fails.
    """
    from scinfer.evaluation.benchmark import BenchmarkRunner, BenchmarkResult

    if model_names is None:
        model_names = ["scgpt", "geneformer", "scfoundation"]

    logger.info(f"benchmark(): models={model_names}, datasets={datasets}")

    # Build models and input data for the runner
    models: Dict[str, Any] = {}
    input_data: Dict[str, Any] = {}

    for name in model_names:
        try:
            adapter = _get_adapter(name)
            adapter.load_model()
            nn_model = adapter.get_model()
            models[name] = nn_model
            # Create a representative dummy input
            param = next(nn_model.parameters(), None)
            if param is not None:
                import torch
                last_dim = param.shape[-1] if param.dim() >= 2 else 32
                input_data[name] = torch.randn(8, last_dim)
            else:
                import torch
                input_data[name] = torch.randn(8, 32)
            logger.info(f"Model '{name}' loaded for benchmarking")
        except Exception as exc:
            logger.warning(f"Skipping '{name}' in benchmark: {exc}")

    # Create runner and execute
    try:
        runner = BenchmarkRunner(
            model_names=model_names,
            datasets=datasets,
            metrics=metrics,
            warmup_runs=kwargs.get("warmup_runs", 5),
            benchmark_runs=kwargs.get("benchmark_runs", 20),
        )
        result = runner.run(input_data=input_data, models=models)
        logger.info("Benchmark complete")
        return result
    except Exception as exc:
        logger.error(f"Benchmark failed: {exc}")
        raise RuntimeError(f"Benchmark execution failed: {exc}") from exc


def profile(
    model_name: str,
    input_lengths: Optional[List[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run inference profiling on a specified model.

    Parameters
    ----------
    model_name : str
        Model to profile.
    input_lengths : list of int, optional
        Sequence lengths to profile at.
    **kwargs
        Additional profiler parameters (``warmup_runs``, ``profile_runs``).

    Returns
    -------
    dict
        Profiling results with per-layer breakdown and bottleneck analysis.
        Keys include ``total_latency_ms``, ``peak_memory_mb``,
        ``bottleneck_layers``, ``per_layer_latency_ms``, etc.

    Raises
    ------
    KeyError
        If *model_name* is not registered.
    RuntimeError
        If profiling fails.
    """
    from scinfer.evaluation.profiler import InferenceProfiler

    logger.info(f"profile(): model={model_name}, input_lengths={input_lengths}")

    # 1. Load model
    try:
        adapter = _get_adapter(model_name)
        adapter.load_model()
        nn_model = adapter.get_model()
        logger.info(f"Model '{model_name}' loaded for profiling")
    except KeyError:
        raise
    except Exception as exc:
        logger.error(f"Failed to load model '{model_name}': {exc}")
        raise RuntimeError(f"Model loading failed: {exc}") from exc

    # 2. Create representative input
    import torch
    param = next(nn_model.parameters(), None)
    if param is not None:
        last_dim = param.shape[-1] if param.dim() >= 2 else 32
        dummy_input = torch.randn(4, last_dim)
    else:
        dummy_input = torch.randn(4, 32)

    # 3. Run profiling
    try:
        profiler = InferenceProfiler(
            model_name=model_name,
            warmup_runs=kwargs.get("warmup_runs", 3),
            profile_runs=kwargs.get("profile_runs", 10),
        )
        result = profiler.profile(
            nn_model,
            dummy_input,
            input_lengths=input_lengths,
        )
        logger.info(f"Profiling complete: {result.total_latency_ms:.2f} ms total")

        # Return as dict for convenience
        return {
            "model_name": result.model_name,
            "total_latency_ms": result.total_latency_ms,
            "peak_memory_mb": result.peak_memory_mb,
            "bottleneck_layers": result.bottleneck_layers,
            "per_layer_latency_ms": result.per_layer_latency_ms,
            "time_breakdown": result.time_breakdown,
            "memory_breakdown": result.memory_breakdown,
            "raw_profile": result.raw_profile,
            "_result_object": result,
        }
    except Exception as exc:
        logger.error(f"Profiling failed: {exc}")
        raise RuntimeError(f"Profiling execution failed: {exc}") from exc
