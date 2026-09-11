# scInfer

**Systematic Inference Optimization Framework for Single-Cell Foundation Models**

scInfer is a systematic inference optimization framework for single-cell foundation models (scGPT, Geneformer, scFoundation, UCE). It adapts LLM inference optimization techniques — quantization, FlashAttention, knowledge distillation, compilation optimization, and dynamic batch scheduling — to the single-cell domain, achieving 3–5x inference speedup while preserving biological fidelity.

## Key Features

- **Unified Model Interface** — A single API for scGPT / Geneformer / scFoundation / UCE
- **Modular Optimization Engines** — Quantization (INT8/INT4), FlashAttention, torch.compile, knowledge distillation — plug and play
- **Composable Optimization Pipeline** — Multiple optimization techniques safely composable with automatic conflict detection
- **Biological Fidelity Monitoring** — Concurrent tracking of inference speed and biological metrics (ARI, NMI, F1, PCC)
- **Hardware Awareness** — Automatic GPU detection and optimal strategy selection
- **Extensible** — Register new models and optimization strategies via decorators

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        scinfer API                              │
│   encode()  ·  optimize()  ·  benchmark()  ·  profile()        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Optimization Combiner                       │   │
│  │  quantization + attention + compilation + scheduling    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │  Quantization │  │  Attention   │  │  Compilation     │     │
│  │  Engine       │  │  Accelerator │  │  Engine          │     │
│  │  (INT8/INT4)  │  │  (FlashAttn) │  │  (torch.compile) │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Model Adapter Layer                         │   │
│  │  scGPT  ·  Geneformer  ·  scFoundation  ·  UCE         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Evaluation Layer                            │   │
│  │  Profiler  ·  Benchmark  ·  Metrics  ·  Report          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Basic installation
pip install scinfer

# With all model dependencies
pip install scinfer[all-models]

# With optimization engines
pip install scinfer[optimization]

# Full installation (all optional dependencies)
pip install scinfer[all]

# Development installation
git clone [GitHub URL — TODO: update before release]  <!-- TODO: update before release -->
cd scinfer
pip install -e ".[all]"
```

### Optional Dependencies

| Dependency Group | Included |
|------------------|----------|
| `scinfer[scgpt]` | scGPT model dependencies |
| `scinfer[geneformer]` | Geneformer model dependencies |
| `scinfer[scfoundation]` | scFoundation model dependencies |
| `scinfer[uce]` | UCE model dependencies |
| `scinfer[optimization]` | FlashAttention, GPTQ, ONNX, and other optimization engines |
| `scinfer[all]` | All optional dependencies |

## Quick Start

```python
import scanpy as sc
import scinfer

# Load single-cell data
adata = sc.read_h5ad("pbmc_10k.h5ad")

# Generate embeddings in one line (automatically selects optimal optimization strategy)
adata = scinfer.encode(adata, model="scgpt", optimize=True)

# Retrieve optimized embeddings
embeddings = adata.obsm["scinfer_embedding"]
```

### Cross-Model Benchmarking

```python
# Compare across models
results = scinfer.benchmark(
    model_names=["scgpt", "geneformer", "scfoundation"],
    datasets=["pbmc_10k", "pancreas"],
    metrics=["throughput", "latency", "ari", "nmi"],
)

# Inference bottleneck profiling
profile = scinfer.profile("scgpt", input_lengths=[512, 1024, 2048])
print(profile.summary())
```

## Experiment Script Guide

### Inference Profiling

```bash
# Profile all models
python scripts/profile_models.py --models scgpt geneformer scfoundation uce --output results/profiling/
```

### Quantization Experiments

```bash
# Run quantization evaluation experiment
python scripts/quantization_experiment.py --config configs/quantization/experiment_config.yaml

# Evaluate quantization effects
python scripts/evaluate_quantization.py --data-dir results/quantization/ --output results/quant_eval/
```

### FlashAttention Experiments

```bash
python scripts/flash_attention_experiment.py --versions v1 v2 v3 --seq-lengths 128 256 512 1024 2048
```

### Knowledge Distillation

```bash
python scripts/distillation_experiment.py --teacher scgpt --student-sizes 12M 6M 3M --epochs 100
```

### Combined Optimization

```bash
python scripts/combined_optimization_experiment.py --config configs/optimization/combined.yaml
```

### GPU Configuration Comparison

```bash
# Compare inference performance across different GPUs
python scripts/compare_gpu_configs.py \
    --gpus T4 V100 A100 H100 RTX4090 \
    --models scgpt geneformer scfoundation \
    --output results/gpu_comparison/

# Run real benchmarks (requires GPU)
python scripts/compare_gpu_configs.py --real-benchmark --output results/gpu_comparison/
```

### Paper Figure Generation

```bash
# Generate paper Fig.1-7
python scripts/generate_figures.py --data-dir results/ --output-dir figures/ --format pdf --dpi 350

# Generate Extended Data Figures
python scripts/generate_extended_data.py --data-dir results/ --output-dir figures/extended/ --format pdf

# Generate specific figures only
python scripts/generate_figures.py --figs 1 3 5 --format png --dpi 300
```

### Million-Cell Demo

```bash
python scripts/million_cell_demo.py --n-cells 1000000 --model scgpt --batch-size 256
```

## Project Structure

```
scInfer/
├── configs/                    # Configuration files
│   ├── models/                 # Model configs (scgpt, geneformer, ...)
│   ├── optimization/           # Optimization strategy configs
│   ├── quantization/           # Quantization experiment configs
│   └── default.yaml            # Default configuration
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_profiling_exploration.ipynb
│   ├── 02_benchmark_results.ipynb
│   ├── 03_quantization_analysis.ipynb
│   ├── 04_distillation_analysis.ipynb
│   ├── 05_combined_optimization.ipynb
│   ├── 06_biological_validation.ipynb
│   └── 07_paper_figures.ipynb  # Paper figure collection
├── results/                    # Experimental results
├── scinfer/                    # Core package
│   ├── adapters/               # Model adapters
│   │   ├── base.py             # Adapter base class
│   │   ├── scgpt.py            # scGPT adapter
│   │   ├── geneformer.py       # Geneformer adapter
│   │   ├── scfoundation.py     # scFoundation adapter
│   │   └── uce.py              # UCE adapter
│   ├── core/                   # Core modules
│   │   ├── config.py           # Configuration management
│   │   ├── registry.py         # Model/strategy registry
│   │   └── types.py            # Type definitions
│   ├── engines/                # Optimization engines
│   │   ├── quantization.py     # Quantization engine (INT8/INT4)
│   │   ├── attention.py        # FlashAttention acceleration
│   │   ├── compilation.py      # torch.compile / ONNX
│   │   ├── scheduling.py       # Dynamic batch scheduling
│   │   └── combiner.py         # Optimization combiner
│   ├── evaluation/             # Evaluation module
│   │   ├── benchmark.py        # Benchmark framework
│   │   ├── profiler.py         # Inference profiler
│   │   ├── bio_validation.py   # Biological validation
│   │   ├── report.py           # Report generation
│   │   └── metrics/            # Evaluation metrics
│   │       ├── biological.py   # Biological metrics (ARI, NMI, F1)
│   │       └── inference.py    # Inference metrics (throughput, latency)
│   ├── utils/                  # Utility functions
│   └── api.py                  # Public API
├── scripts/                    # Experiment scripts
│   ├── profile_models.py       # Model profiling
│   ├── benchmark_inference.py  # Inference benchmark
│   ├── quantization_experiment.py
│   ├── evaluate_quantization.py
│   ├── flash_attention_experiment.py
│   ├── combined_optimization_experiment.py
│   ├── million_cell_demo.py    # Million-cell demonstration
│   ├── generate_figures.py     # Paper figure generation (Fig.1-7)
│   ├── generate_extended_data.py  # Extended Data Figures
│   └── compare_gpu_configs.py  # GPU configuration comparison
├── tests/                      # Unit tests
├── pyproject.toml              # Project configuration
└── LICENSE                     # Apache 2.0
```

## Comparison with Existing Tools

| Feature | scInfer | Native Model Code | vLLM / TGI |
|---------|---------|-------------------|------------|
| Single-cell model support | ✅ Native | ❌ Per-model adaptation | ❌ LLM only |
| Quantization (INT8/INT4) | ✅ Unified interface | ⚠️ Manual implementation | ✅ |
| FlashAttention | ✅ Automatic selection | ⚠️ Manual integration | ✅ |
| Biological metrics | ✅ ARI/NMI/F1/PCC | ⚠️ Custom code | ❌ |
| Cross-model benchmarking | ✅ Built-in | ❌ | ❌ |
| Composable optimization | ✅ Pipeline | ❌ | ⚠️ Limited |
| Knowledge distillation | ✅ Single-cell adapted | ❌ | ❌ |
| Hardware-aware scheduling | ✅ Automatic GPU detection | ❌ | ⚠️ Basic |
| Million-cell support | ✅ Optimized batching | ❌ OOM | ✅ |

## Roadmap

- [x] **Phase 1** — Inference profiling and bottleneck analysis (4 models)
- [x] **Phase 2** — Quantization engine (INT8/INT4) + biological fidelity tracking
- [x] **Phase 3** — FlashAttention integration (v1/v2/v3)
- [x] **Phase 4** — Knowledge distillation + torch.compile optimization
- [x] **Phase 5** — Combined optimization + biological validation
- [ ] **Phase 6** — Production-grade inference serving (dynamic batch scheduling, REST API)
- [ ] **Phase 7** — Additional single-cell model support (scBERT, scToken, ...)

## Citation

```bibtex
@software{scinfer2025,
  title  = {scInfer: Systematic Inference Optimization for Single-Cell Foundation Models},
  author = {scInfer Team},
  year   = {2025},
  url    = {[GitHub URL — TODO: update before release]},  <!-- TODO: update before release -->
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
