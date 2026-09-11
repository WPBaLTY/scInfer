#!/usr/bin/env python
"""Knowledge distillation experiment script.

Performs systematic distillation experiments on single-cell foundation models
(Geneformer, scGPT) to evaluate how different distillation strategies affect:
- Inference speed (ms/cell, items/s)
- Model size (MB, parameter count)
- Embedding quality (cosine similarity to teacher)
- Biological fidelity (cell-type clustering quality)

Three distillation strategies are compared:
- ``representation``: match intermediate feature maps
- ``task``: match softened output logits (KL-divergence)
- ``contrastive``: align teacher-student embedding spaces via triplet loss

If real model weights or training data are unavailable, the script
automatically falls back to synthetic models and data.

Usage
-----
    python scripts/distillation_experiment.py \\
        --teacher geneformer --teacher-size 316m \\
        --student-sizes 38m 10m \\
        --strategy representation task contrastive \\
        --output results/distillation/

    python scripts/distillation_experiment.py \\
        --teacher scgpt --teacher-size 53m \\
        --student-sizes 38m 10m \\
        --strategy representation \\
        --n-epochs 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scinfer.engines.distillation import (
    DistillationConfig,
    DistillationEngine,
    _generate_synthetic_data,
)


# ---------------------------------------------------------------------------
# Synthetic model (shared with quantization_experiment.py for consistency)
# ---------------------------------------------------------------------------

class SyntheticSCModel(nn.Module):
    """Synthetic single-cell foundation model for fallback experiments.

    Architecture: Embedding → TransformerEncoder × N → LayerNorm → Linear Head

    Parameters
    ----------
    n_genes : int
        Number of input genes.
    embed_dim : int
        Hidden / embedding dimension.
    n_layers : int
        Number of Transformer encoder layers.
    n_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        n_genes: int = 2000,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(n_genes, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = self.input_proj(x).unsqueeze(1)
        else:
            x = self.input_proj(x)

        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # average pooling
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Model architecture configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "geneformer": {
        "10m":  {"n_genes": 2000, "embed_dim": 256,  "n_layers": 6,  "n_heads": 4},
        "38m":  {"n_genes": 2000, "embed_dim": 512,  "n_layers": 12, "n_heads": 8},
        "316m": {"n_genes": 2000, "embed_dim": 1280, "n_layers": 32, "n_heads": 20},
    },
    "scgpt": {
        "53m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8},
    },
}

# Reverse mapping: model_size → rough parameter count (for logging)
SIZE_APPROX_PARAMS = {
    "10m": 10_000_000,
    "38m": 38_000_000,
    "53m": 53_000_000,
    "316m": 316_000_000,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(
    model_name: str, model_size: str, device: torch.device
) -> nn.Module:
    """Instantiate a synthetic model matching the given architecture spec.

    Parameters
    ----------
    model_name : str
        Model family (``"geneformer"`` or ``"scgpt"``).
    model_size : str
        Size tag (e.g. ``"316m"``, ``"53m"``).
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module
        Initialised model on *device*.
    """
    cfg = MODEL_CONFIGS[model_name][model_size]
    model = SyntheticSCModel(**cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.normal_(p, std=0.02)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_embedding_cosine_similarity(
    teacher: nn.Module,
    student: nn.Module,
    data: torch.Tensor,
) -> Dict[str, float]:
    """Compute cosine similarity between teacher and student embeddings.

    Parameters
    ----------
    teacher : nn.Module
        Teacher model.
    student : nn.Module
        Student model.
    data : torch.Tensor
        Input expression data.

    Returns
    -------
    dict[str, float]
        Cosine similarity statistics.
    """
    import torch.nn.functional as F

    teacher.eval()
    student.eval()

    device = (
        next(teacher.parameters()).device
        if list(teacher.parameters())
        else torch.device("cpu")
    )
    if data.device != device:
        data = data.to(device)

    with torch.no_grad():
        t_emb = teacher(data)
        s_emb = student(data)

    # Align last dimension
    if t_emb.shape[-1] != s_emb.shape[-1]:
        min_dim = min(t_emb.shape[-1], s_emb.shape[-1])
        t_emb = t_emb[..., :min_dim]
        s_emb = s_emb[..., :min_dim]

    cos_sim = F.cosine_similarity(t_emb, s_emb, dim=-1)
    return {
        "cosine_similarity_mean": round(float(cos_sim.mean()), 6),
        "cosine_similarity_std": round(float(cos_sim.std()), 6),
        "cosine_similarity_min": round(float(cos_sim.min()), 6),
        "cosine_similarity_max": round(float(cos_sim.max()), 6),
    }


def evaluate_clustering_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate embedding quality via k-means clustering metrics.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings ``(n_cells, dim)``.
    labels : np.ndarray
        Ground-truth cell-type labels.

    Returns
    -------
    dict[str, float]
        Adjusted Rand Index and Normalised Mutual Information.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(embeddings)

        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        return {
            "adjusted_rand_index": round(float(ari), 4),
            "normalized_mutual_info": round(float(nmi), 4),
        }
    except ImportError:
        logger.debug("sklearn not available – skipping clustering evaluation")
        return {"adjusted_rand_index": None, "normalized_mutual_info": None}


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def run_single_distillation(
    teacher: nn.Module,
    teacher_name: str,
    teacher_size: str,
    student_size: str,
    strategy: str,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    engine: DistillationEngine,
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    n_genes: int,
) -> Dict[str, Any]:
    """Execute one distillation experiment and return results.

    Parameters
    ----------
    teacher : nn.Module
        Teacher model (frozen during training).
    teacher_name : str
        Teacher model family name.
    teacher_size : str
        Teacher size tag.
    student_size : str
        Student size tag (determines architecture).
    strategy : str
        Distillation strategy.
    train_data : torch.Tensor
        Training expression data.
    test_data : torch.Tensor
        Test expression data.
    test_labels : np.ndarray
        Ground-truth cell-type labels for test data.
    engine : DistillationEngine
        Distillation engine instance.
    device : torch.device
        Compute device.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    temperature : float
        Softmax temperature for task strategy.
    n_genes : int
        Number of genes.

    Returns
    -------
    dict
        Experiment result dictionary.
    """
    result: Dict[str, Any] = {
        "teacher": teacher_name,
        "teacher_size": teacher_size,
        "student_size": student_size,
        "strategy": strategy,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"  Distillation: strategy={strategy}, student={student_size}")

    # Create student model
    student_cfg = MODEL_CONFIGS[teacher_name].get(
        student_size, MODEL_CONFIGS[teacher_name][teacher_size]
    )
    student = SyntheticSCModel(**student_cfg).to(device)
    for p in student.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.normal_(p, std=0.02)

    # Build config
    config = DistillationConfig(
        strategy=strategy,
        temperature=temperature,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_genes=n_genes,
    )

    # Train
    t0 = time.perf_counter()
    student = engine.train_distillation(
        teacher=teacher,
        student=student,
        data=train_data,
        config=config,
    )
    train_time = time.perf_counter() - t0
    result["training_time_s"] = round(train_time, 3)

    # Benchmark
    bench = engine.benchmark(student, test_data)
    result.update(bench)

    # Embedding similarity
    similarity = compute_embedding_cosine_similarity(teacher, student, test_data)
    result.update(similarity)

    # Clustering quality
    student.eval()
    with torch.no_grad():
        s_emb = student(test_data).cpu().numpy()
    clustering = evaluate_clustering_quality(s_emb, test_labels)
    result.update(clustering)

    # Parameter counts
    result["teacher_params"] = count_parameters(teacher)
    result["student_params"] = count_parameters(student)
    result["compression_ratio"] = round(
        result["teacher_params"] / max(result["student_params"], 1), 2
    )

    logger.info(
        f"  Result: cos_sim={result['cosine_similarity_mean']:.4f}, "
        f"latency={result['latency_ms']:.3f} ms, "
        f"compression={result['compression_ratio']:.1f}x"
    )
    return result


def run_baseline_comparison(
    teacher: nn.Module,
    teacher_name: str,
    teacher_size: str,
    student_size: str,
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    engine: DistillationEngine,
    device: torch.device,
    n_genes: int,
) -> List[Dict[str, Any]]:
    """Compare distillation vs. direct training for the student model.

    Parameters
    ----------
    teacher : nn.Module
        Teacher model.
    teacher_name : str
        Teacher model name.
    teacher_size : str
        Teacher size tag.
    student_size : str
        Student size tag.
    test_data : torch.Tensor
        Test data.
    test_labels : np.ndarray
        Test labels.
    engine : DistillationEngine
        Distillation engine.
    device : torch.device
        Compute device.
    n_genes : int
        Number of genes.

    Returns
    -------
    list[dict]
        Results for both distillation and direct-training baselines.
    """
    results: List[Dict[str, Any]] = []

    student_cfg = MODEL_CONFIGS[teacher_name].get(
        student_size, MODEL_CONFIGS[teacher_name][teacher_size]
    )

    # --- Baseline 1: student trained from scratch (no teacher) ---
    student_direct = SyntheticSCModel(**student_cfg).to(device)
    for p in student_direct.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.normal_(p, std=0.02)

    # Simple self-supervised training: reconstruct input via MSE
    optimizer = torch.optim.Adam(student_direct.parameters(), lr=1e-4)
    student_direct.train()
    n_epochs_direct = 10
    for epoch in range(n_epochs_direct):
        with torch.no_grad():
            batch_idx = torch.randperm(test_data.shape[0], device=device)[:64]
        batch = test_data[batch_idx]
        out = student_direct(batch)
        # Reconstruction target: project input to embed_dim space
        target = batch[:, :out.shape[-1]] if batch.shape[-1] != out.shape[-1] else batch
        loss = nn.functional.mse_loss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_direct.eval()
    bench_direct = engine.benchmark(student_direct, test_data)
    with torch.no_grad():
        emb_direct = student_direct(test_data).cpu().numpy()
    clustering_direct = evaluate_clustering_quality(emb_direct, test_labels)

    direct_result: Dict[str, Any] = {
        "teacher": teacher_name,
        "teacher_size": teacher_size,
        "student_size": student_size,
        "strategy": "direct_training",
        "n_epochs": n_epochs_direct,
        "timestamp": datetime.now().isoformat(),
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student_direct),
        **bench_direct,
        **clustering_direct,
    }
    # Cosine similarity to teacher
    sim_direct = compute_embedding_cosine_similarity(teacher, student_direct, test_data)
    direct_result.update(sim_direct)
    direct_result["compression_ratio"] = round(
        direct_result["teacher_params"]
        / max(direct_result["student_params"], 1),
        2,
    )
    results.append(direct_result)

    logger.info(
        f"  Direct training baseline: cos_sim={direct_result['cosine_similarity_mean']:.4f}, "
        f"latency={direct_result['latency_ms']:.3f} ms"
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation experiments for single-cell foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/distillation_experiment.py \\
      --teacher geneformer --teacher-size 316m \\
      --student-sizes 38m 10m \\
      --strategy representation task contrastive

  python scripts/distillation_experiment.py \\
      --teacher scgpt --teacher-size 53m \\
      --student-sizes 38m 10m \\
      --strategy representation --n-epochs 20
        """,
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="geneformer",
        choices=["geneformer", "scgpt"],
        help="Teacher model family (default: geneformer)",
    )
    parser.add_argument(
        "--teacher-size",
        type=str,
        default="316m",
        help="Teacher model size tag (default: 316m)",
    )
    parser.add_argument(
        "--student-sizes",
        type=str,
        nargs="+",
        default=["38m", "10m"],
        help="Student model size tags (default: 38m 10m)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        nargs="+",
        default=["representation", "task", "contrastive"],
        choices=["representation", "task", "contrastive"],
        help="Distillation strategies to compare (default: all three)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/distillation/",
        help="Output directory (default: results/distillation/)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="Number of distillation training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--temperature", type=float, default=4.0,
        help="Softmax temperature for task strategy (default: 4.0)",
    )
    parser.add_argument(
        "--n-train-samples", type=int, default=512,
        help="Number of training samples (default: 512)",
    )
    parser.add_argument(
        "--n-test-samples", type=int, default=256,
        help="Number of test samples (default: 256)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run systematic distillation experiments."""
    args = parse_args()

    # Logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Teacher: {args.teacher} ({args.teacher_size})")
    logger.info(f"Student sizes: {args.student_sizes}")
    logger.info(f"Strategies: {args.strategy}")
    logger.info(f"Output: {output_dir}")

    # Validate teacher size
    if args.teacher_size not in MODEL_CONFIGS[args.teacher]:
        available = list(MODEL_CONFIGS[args.teacher].keys())
        logger.error(
            f"Teacher size '{args.teacher_size}' not available for {args.teacher}. "
            f"Available: {available}"
        )
        sys.exit(1)

    engine = DistillationEngine()
    all_results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_genes = MODEL_CONFIGS[args.teacher][args.teacher_size]["n_genes"]

    # Create teacher
    teacher = create_model(args.teacher, args.teacher_size, device)
    teacher.eval()
    logger.info(
        f"Teacher parameters: {count_parameters(teacher):,}"
    )

    # Generate synthetic data (always used as fallback)
    logger.info("Generating synthetic training and test data...")
    train_data, _ = _generate_synthetic_data(
        n_samples=args.n_train_samples,
        n_genes=n_genes,
        n_cell_types=5,
        device=device,
        seed=args.seed,
    )
    test_data, test_labels = _generate_synthetic_data(
        n_samples=args.n_test_samples,
        n_genes=n_genes,
        n_cell_types=5,
        device=device,
        seed=args.seed + 1000,
    )
    test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels

    # --- Teacher baseline ---
    logger.info("\n" + "=" * 60)
    logger.info("Teacher baseline")
    logger.info("=" * 60)
    teacher_bench = engine.benchmark(teacher, test_data)
    with torch.no_grad():
        teacher_emb = teacher(test_data).cpu().numpy()
    teacher_clustering = evaluate_clustering_quality(teacher_emb, test_labels_np)
    teacher_result: Dict[str, Any] = {
        "teacher": args.teacher,
        "teacher_size": args.teacher_size,
        "student_size": args.teacher_size,
        "strategy": "teacher_baseline",
        "n_epochs": 0,
        "timestamp": datetime.now().isoformat(),
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(teacher),
        "compression_ratio": 1.0,
        "cosine_similarity_mean": 1.0,
        "cosine_similarity_std": 0.0,
        **teacher_bench,
        **teacher_clustering,
    }
    all_results.append(teacher_result)
    logger.info(
        f"  Teacher: latency={teacher_result['latency_ms']:.3f} ms, "
        f"ARI={teacher_result.get('adjusted_rand_index', 'N/A')}"
    )

    # --- Distillation experiments ---
    for student_size in args.student_sizes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Student: {student_size}")
        logger.info(f"{'=' * 60}")

        # Check if student size exists for this teacher model
        if student_size not in MODEL_CONFIGS[args.teacher]:
            logger.warning(
                f"Student size '{student_size}' not available for {args.teacher}, "
                f"using teacher architecture as fallback"
            )

        # Distillation strategies
        for strategy in args.strategy:
            logger.info(f"\n--- Strategy: {strategy} ---")
            result = run_single_distillation(
                teacher=teacher,
                teacher_name=args.teacher,
                teacher_size=args.teacher_size,
                student_size=student_size,
                strategy=strategy,
                train_data=train_data,
                test_data=test_data,
                test_labels=test_labels_np,
                engine=engine,
                device=device,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                temperature=args.temperature,
                n_genes=n_genes,
            )
            all_results.append(result)

        # Direct training baseline (for first strategy only, to avoid repetition)
        logger.info(f"\n--- Baseline: direct training ---")
        baseline_results = run_baseline_comparison(
            teacher=teacher,
            teacher_name=args.teacher,
            teacher_size=args.teacher_size,
            student_size=student_size,
            test_data=test_data,
            test_labels=test_labels_np,
            engine=engine,
            device=device,
            n_genes=n_genes,
        )
        all_results.extend(baseline_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Experiments complete: {len(all_results)} result sets")
    logger.info(f"{'=' * 60}")

    # JSON
    json_path = output_dir / f"distillation_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "knowledge_distillation",
                "timestamp": timestamp,
                "config": {
                    "teacher": args.teacher,
                    "teacher_size": args.teacher_size,
                    "student_sizes": args.student_sizes,
                    "strategies": args.strategy,
                    "n_epochs": args.n_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "temperature": args.temperature,
                    "n_train_samples": args.n_train_samples,
                    "n_test_samples": args.n_test_samples,
                },
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"JSON results saved: {json_path}")

    # CSV
    try:
        csv_path = output_dir / f"distillation_results_{timestamp}.csv"
        if all_results:
            all_keys: List[str] = []
            seen: set = set()
            for r in all_results:
                for k in r.keys():
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for r in all_results:
                    writer.writerow(r)
            logger.info(f"CSV results saved: {csv_path}")
    except Exception as e:
        logger.warning(f"CSV save failed: {e}")

    # Summary
    _print_summary(all_results)


def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary table."""
    logger.info("\n" + "=" * 90)
    logger.info("Distillation Experiment Summary")
    logger.info("=" * 90)

    header = (
        f"  {'Strategy':<22} {'Student':<10} {'Cos Sim':<10} "
        f"{'Latency(ms)':<13} {'Size(MB)':<10} {'ARI':<8} {'Compression':<12}"
    )
    logger.info(header)
    logger.info(f"  {'-' * 85}")

    for r in results:
        strategy = r.get("strategy", "N/A")
        student = r.get("student_size", "N/A")
        cos_sim = r.get("cosine_similarity_mean", "N/A")
        latency = r.get("latency_ms", "N/A")
        size_mb = r.get("model_size_mb", "N/A")
        ari = r.get("adjusted_rand_index", "N/A")
        compression = r.get("compression_ratio", "N/A")

        def fmt(v: Any, spec: str) -> str:
            if v is None or v == "N/A":
                return "N/A"
            if isinstance(v, float):
                return f"{v:{spec}}"
            return str(v)

        logger.info(
            f"  {strategy:<22} {student:<10} {fmt(cos_sim, '.4f'):<10} "
            f"{fmt(latency, '.3f'):<13} {fmt(size_mb, '.1f'):<10} "
            f"{fmt(ari, '.4f'):<8} {fmt(compression, '.1f') + 'x':<12}"
        )


if __name__ == "__main__":
    main()
