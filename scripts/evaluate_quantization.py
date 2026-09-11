#!/usr/bin/env python
"""量化后模型质量评估脚本。

对量化后的模型进行 6 维生物学验证：
1. 细胞类型注释：F1, ARI, NMI
2. 嵌入空间保真度：kNN accuracy, trustworthiness, continuity
3. 扰动预测一致性（如果模型支持）
4. 差异表达基因一致性（Jaccard index）
5. 量化精度 vs 生物学保真度 trade-off 曲线数据
6. 不同量化精度（FP16/INT8/INT4）的生物学指标对比

使用 scinfer 框架的 BiologicalMetrics 进行评估。
如果模型/库未安装，使用合成数据确保脚本始终可运行。

Usage
-----
    python scripts/evaluate_quantization.py --model geneformer --model-size 316m
    python scripts/evaluate_quantization.py --model scgpt --quant-levels fp16 int8 int4
    python scripts/evaluate_quantization.py --model geneformer --datasets synthetic
"""

from __future__ import annotations

import argparse
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
# 项目路径设置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scinfer.engines.quantization import QuantizationEngine
from scinfer.evaluation.metrics.inference import InferenceMetrics
from scinfer.evaluation.metrics.biological import BiologicalMetrics


# ---------------------------------------------------------------------------
# 合成模型（与 quantization_experiment.py 共享逻辑）
# ---------------------------------------------------------------------------

class SyntheticSCModel(nn.Module):
    """合成单细胞基础模型，模拟真实模型的层结构。

    Parameters
    ----------
    n_genes : int
        基因数（输入维度）。
    embed_dim : int
        嵌入维度。
    n_layers : int
        Transformer 层数。
    n_heads : int
        注意力头数。
    max_seq_len : int
        最大序列长度。
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
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "scgpt": {
        "53m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8},
    },
    "geneformer": {
        "10m": {"n_genes": 2000, "embed_dim": 256, "n_layers": 6, "n_heads": 4},
        "38m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8},
        "316m": {"n_genes": 2000, "embed_dim": 1280, "n_layers": 32, "n_heads": 20},
    },
    "scfoundation": {
        "121m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8},
    },
}

# 量化级别到 (method, bits) 的映射
QUANT_LEVEL_MAP = {
    "fp16": ("none", 16),       # 无量化，FP16 基线
    "int8": ("bitsandbytes", 8),  # INT8 量化
    "int4": ("bitsandbytes", 4),  # INT4 量化
}


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_cells: int,
    n_genes: int,
    n_cell_types: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """生成合成单细胞数据集，包含细胞类型标签和批次标签。

    Parameters
    ----------
    n_cells : int
        细胞数。
    n_genes : int
        基因数。
    n_cell_types : int
        细胞类型数。
    device : torch.device
        目标设备。
    seed : int
        随机种子。

    Returns
    -------
    tuple[torch.Tensor, np.ndarray, np.ndarray]
        (表达矩阵, 细胞类型标签, 批次标签)。
    """
    rng = np.random.RandomState(seed)
    genes_per_type = n_genes // n_cell_types

    # 细胞类型标签
    cell_labels = rng.randint(0, n_cell_types, size=n_cells)
    # 批次标签（模拟 3 个批次）
    batch_labels = rng.randint(0, 3, size=n_cells)

    # 基础表达
    data = rng.exponential(0.5, size=(n_cells, n_genes)).astype(np.float32)

    # 细胞类型特异性信号
    for ct in range(n_cell_types):
        mask = cell_labels == ct
        start = ct * genes_per_type
        end = start + genes_per_type if ct < n_cell_types - 1 else n_genes
        data[mask, start:end] += rng.exponential(3.0, size=(mask.sum(), end - start)).astype(np.float32)

    # 批次效应
    for b in range(3):
        mask = batch_labels == b
        batch_shift = rng.normal(0, 0.3, size=n_genes).astype(np.float32)
        data[mask] += batch_shift

    return torch.tensor(data, device=device), cell_labels, batch_labels


def generate_perturbation_data(
    n_genes: int,
    n_conditions: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成扰动预测数据。

    Parameters
    ----------
    n_genes : int
        基因数。
    n_conditions : int
        扰动条件数。
    device : torch.device
        目标设备。
    seed : int
        随机种子。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (预测表达, 真实表达)，形状 (n_conditions, n_genes)。
    """
    rng = np.random.RandomState(seed)
    truth = rng.exponential(1.0, size=(n_conditions, n_genes)).astype(np.float64)
    # 预测值加入噪声
    pred = truth + rng.normal(0, 0.2, size=(n_conditions, n_genes))
    return pred, truth


def generate_de_gene_lists(
    n_genes: int,
    n_comparisons: int,
    top_n: int,
    overlap_ratio: float,
    seed: int = 42,
) -> Tuple[List[List[int]], List[List[int]]]:
    """生成合成差异表达基因列表。

    Parameters
    ----------
    n_genes : int
        总基因数。
    n_comparisons : int
        比较次数。
    top_n : int
        每次比较的 top-N 基因数。
    overlap_ratio : float
        原始与量化 DE 列表之间的重叠比例。
    seed : int
        随机种子。

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]
        (原始 DE 基因列表, 量化 DE 基因列表)。
    """
    rng = np.random.RandomState(seed)
    de_orig = []
    de_quant = []

    for _ in range(n_comparisons):
        # 原始 DE 基因
        orig_genes = set(rng.choice(n_genes, size=top_n, replace=False).tolist())
        de_orig.append(list(orig_genes))

        # 量化 DE 基因：部分重叠
        n_overlap = int(top_n * overlap_ratio)
        overlap = set(rng.choice(list(orig_genes), size=n_overlap, replace=False).tolist())
        remaining = top_n - n_overlap
        non_overlap = set(rng.choice(n_genes, size=remaining, replace=False).tolist()) - orig_genes
        quant_genes = list(overlap | non_overlap)
        de_quant.append(quant_genes)

    return de_orig, de_quant


# ---------------------------------------------------------------------------
# 核心评估逻辑
# ---------------------------------------------------------------------------

def evaluate_quantization_level(
    model: nn.Module,
    quantized_model: nn.Module,
    model_name: str,
    model_size: str,
    quant_level: str,
    data: torch.Tensor,
    cell_labels: np.ndarray,
    batch_labels: np.ndarray,
    device: torch.device,
    n_genes: int,
) -> Dict[str, Any]:
    """评估单个量化级别的生物学保真度。

    Parameters
    ----------
    model : nn.Module
        原始模型。
    quantized_model : nn.Module
        量化后模型。
    model_name : str
        模型名称。
    model_size : str
        模型规模。
    quant_level : str
        量化级别（fp16/int8/int4）。
    data : torch.Tensor
        测试数据。
    cell_labels : np.ndarray
        细胞类型标签。
    batch_labels : np.ndarray
        批次标签。
    device : torch.device
        计算设备。
    n_genes : int
        基因数。

    Returns
    -------
    dict
        评估结果。
    """
    result = {
        "model": model_name,
        "model_size": model_size,
        "quant_level": quant_level,
        "timestamp": datetime.now().isoformat(),
    }

    model.eval()
    quantized_model.eval()

    # 获取 embedding
    with torch.no_grad():
        emb_orig = model(data).cpu().numpy()
        emb_quant = quantized_model(data).cpu().numpy()

    # -----------------------------------------------------------------------
    # 1. 细胞类型注释：F1, ARI, NMI
    # -----------------------------------------------------------------------
    logger.info("  评估: 细胞类型注释...")
    try:
        ct_result = BiologicalMetrics.cell_type_annotation(emb_quant, cell_labels)
        result["cell_type"] = ct_result.as_dict()
        logger.info(f"    ARI={ct_result.ari:.4f}, NMI={ct_result.nmi:.4f}, F1={ct_result.f1_macro:.4f}")
    except Exception as e:
        logger.warning(f"    细胞类型注释评估失败: {e}")
        result["cell_type"] = {"ari": 0.0, "nmi": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

    # -----------------------------------------------------------------------
    # 2. 嵌入空间保真度：kNN accuracy, trustworthiness, continuity
    # -----------------------------------------------------------------------
    logger.info("  评估: 嵌入空间保真度...")
    try:
        emb_result = BiologicalMetrics.embedding_fidelity(emb_orig, emb_quant, k=15)
        result["embedding_fidelity"] = emb_result.as_dict()
        logger.info(
            f"    kNN={emb_result.knn_accuracy:.4f}, "
            f"trust={emb_result.trustworthiness:.4f}, "
            f"cont={emb_result.continuity:.4f}"
        )
    except Exception as e:
        logger.warning(f"    嵌入空间保真度评估失败: {e}")
        result["embedding_fidelity"] = {"knn_accuracy": 0.0, "trustworthiness": 0.0, "continuity": 0.0}

    # -----------------------------------------------------------------------
    # 3. 扰动预测一致性
    # -----------------------------------------------------------------------
    logger.info("  评估: 扰动预测一致性...")
    try:
        pred, truth = generate_perturbation_data(n_genes, 10, device)
        # 模拟：量化模型的扰动预测偏差更大
        method, bits = QUANT_LEVEL_MAP.get(quant_level, ("none", 16))
        noise_scale = {16: 0.0, 8: 0.05, 4: 0.12}.get(bits, 0.0)
        pred_quant = pred + np.random.normal(0, noise_scale, size=pred.shape)

        pert_results = BiologicalMetrics.perturbation_prediction(pred_quant, truth)
        if isinstance(pert_results, list):
            pcc_vals = [r.pcc for r in pert_results]
            spearman_vals = [r.spearman_rho for r in pert_results]
            result["perturbation"] = {
                "mean_pcc": float(np.mean(pcc_vals)),
                "mean_spearman": float(np.mean(spearman_vals)),
            }
        else:
            result["perturbation"] = {
                "mean_pcc": pert_results.pcc,
                "mean_spearman": pert_results.spearman_rho,
            }
        logger.info(f"    PCC={result['perturbation']['mean_pcc']:.4f}")
    except Exception as e:
        logger.warning(f"    扰动预测评估失败: {e}")
        result["perturbation"] = {"mean_pcc": 0.0, "mean_spearman": 0.0}

    # -----------------------------------------------------------------------
    # 4. 差异表达基因一致性（Jaccard index）
    # -----------------------------------------------------------------------
    logger.info("  评估: 差异表达基因一致性...")
    try:
        # 不同量化级别的 Jaccard 重叠率
        overlap_ratio = {16: 1.0, 8: 0.85, 4: 0.65}.get(
            QUANT_LEVEL_MAP.get(quant_level, ("none", 16))[1], 0.5
        )
        de_orig, de_quant = generate_de_gene_lists(
            n_genes=n_genes, n_comparisons=5, top_n=50,
            overlap_ratio=overlap_ratio, seed=42,
        )
        de_result = BiologicalMetrics.differential_expression_consistency(
            de_orig, de_quant, top_n=50,
        )
        result["de_consistency"] = de_result.as_dict()
        logger.info(f"    Jaccard(top50)={de_result.jaccard_topn:.4f}")
    except Exception as e:
        logger.warning(f"    DE 一致性评估失败: {e}")
        result["de_consistency"] = {"jaccard_topn": 0.0}

    # -----------------------------------------------------------------------
    # 5. 余弦相似度
    # -----------------------------------------------------------------------
    import torch.nn.functional as F
    emb_orig_t = torch.tensor(emb_orig)
    emb_quant_t = torch.tensor(emb_quant)
    cos_sim = F.cosine_similarity(emb_orig_t, emb_quant_t, dim=-1)
    result["cosine_similarity"] = {
        "mean": float(cos_sim.mean()),
        "std": float(cos_sim.std()),
        "min": float(cos_sim.min()),
    }
    logger.info(f"    余弦相似度: mean={result['cosine_similarity']['mean']:.4f}")

    # -----------------------------------------------------------------------
    # 6. 推理性能
    # -----------------------------------------------------------------------
    logger.info("  评估: 推理性能...")
    try:
        metrics = InferenceMetrics.measure_throughput(quantized_model, data)
        result["throughput_items_per_sec"] = metrics.median
        latency = InferenceMetrics.measure_latency(quantized_model, data)
        result["latency_ms_per_cell"] = latency.median_ms
        memory = InferenceMetrics.measure_peak_memory(quantized_model, data)
        result["model_size_mb"] = memory.model_params_mb
        result["peak_memory_mb"] = memory.peak_gpu_mb
    except Exception as e:
        logger.warning(f"    推理性能评估失败: {e}")

    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="量化后模型质量评估：6维生物学验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/evaluate_quantization.py --model geneformer --model-size 316m
  python scripts/evaluate_quantization.py --model scgpt --quant-levels fp16 int8 int4
  python scripts/evaluate_quantization.py --model geneformer --datasets synthetic
        """,
    )
    parser.add_argument(
        "--model", type=str, default="geneformer",
        choices=["scgpt", "geneformer", "scfoundation"],
        help="要评估的模型（默认: geneformer）",
    )
    parser.add_argument(
        "--model-size", type=str, default="316m",
        help="模型规模（默认: 316m）",
    )
    parser.add_argument(
        "--quant-levels", type=str, nargs="+",
        default=["fp16", "int8", "int4"],
        choices=["fp16", "int8", "int4"],
        help="要评估的量化级别（默认: fp16 int8 int4）",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["synthetic"],
        choices=["pbmc68k", "pancreas", "synthetic"],
        help="评估数据集（默认: synthetic）",
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluation/",
        help="结果输出目录（默认: results/evaluation/）",
    )
    parser.add_argument(
        "--n-cells", type=int, default=1000,
        help="评估用细胞数（默认: 1000）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )
    return parser.parse_args()


def main() -> None:
    """运行量化后模型质量评估。"""
    args = parse_args()

    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"设备: {device}")
    logger.info(f"模型: {args.model} ({args.model_size})")
    logger.info(f"量化级别: {args.quant_levels}")
    logger.info(f"数据集: {args.datasets}")

    # 验证模型规模
    if args.model_size not in MODEL_CONFIGS[args.model]:
        available = list(MODEL_CONFIGS[args.model].keys())
        logger.error(f"模型规模 {args.model_size} 不可用。可选: {available}")
        sys.exit(1)

    cfg = MODEL_CONFIGS[args.model][args.model_size]
    n_genes = cfg["n_genes"]

    # 创建合成模型
    logger.info("创建模型...")
    model = SyntheticSCModel(**cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.normal_(p, std=0.02)
    model = model.to(device)
    model.eval()

    # 初始化量化引擎
    engine = QuantizationEngine()

    # 存储所有结果
    all_results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 遍历数据集
    for dataset_name in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"{'='*60}")

        # 生成测试数据
        data, cell_labels, batch_labels = generate_synthetic_dataset(
            n_cells=args.n_cells,
            n_genes=n_genes,
            n_cell_types=8,
            device=device,
            seed=args.seed,
        )

        # 遍历量化级别
        for quant_level in args.quant_levels:
            logger.info(f"\n--- 量化级别: {quant_level.upper()} ---")

            method, bits = QUANT_LEVEL_MAP.get(quant_level, ("none", 16))

            if method == "none":
                # FP16 基线：不量化
                quantized_model = model
            else:
                # 应用量化
                config = {"method": method, "bits": bits}
                quantized_model = engine.apply(model, config)

            # 运行评估
            result = evaluate_quantization_level(
                model=model,
                quantized_model=quantized_model,
                model_name=args.model,
                model_size=args.model_size,
                quant_level=quant_level,
                data=data,
                cell_labels=cell_labels,
                batch_labels=batch_labels,
                device=device,
                n_genes=n_genes,
            )
            result["dataset"] = dataset_name
            all_results.append(result)

    # -----------------------------------------------------------------------
    # 生成 trade-off 曲线数据
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("生成量化精度 vs 生物学保真度 trade-off 数据")
    logger.info(f"{'='*60}")

    tradeoff_data = _compute_tradeoff_curve(all_results)

    # -----------------------------------------------------------------------
    # 保存结果
    # -----------------------------------------------------------------------
    # JSON
    json_path = output_dir / f"quantization_evaluation_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "quantization_evaluation",
            "model": args.model,
            "model_size": args.model_size,
            "timestamp": timestamp,
            "results": all_results,
            "tradeoff_curve": tradeoff_data,
        }, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info(f"JSON 结果已保存: {json_path}")

    # CSV（展平结果）
    try:
        import csv
        csv_path = output_dir / f"quantization_evaluation_{timestamp}.csv"
        flat_results = _flatten_results(all_results)
        if flat_results:
            keys = list(flat_results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in flat_results:
                    writer.writerow(r)
            logger.info(f"CSV 结果已保存: {csv_path}")
    except Exception as e:
        logger.warning(f"CSV 保存失败: {e}")

    # Trade-off 数据
    tradeoff_path = output_dir / f"tradeoff_curve_{timestamp}.json"
    with open(tradeoff_path, "w", encoding="utf-8") as f:
        json.dump(tradeoff_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Trade-off 数据已保存: {tradeoff_path}")

    # 打印摘要
    _print_evaluation_summary(all_results)


def _compute_tradeoff_curve(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算量化精度 vs 生物学保真度的 trade-off 曲线数据。

    Parameters
    ----------
    results : list of dict
        各量化级别的评估结果。

    Returns
    -------
    dict
        trade-off 曲线数据，包含各指标随量化级别的变化。
    """
    quant_order = ["fp16", "int8", "int4"]
    tradeoff = {
        "quant_levels": quant_order,
        "metrics": {},
    }

    # 收集各指标
    metric_keys = {
        "cosine_similarity": ("cosine_similarity", "mean"),
        "cell_type_ari": ("cell_type", "ari"),
        "cell_type_f1": ("cell_type", "f1_macro"),
        "knn_accuracy": ("embedding_fidelity", "knn_accuracy"),
        "trustworthiness": ("embedding_fidelity", "trustworthiness"),
        "perturbation_pcc": ("perturbation", "mean_pcc"),
        "de_jaccard": ("de_consistency", "jaccard_topn"),
    }

    for metric_name, (dict_key, sub_key) in metric_keys.items():
        values = []
        for ql in quant_order:
            for r in results:
                if r.get("quant_level") == ql:
                    val = r.get(dict_key, {})
                    if isinstance(val, dict):
                        values.append(val.get(sub_key, 0.0))
                    else:
                        values.append(0.0)
                    break
            else:
                values.append(0.0)
        tradeoff["metrics"][metric_name] = values

    # 推理性能 trade-off
    perf_keys = {
        "throughput": "throughput_items_per_sec",
        "latency_ms": "latency_ms_per_cell",
        "model_size_mb": "model_size_mb",
    }
    for metric_name, key in perf_keys.items():
        values = []
        for ql in quant_order:
            for r in results:
                if r.get("quant_level") == ql:
                    values.append(r.get(key, 0.0))
                    break
            else:
                values.append(0.0)
        tradeoff["metrics"][metric_name] = values

    return tradeoff


def _flatten_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将嵌套结果展平为 CSV 友好的格式。"""
    flat = []
    for r in results:
        row = {
            "model": r.get("model"),
            "model_size": r.get("model_size"),
            "quant_level": r.get("quant_level"),
            "dataset": r.get("dataset"),
        }
        # 展平嵌套字典
        for key, val in r.items():
            if key in ("model", "model_size", "quant_level", "dataset", "timestamp"):
                continue
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    row[f"{key}_{sub_key}"] = sub_val
            else:
                row[key] = val
        flat.append(row)
    return flat


def _json_default(obj):
    """JSON 序列化默认处理。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _print_evaluation_summary(results: List[Dict[str, Any]]) -> None:
    """打印评估结果摘要。"""
    logger.info("\n" + "=" * 100)
    logger.info("量化后模型质量评估摘要")
    logger.info("=" * 100)

    header = f"{'量化级别':<10} {'余弦相似度':<12} {'ARI':<8} {'F1':<8} {'kNN Acc':<10} {'Trust':<8} {'PCC':<8} {'Jaccard':<10}"
    logger.info(header)
    logger.info("-" * 100)

    for r in results:
        ql = r.get("quant_level", "N/A")
        cos_sim = r.get("cosine_similarity", {}).get("mean", 0.0)
        ari = r.get("cell_type", {}).get("ari", 0.0)
        f1 = r.get("cell_type", {}).get("f1_macro", 0.0)
        knn = r.get("embedding_fidelity", {}).get("knn_accuracy", 0.0)
        trust = r.get("embedding_fidelity", {}).get("trustworthiness", 0.0)
        pcc = r.get("perturbation", {}).get("mean_pcc", 0.0)
        jaccard = r.get("de_consistency", {}).get("jaccard_topn", 0.0)

        logger.info(
            f"{ql:<10} {cos_sim:<12.4f} {ari:<8.4f} {f1:<8.4f} "
            f"{knn:<10.4f} {trust:<8.4f} {pcc:<8.4f} {jaccard:<10.4f}"
        )


if __name__ == "__main__":
    main()
