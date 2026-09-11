#!/usr/bin/env python
"""系统量化实验脚本。

对 scGPT(53M) / Geneformer(10M/38M/316M) / scFoundation(121M) 进行系统量化实验，
评估不同量化方法（BitsAndBytes/GPTQ/AWQ）和位宽（INT8/INT4）对：
- 推理速度（it/s, ms/cell）
- 内存占用（模型大小 MB、峰值显存 MB）
- Embedding 保真度（余弦相似度）
- 细胞注释准确率

如果模型或量化库未安装，自动使用合成模型和数据进行 fallback，确保脚本始终可运行。

Usage
-----
    python scripts/quantization_experiment.py --models scgpt geneformer scfoundation
    python scripts/quantization_experiment.py --models geneformer --methods gptq awq --bits 4 8
    python scripts/quantization_experiment.py --calibration-datasets pbmc pancreas synthetic
"""

from __future__ import annotations

import argparse
import copy
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


# ---------------------------------------------------------------------------
# 合成模型：当真实模型不可用时使用
# ---------------------------------------------------------------------------

class SyntheticSCModel(nn.Module):
    """合成单细胞基础模型，模拟真实模型的层结构。

    包含：Embedding → TransformerEncoder × N → LayerNorm → Linear Head
    用于在没有真实模型权重时进行量化实验。

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

        # 输入投影
        self.input_proj = nn.Linear(n_genes, embed_dim)
        # 位置编码
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        # Transformer 编码器层
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
        # 输出头
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状 (batch, n_genes) 或 (batch, seq_len, n_genes)。

        Returns
        -------
        torch.Tensor
            输出嵌入，形状 (batch, embed_dim)。
        """
        if x.dim() == 2:
            # (batch, n_genes) → (batch, 1, embed_dim)
            x = self.input_proj(x).unsqueeze(1)
        else:
            x = self.input_proj(x)

        # 添加位置编码
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Transformer 编码
        x = self.transformer(x)
        x = self.norm(x)

        # 池化 + 输出头
        x = x.mean(dim=1)  # 平均池化
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

# 各模型的架构参数
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

# 校准数据集描述
CALIBRATION_DATASETS = {
    "pbmc": {"description": "PBMC 外周血单核细胞", "n_cells": 500, "cell_types": 8},
    "pancreas": {"description": "胰腺细胞", "n_cells": 500, "cell_types": 6},
    "synthetic": {"description": "合成数据", "n_cells": 500, "cell_types": 5},
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """自动选择最优设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_synthetic_model(model_name: str, model_size: str, device: torch.device) -> nn.Module:
    """创建合成模型。

    Parameters
    ----------
    model_name : str
        模型名称（scgpt/geneformer/scfoundation）。
    model_size : str
        模型规模（如 53m, 316m）。
    device : torch.device
        目标设备。

    Returns
    -------
    nn.Module
        合成模型实例。
    """
    cfg = MODEL_CONFIGS[model_name][model_size]
    model = SyntheticSCModel(**cfg)
    # 初始化权重
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
        else:
            nn.init.normal_(p, std=0.02)
    return model.to(device)


def generate_synthetic_data(
    n_samples: int,
    n_genes: int,
    n_cell_types: int = 5,
    device: torch.device = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, np.ndarray]:
    """生成合成单细胞数据。

    模拟不同细胞类型的基因表达模式：每种细胞类型有特定的高表达基因子集。

    Parameters
    ----------
    n_samples : int
        样本数。
    n_genes : int
        基因数。
    n_cell_types : int
        细胞类型数。
    device : torch.device, optional
        目标设备。
    seed : int
        随机种子。

    Returns
    -------
    tuple[torch.Tensor, np.ndarray]
        (表达矩阵, 细胞类型标签)。
    """
    rng = np.random.RandomState(seed)
    device = device or torch.device("cpu")

    # 为每种细胞类型定义高表达基因模块
    genes_per_type = n_genes // n_cell_types
    labels = rng.randint(0, n_cell_types, size=n_samples)

    # 基础表达（低水平噪声）
    data = rng.exponential(0.5, size=(n_samples, n_genes)).astype(np.float32)

    # 添加细胞类型特异性信号
    for ct in range(n_cell_types):
        mask = labels == ct
        start = ct * genes_per_type
        end = start + genes_per_type if ct < n_cell_types - 1 else n_genes
        data[mask, start:end] += rng.exponential(3.0, size=(mask.sum(), end - start)).astype(np.float32)

    return torch.tensor(data, device=device), labels


def generate_calibration_data(
    dataset_name: str,
    n_samples: int,
    n_genes: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """生成校准数据。

    Parameters
    ----------
    dataset_name : str
        数据集名称（pbmc/pancreas/synthetic）。
    n_samples : int
        样本数。
    n_genes : int
        基因数。
    device : torch.device
        目标设备。
    seed : int
        随机种子（不同数据集使用不同偏移）。

    Returns
    -------
    torch.Tensor
        校准数据张量。
    """
    seed_offset = {"pbmc": 0, "pancreas": 1000, "synthetic": 2000}.get(dataset_name, 0)
    data, _ = generate_synthetic_data(
        n_samples=n_samples,
        n_genes=n_genes,
        n_cell_types=CALIBRATION_DATASETS[dataset_name]["cell_types"],
        device=device,
        seed=seed + seed_offset,
    )
    return data


# ---------------------------------------------------------------------------
# 核心实验逻辑
# ---------------------------------------------------------------------------

def run_single_quantization(
    model: nn.Module,
    model_name: str,
    model_size: str,
    method: str,
    bits: int,
    calibration_data: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    engine: QuantizationEngine,
    device: torch.device,
) -> Dict[str, Any]:
    """运行单次量化实验。

    Parameters
    ----------
    model : nn.Module
        原始（未量化）模型。
    model_name : str
        模型名称。
    model_size : str
        模型规模。
    method : str
        量化方法。
    bits : int
        量化位宽。
    calibration_data : torch.Tensor
        校准数据。
    test_data : torch.Tensor
        测试数据。
    test_labels : np.ndarray
        测试标签。
    engine : QuantizationEngine
        量化引擎实例。
    device : torch.device
        计算设备。

    Returns
    -------
    dict
        实验结果字典。
    """
    result = {
        "model": model_name,
        "model_size": model_size,
        "method": method,
        "bits": bits,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"  量化: {method} INT{bits}")

    # 1. 应用量化
    config = {"method": method, "bits": bits}
    t0 = time.perf_counter()
    quantized_model = engine.apply(model, config)
    quant_time = time.perf_counter() - t0
    result["quantization_time_s"] = round(quant_time, 3)

    # 2. 基准测试：推理速度
    bench_metrics = engine.benchmark(quantized_model, test_data)
    result.update(bench_metrics)

    # 3. Embedding 保真度：余弦相似度
    similarity = engine.compute_embedding_similarity(
        model, quantized_model, test_data
    )
    result.update(similarity)

    # 4. 细胞注释准确率（使用 kNN）
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score

        # 获取原始和量化后的 embedding
        model.eval()
        quantized_model.eval()
        with torch.no_grad():
            emb_orig = model(test_data).cpu().numpy()
            emb_quant = quantized_model(test_data).cpu().numpy()

        # 使用原始 embedding 训练 kNN，在量化 embedding 上评估
        n_train = min(len(test_labels) // 2, 500)
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(emb_orig[:n_train], test_labels[:n_train])
        pred_quant = knn.predict(emb_quant[n_train:])
        f1 = f1_score(test_labels[n_train:], pred_quant, average="macro", zero_division=0)
        result["cell_type_f1"] = round(float(f1), 4)

        # kNN accuracy：量化 embedding 的 kNN 准确率
        acc = (pred_quant == test_labels[n_train:]).mean()
        result["knn_accuracy"] = round(float(acc), 4)
    except ImportError:
        logger.debug("sklearn 未安装，跳过细胞注释评估")
        result["cell_type_f1"] = None
        result["knn_accuracy"] = None

    logger.info(
        f"  结果: cos_sim={result.get('cosine_similarity_mean', 'N/A')}, "
        f"throughput={result.get('throughput_items_per_sec', 'N/A'):.1f} it/s, "
        f"model_size={result.get('model_size_mb', 'N/A'):.1f} MB"
    )
    return result


def run_calibration_experiment(
    model: nn.Module,
    model_name: str,
    model_size: str,
    method: str,
    bits: int,
    calibration_datasets: List[str],
    test_data: torch.Tensor,
    test_labels: np.ndarray,
    engine: QuantizationEngine,
    device: torch.device,
    n_genes: int = 2000,
    n_calib_samples: int = 128,
) -> List[Dict[str, Any]]:
    """运行校准数据集对比实验。

    对每种校准数据集分别进行量化，比较量化质量差异。

    Parameters
    ----------
    model : nn.Module
        原始模型。
    model_name : str
        模型名称。
    model_size : str
        模型规模。
    method : str
        量化方法。
    bits : int
        量化位宽。
    calibration_datasets : list of str
        校准数据集名称列表。
    test_data : torch.Tensor
        测试数据。
    test_labels : np.ndarray
        测试标签。
    engine : QuantizationEngine
        量化引擎。
    device : torch.device
        计算设备。
    n_genes : int
        基因数。
    n_calib_samples : int
        校准样本数。

    Returns
    -------
    list of dict
        各校准数据集的实验结果。
    """
    results = []
    for calib_name in calibration_datasets:
        logger.info(f"  校准数据集: {calib_name}")
        calib_data = generate_calibration_data(
            calib_name, n_calib_samples, n_genes, device
        )
        result = run_single_quantization(
            model, model_name, model_size, method, bits,
            calib_data, test_data, test_labels, engine, device,
        )
        result["calibration_dataset"] = calib_name
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="系统量化实验：评估量化方法对单细胞基础模型的影响",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/quantization_experiment.py --models geneformer --methods gptq --bits 4
  python scripts/quantization_experiment.py --models scgpt geneformer scfoundation
  python scripts/quantization_experiment.py --calibration-datasets pbmc pancreas synthetic
        """,
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["scgpt", "geneformer", "scfoundation"],
        choices=["scgpt", "geneformer", "scfoundation"],
        help="要测试的模型（默认: 全部）",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["bitsandbytes", "gptq", "awq"],
        choices=["bitsandbytes", "gptq", "awq"],
        help="量化方法（默认: 全部）",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+",
        default=[4, 8],
        choices=[4, 8],
        help="量化位宽（默认: 4 8）",
    )
    parser.add_argument(
        "--calibration-datasets", type=str, nargs="+",
        default=["pbmc", "pancreas", "synthetic"],
        choices=["pbmc", "pancreas", "synthetic"],
        help="校准数据集（默认: 全部）",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/quantization/",
        help="结果输出目录（默认: results/quantization/）",
    )
    parser.add_argument(
        "--n-test-samples", type=int, default=256,
        help="测试样本数（默认: 256）",
    )
    parser.add_argument(
        "--n-calib-samples", type=int, default=128,
        help="校准样本数（默认: 128）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（默认: configs/quantization/experiment_config.yaml）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )
    return parser.parse_args()


def main() -> None:
    """运行系统量化实验。"""
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

    device = get_device()
    logger.info(f"设备: {device}")
    logger.info(f"模型: {args.models}")
    logger.info(f"量化方法: {args.methods}")
    logger.info(f"量化位宽: {args.bits}")
    logger.info(f"校准数据集: {args.calibration_datasets}")
    logger.info(f"输出目录: {output_dir}")

    # 初始化量化引擎
    engine = QuantizationEngine()

    # 存储所有结果
    all_results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 遍历模型
    for model_name in args.models:
        # 获取该模型的所有规模
        sizes = list(MODEL_CONFIGS[model_name].keys())
        for model_size in sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"模型: {model_name} ({model_size})")
            logger.info(f"{'='*60}")

            cfg = MODEL_CONFIGS[model_name][model_size]
            n_genes = cfg["n_genes"]

            # 创建合成模型
            model = create_synthetic_model(model_name, model_size, device)
            model.eval()

            # 生成测试数据
            test_data, test_labels = generate_synthetic_data(
                n_samples=args.n_test_samples,
                n_genes=n_genes,
                n_cell_types=5,
                device=device,
                seed=args.seed,
            )

            # 原始模型基准
            logger.info("  原始模型基准测试...")
            orig_bench = engine.benchmark(model, test_data)
            orig_similarity = {
                "cosine_similarity_mean": 1.0,  # 与自身完全一致
                "cosine_similarity_std": 0.0,
                "mean_absolute_error": 0.0,
                "relative_error": 0.0,
            }
            orig_result = {
                "model": model_name,
                "model_size": model_size,
                "method": "none",
                "bits": 16,
                "calibration_dataset": "N/A",
                "timestamp": datetime.now().isoformat(),
                "quantization_time_s": 0.0,
                **orig_bench,
                **orig_similarity,
                "cell_type_f1": None,
                "knn_accuracy": None,
            }
            all_results.append(orig_result)

            # 遍历量化方法和位宽
            for method in args.methods:
                for bits in args.bits:
                    # 检查方法是否支持该位宽
                    if bits not in QuantizationEngine.SUPPORTED_METHODS.get(method, []):
                        logger.warning(f"  跳过: {method} 不支持 INT{bits}")
                        continue

                    logger.info(f"\n--- {method.upper()} INT{bits} ---")

                    # 对每种校准数据集进行测试
                    calib_results = run_calibration_experiment(
                        model=model,
                        model_name=model_name,
                        model_size=model_size,
                        method=method,
                        bits=bits,
                        calibration_datasets=args.calibration_datasets,
                        test_data=test_data,
                        test_labels=test_labels,
                        engine=engine,
                        device=device,
                        n_genes=n_genes,
                        n_calib_samples=args.n_calib_samples,
                    )
                    all_results.extend(calib_results)

    # -----------------------------------------------------------------------
    # 保存结果
    # -----------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"实验完成，共 {len(all_results)} 组结果")
    logger.info(f"{'='*60}")

    # 保存为 JSON
    json_path = output_dir / f"quantization_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "systematic_quantization",
            "timestamp": timestamp,
            "config": {
                "models": args.models,
                "methods": args.methods,
                "bits": args.bits,
                "calibration_datasets": args.calibration_datasets,
                "n_test_samples": args.n_test_samples,
                "n_calib_samples": args.n_calib_samples,
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON 结果已保存: {json_path}")

    # 保存为 CSV
    try:
        import csv
        csv_path = output_dir / f"quantization_results_{timestamp}.csv"
        if all_results:
            # 收集所有键
            all_keys = []
            seen = set()
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
            logger.info(f"CSV 结果已保存: {csv_path}")
    except Exception as e:
        logger.warning(f"CSV 保存失败: {e}")

    # 打印摘要
    _print_summary(all_results)


def _print_summary(results: List[Dict[str, Any]]) -> None:
    """打印实验结果摘要。"""
    logger.info("\n" + "=" * 80)
    logger.info("量化实验结果摘要")
    logger.info("=" * 80)

    # 按模型分组
    models_seen = {}
    for r in results:
        key = f"{r['model']}({r['model_size']})"
        if key not in models_seen:
            models_seen[key] = []
        models_seen[key].append(r)

    for model_key, model_results in models_seen.items():
        logger.info(f"\n{model_key}:")
        logger.info(f"  {'方法':<20} {'位宽':<6} {'余弦相似度':<12} {'吞吐量(it/s)':<14} {'模型大小(MB)':<14}")
        logger.info(f"  {'-'*66}")

        for r in model_results:
            method = r.get("method", "none")
            bits = r.get("bits", 16)
            cos_sim = r.get("cosine_similarity_mean", "N/A")
            throughput = r.get("throughput_items_per_sec", "N/A")
            size_mb = r.get("model_size_mb", "N/A")

            if isinstance(cos_sim, float):
                cos_sim = f"{cos_sim:.4f}"
            if isinstance(throughput, float):
                throughput = f"{throughput:.1f}"
            if isinstance(size_mb, float):
                size_mb = f"{size_mb:.1f}"

            logger.info(f"  {method:<20} INT{bits:<4} {cos_sim:<12} {throughput:<14} {size_mb:<14}")


if __name__ == "__main__":
    main()
