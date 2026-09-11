#!/usr/bin/env python
"""推理 Benchmark 脚本 — 单细胞基础模型统一推理速度对比

对每个模型测量：
- 吞吐量 (it/s)
- 延迟 (ms/cell)
- 峰值显存 (MB)

测量规范：warmup 10次 → 正式测量 50次 → 报告中位数 ± 标准差
多 batch size 测试：1, 4, 16, 64, 256
结果导出为 CSV 和 JSON，并生成 Markdown 格式对比表格。
如果模型未安装，使用合成数据（随机 tensor）进行 benchmark。

Usage
-----
    python scripts/benchmark_inference.py --models scgpt geneformer scfoundation --batch-sizes 1 4 16 64 --output results/
    python scripts/benchmark_inference.py --models all --output results/
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from loguru import logger

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path，确保 scinfer 可被导入
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scinfer.evaluation import BenchmarkRunner, BenchmarkResult
from scinfer.evaluation.metrics.inference import InferenceMetrics


# ===========================================================================
# 合成模型：当真实模型未安装时，使用合成 Transformer 模型进行 benchmark
# ===========================================================================

class SyntheticSCModel(nn.Module):
    """模拟单细胞基础模型的合成 Transformer 模型。

    架构与 scGPT / Geneformer 类似：Embedding → N 层 Transformer → LayerNorm → Head。
    """

    def __init__(
        self,
        n_genes: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes + 1, d_model)
        self.pos_embedding = nn.Embedding(n_genes + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：接收 token ids 或浮点 tensor，返回嵌入向量。"""
        # 兼容处理：支持 1D 和 2D 输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        # 如果输入是浮点 tensor（非 token ids），取 argmax 作为 token
        if x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            token_ids = x.argmax(dim=-1).clamp(0, self.gene_embedding.num_embeddings - 1).long()
        else:
            token_ids = x.long().clamp(0, self.gene_embedding.num_embeddings - 1)
        gene_emb = self.gene_embedding(token_ids)
        pos_emb = self.pos_embedding(positions.long())
        x_emb = gene_emb + pos_emb
        out = self.transformer_encoder(x_emb)
        out = self.output_norm(out)
        out = self.output_head(out)
        return out


# ===========================================================================
# 模型加载工具
# ===========================================================================

def _get_device(gpu_id: Optional[int]) -> torch.device:
    """根据命令行参数确定计算设备。"""
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_model_or_synthetic(
    model_name: str,
    device: torch.device,
) -> Tuple[nn.Module, str, bool]:
    """尝试加载真实模型，失败则返回合成模型。

    Returns
    -------
    tuple
        (model, display_name, is_synthetic)
    """
    _LOADERS = {
        "scgpt": _try_load_scgpt,
        "geneformer": _try_load_geneformer,
        "scfoundation": _try_load_scfoundation,
        "uce": _try_load_uce,
    }

    loader = _LOADERS.get(model_name.lower())
    if loader is not None:
        try:
            model = loader(device)
            logger.info(f"成功加载真实模型: {model_name}")
            return model, model_name, False
        except Exception as e:
            logger.warning(f"无法加载 {model_name}: {e}，使用合成模型替代")

    model = SyntheticSCModel(n_genes=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"使用合成模型 '{model_name}'（参数量: {n_params / 1e6:.1f}M）")
    return model, f"{model_name}(synthetic)", True


def _try_load_scgpt(device: torch.device) -> nn.Module:
    """尝试加载 scGPT 模型。"""
    try:
        from scgpt.model import TransformerModel
        model = TransformerModel(
            ntoken=60000, d_model=512, nhead=8, d_hid=2048,
            nlayers=6, n_cls=0, pad_token=0,
        )
        return model.to(device)
    except ImportError:
        raise ImportError("scgpt 未安装")


def _try_load_geneformer(device: torch.device) -> nn.Module:
    """尝试加载 Geneformer 模型。"""
    try:
        from geneformer import GeneformerModel
        model = GeneformerModel.from_pretrained("ctheodoris/Geneformer")
        return model.to(device)
    except ImportError:
        raise ImportError("geneformer 未安装")


def _try_load_scfoundation(device: torch.device) -> nn.Module:
    """尝试加载 scFoundation 模型。"""
    try:
        from scfoundation import Geneformer
        model = Geneformer.from_pretrained("scfoundation")
        return model.to(device)
    except ImportError:
        raise ImportError("scfoundation 未安装")


def _try_load_uce(device: torch.device) -> nn.Module:
    """尝试加载 UCE 模型。"""
    try:
        from uce import UCEModel
        model = UCEModel.from_pretrained("uce")
        return model.to(device)
    except ImportError:
        raise ImportError("uce 未安装")


# ===========================================================================
# Benchmark 核心逻辑
# ===========================================================================

def benchmark_single_model_at_batch(
    model: nn.Module,
    model_name: str,
    batch_size: int,
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, Any]:
    """对单个模型在指定 batch size 下进行完整 benchmark。

    测量：吞吐量(it/s)、延迟(ms/cell)、峰值显存(MB)
    规范：warmup 10次 → 正式 50次 → 中位数 ± 标准差
    """
    # 构造输入数据
    input_length = 512  # 默认基因数量
    input_data = torch.randn(batch_size, input_length, device=device)

    metrics: Dict[str, Any] = {}

    # 1) 吞吐量测量
    throughput_result = InferenceMetrics.measure_throughput(
        model, input_data, batch_size=batch_size,
        n_warmup=warmup, n_measure=runs,
    )
    metrics["throughput_median_ips"] = throughput_result.median
    metrics["throughput_std_ips"] = throughput_result.std
    metrics["throughput_all"] = throughput_result.all_measurements

    # 2) 延迟测量
    latency_result = InferenceMetrics.measure_latency(
        model, input_data, n_warmup=warmup, n_measure=runs,
    )
    metrics["latency_median_ms"] = latency_result.median_ms
    metrics["latency_std_ms"] = latency_result.std_ms
    metrics["latency_percentiles"] = latency_result.percentiles
    metrics["latency_all_ms"] = latency_result.all_measurements_ms

    # 3) 峰值显存测量
    memory_result = InferenceMetrics.measure_peak_memory(
        model, input_data, batch_size=batch_size,
    )
    metrics["peak_gpu_memory_mb"] = memory_result.peak_gpu_mb
    metrics["cpu_memory_mb"] = memory_result.cpu_mb
    metrics["model_params_mb"] = memory_result.model_params_mb

    # 4) 每 cell 延迟（总延迟 / batch_size）
    metrics["latency_per_cell_ms"] = latency_result.median_ms / max(batch_size, 1)

    return metrics


def run_benchmark_experiment(
    model_names: List[str],
    batch_sizes: List[int],
    gpu_id: Optional[int],
    warmup: int,
    runs: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """运行完整的 benchmark 实验：多模型 × 多 batch size。

    Returns
    -------
    dict
        包含所有 benchmark 结果的字典。
    """
    device = _get_device(gpu_id)
    logger.info(f"使用设备: {device}")

    all_results: Dict[str, Any] = {
        "experiment_config": {
            "models": model_names,
            "batch_sizes": batch_sizes,
            "device": str(device),
            "warmup": warmup,
            "runs": runs,
            "timestamp": datetime.utcnow().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "gpu_info": {},
        "benchmark_results": {},
    }

    # GPU 信息
    if torch.cuda.is_available():
        gpu_idx = gpu_id if gpu_id is not None else 0
        props = torch.cuda.get_device_properties(gpu_idx)
        all_results["gpu_info"] = {
            "name": torch.cuda.get_device_name(gpu_idx),
            "total_memory_mb": props.total_memory / (1024 ** 2),
            "cuda_version": torch.version.cuda,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    else:
        all_results["gpu_info"] = {"name": "CPU", "note": "CUDA not available"}

    # 逐模型 benchmark
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'='*60}")

        model, display_name, is_synthetic = load_model_or_synthetic(model_name, device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数量: {n_params / 1e6:.1f}M")

        model_results = {}
        for bs in batch_sizes:
            logger.info(f"  Batch size = {bs}")
            try:
                result = benchmark_single_model_at_batch(
                    model, display_name, bs, device, warmup, runs,
                )
                model_results[bs] = result
                logger.info(
                    f"    吞吐量: {result['throughput_median_ips']:.1f} it/s | "
                    f"延迟: {result['latency_median_ms']:.2f} ms | "
                    f"显存: {result['peak_gpu_memory_mb']:.1f} MB"
                )
            except Exception as e:
                logger.error(f"    Batch size {bs} 失败: {e}")
                model_results[bs] = {"error": str(e)}

        all_results["benchmark_results"][display_name] = {
            "is_synthetic": is_synthetic,
            "n_params": n_params,
            "batch_results": {str(k): v for k, v in model_results.items()},
        }

        # 清理 GPU 内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ===========================================================================
# 结果导出工具
# ===========================================================================

def export_json(results: Dict[str, Any], path: Path) -> None:
    """导出 JSON 格式结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"JSON 结果已保存到 {path}")


def export_csv(results: Dict[str, Any], path: Path) -> None:
    """导出 CSV 格式结果（扁平化表格）。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model_data in results.get("benchmark_results", {}).items():
        for bs_str, metrics in model_data.get("batch_results", {}).items():
            if "error" in metrics:
                continue
            row = {
                "model": model_name,
                "batch_size": bs_str,
                "throughput_median_ips": metrics.get("throughput_median_ips", ""),
                "throughput_std_ips": metrics.get("throughput_std_ips", ""),
                "latency_median_ms": metrics.get("latency_median_ms", ""),
                "latency_std_ms": metrics.get("latency_std_ms", ""),
                "latency_per_cell_ms": metrics.get("latency_per_cell_ms", ""),
                "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", ""),
                "model_params_mb": metrics.get("model_params_mb", ""),
            }
            rows.append(row)

    if not rows:
        logger.warning("没有可导出的数据")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV 结果已保存到 {path}")


def export_markdown(results: Dict[str, Any], path: Path) -> None:
    """导出 Markdown 格式对比表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 单细胞基础模型推理 Benchmark 结果",
        "",
        f"*生成时间: {results['experiment_config'].get('timestamp', 'N/A')}*",
        "",
        "## 实验配置",
        "",
        f"- **设备**: {results['experiment_config'].get('device', 'N/A')}",
        f"- **GPU**: {results.get('gpu_info', {}).get('name', 'N/A')}",
        f"- **Warmup**: {results['experiment_config'].get('warmup', 'N/A')} 次",
        f"- **测量次数**: {results['experiment_config'].get('runs', 'N/A')} 次",
        "",
    ]

    # 按 batch size 生成对比表
    batch_sizes = results["experiment_config"].get("batch_sizes", [])
    benchmark_data = results.get("benchmark_results", {})

    for bs in batch_sizes:
        bs_str = str(bs)
        lines.append(f"## Batch Size = {bs}")
        lines.append("")
        lines.append("| 模型 | 吞吐量 (it/s) | 延迟 (ms) | 每cell延迟 (ms) | 峰值显存 (MB) | 参数量 (M) |")
        lines.append("|------|--------------|-----------|----------------|--------------|-----------|")

        for model_name, model_data in benchmark_data.items():
            batch_res = model_data.get("batch_results", {}).get(bs_str, {})
            if "error" in batch_res:
                lines.append(f"| {model_name} | ERROR | - | - | - | - |")
                continue
            tp = batch_res.get("throughput_median_ips", 0)
            tp_std = batch_res.get("throughput_std_ips", 0)
            lat = batch_res.get("latency_median_ms", 0)
            lat_std = batch_res.get("latency_std_ms", 0)
            lat_cell = batch_res.get("latency_per_cell_ms", 0)
            mem = batch_res.get("peak_gpu_memory_mb", 0)
            n_params = model_data.get("n_params", 0) / 1e6
            lines.append(
                f"| {model_name} | {tp:.1f} ± {tp_std:.1f} | "
                f"{lat:.2f} ± {lat_std:.2f} | {lat_cell:.3f} | "
                f"{mem:.1f} | {n_params:.1f} |"
            )
        lines.append("")

    # 汇总对比表（取 batch_size=16 的结果）
    lines.append("## 汇总对比（Batch Size = 16）")
    lines.append("")
    summary_bs = "16"
    lines.append("| 模型 | 吞吐量 (it/s) | 延迟 (ms) | 峰值显存 (MB) |")
    lines.append("|------|--------------|-----------|--------------|")
    for model_name, model_data in benchmark_data.items():
        batch_res = model_data.get("batch_results", {}).get(summary_bs, {})
        if "error" in batch_res:
            lines.append(f"| {model_name} | ERROR | - | - |")
            continue
        tp = batch_res.get("throughput_median_ips", 0)
        lat = batch_res.get("latency_median_ms", 0)
        mem = batch_res.get("peak_gpu_memory_mb", 0)
        lines.append(f"| {model_name} | {tp:.1f} | {lat:.2f} | {mem:.1f} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown 结果已保存到 {path}")


# ===========================================================================
# 命令行接口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="推理 Benchmark 脚本 — 单细胞基础模型统一推理速度对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/benchmark_inference.py --models scgpt geneformer scfoundation --batch-sizes 1 4 16 64
  python scripts/benchmark_inference.py --models all --output results/
  python scripts/benchmark_inference.py --models scgpt --batch-sizes 1 4 16 64 256 --gpu 0
        """,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["scgpt"],
        help="要 benchmark 的模型（scgpt, geneformer, scfoundation, uce, all）",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256],
        help="batch size 列表（默认: 1 4 16 64 256）",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU 设备 ID（默认: None 使用 CPU）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="预热次数（默认: 10）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="正式测量次数（默认: 50）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="输出目录（默认: results/）",
    )
    return parser.parse_args()


def main() -> None:
    """运行推理 benchmark 主入口。"""
    args = parse_args()

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

    # 展开 "all" 为全部模型
    model_names = args.models
    if "all" in model_names:
        model_names = ["scgpt", "geneformer", "scfoundation", "uce"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"推理 Benchmark 开始")
    logger.info(f"模型: {model_names}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Warmup: {args.warmup}, Runs: {args.runs}")
    logger.info(f"输出目录: {output_dir}")

    # 运行 benchmark 实验
    results = run_benchmark_experiment(
        model_names=model_names,
        batch_sizes=args.batch_sizes,
        gpu_id=args.gpu,
        warmup=args.warmup,
        runs=args.runs,
        output_dir=output_dir,
    )

    # 导出结果
    export_json(results, output_dir / "benchmark_results.json")
    export_csv(results, output_dir / "benchmark_results.csv")
    export_markdown(results, output_dir / "benchmark_results.md")

    # 打印摘要
    print("\n" + "=" * 70)
    print("Benchmark 结果摘要")
    print("=" * 70)
    for model_name, model_data in results["benchmark_results"].items():
        print(f"\n--- {model_name} (参数量: {model_data.get('n_params', 0) / 1e6:.1f}M) ---")
        for bs_str, metrics in model_data.get("batch_results", {}).items():
            if "error" in metrics:
                print(f"  BS={bs_str:>4s}: ERROR - {metrics['error']}")
                continue
            tp = metrics.get("throughput_median_ips", 0)
            lat = metrics.get("latency_median_ms", 0)
            mem = metrics.get("peak_gpu_memory_mb", 0)
            print(f"  BS={bs_str:>4s}: {tp:>10.1f} it/s | {lat:>8.2f} ms | {mem:>8.1f} MB")
    print("=" * 70)
    print(f"\n结果已保存到 {output_dir}/")


if __name__ == "__main__":
    main()
