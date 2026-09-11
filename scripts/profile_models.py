#!/usr/bin/env python
"""推理 Profiling 脚本 — 单细胞基础模型推理瓶颈分析

对 scGPT / Geneformer / scFoundation 进行深度推理 profiling，包括：
- 时间分解：使用 PyTorch profiler 将各层耗时分类为 attention / FFN / normalization / embedding / 其他
- 内存分解：逐层统计参数内存与激活值内存
- 不同输入长度测试：128 / 512 / 2048 / 4096 genes 下的瓶颈变化
- 与 GPT-2 124M 对比：加载 GPT-2 作为 LLM 参考模型
- 如果模型未安装，使用合成数据（随机 tensor）进行 profiling（确保脚本始终可运行）

Usage
-----
    python scripts/profile_models.py --model scgpt --input-lengths 128 512 2048 --gpu 0 --output results/
    python scripts/profile_models.py --model all --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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

from scinfer.evaluation import InferenceProfiler, ProfileResult


# ===========================================================================
# 合成模型：当真实模型未安装时，使用合成 Transformer 模型进行 profiling
# ===========================================================================

class SyntheticSCModel(nn.Module):
    """模拟单细胞基础模型的合成 Transformer 模型。

    架构与 scGPT / Geneformer 类似：Embedding → N 层 Transformer → LayerNorm → Head。
    用于在模型未安装时提供可运行的 profiling 数据。
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
        """前向传播：接收 token ids，返回嵌入向量。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        gene_emb = self.gene_embedding(x.long())
        pos_emb = self.pos_embedding(positions.long())
        x_emb = gene_emb + pos_emb
        out = self.transformer_encoder(x_emb)
        out = self.output_norm(out)
        out = self.output_head(out)
        return out


class SyntheticGPT2Model(nn.Module):
    """模拟 GPT-2 124M 的合成模型，用作 LLM 参考对比。

    参数量约 124M：12 层 Transformer，d_model=768，n_heads=12。
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, n_layers: int = 12, vocab_size: int = 50257) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(1024, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        h = self.wte(x.long()) + self.wpe(positions.long())
        h = self.transformer(h)
        h = self.ln_f(h)
        return h


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
    n_genes: int = 512,
) -> Tuple[nn.Module, str, bool]:
    """尝试加载真实模型，失败则返回合成模型。

    Returns
    -------
    tuple
        (model, display_name, is_synthetic)
    """
    # 尝试加载真实模型的策略
    _LOADERS = {
        "scgpt": _try_load_scgpt,
        "geneformer": _try_load_geneformer,
        "scfoundation": _try_load_scfoundation,
        "gpt2": _try_load_gpt2,
    }

    loader = _LOADERS.get(model_name.lower())
    if loader is not None:
        try:
            model = loader(device)
            logger.info(f"成功加载真实模型: {model_name}")
            return model, model_name, False
        except Exception as e:
            logger.warning(f"无法加载 {model_name}: {e}，使用合成模型替代")

    # 合成模型 fallback
    if model_name.lower() == "gpt2":
        model = SyntheticGPT2Model().to(device)
    else:
        model = SyntheticSCModel(n_genes=n_genes).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"使用合成模型 '{model_name}'（参数量: {n_params / 1e6:.1f}M）")
    return model, f"{model_name}(synthetic)", True


def _try_load_scgpt(device: torch.device) -> nn.Module:
    """尝试加载 scGPT 模型。"""
    try:
        from scgpt.model import TransformerModel
        # scGPT 典型配置
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


def _try_load_gpt2(device: torch.device) -> nn.Module:
    """尝试加载 GPT-2 124M 模型。"""
    try:
        from transformers import GPT2Model, GPT2Config
        config = GPT2Config()  # 默认即 124M
        model = GPT2Model(config)
        return model.to(device)
    except ImportError:
        raise ImportError("transformers 未安装")


# ===========================================================================
# Profiling 核心逻辑
# ===========================================================================

def profile_single_model(
    model: nn.Module,
    model_name: str,
    input_length: int,
    device: torch.device,
    warmup: int = 3,
    runs: int = 10,
) -> Dict[str, Any]:
    """对单个模型在指定输入长度下进行完整 profiling。

    包括：
    1. 使用 InferenceProfiler.profile() 进行逐层时间分解
    2. 使用 InferenceProfiler.profile_time() 进行 PyTorch profiler 时间分解
    3. 使用 InferenceProfiler.profile_memory() 进行内存分解
    4. 使用 InferenceProfiler.estimate_flops() 进行 FLOPs 估计
    """
    profiler = InferenceProfiler(
        model_name=model_name,
        warmup_runs=warmup,
        profile_runs=runs,
    )

    # 构造输入数据：token ids（模拟基因 token）
    batch_size = 8
    input_ids = torch.randint(0, 1000, (batch_size, input_length), device=device)

    # 1) 逐层 profiling（hook-based）
    logger.info(f"  [Hook-based profiling] {model_name} @ {input_length} genes")
    profile_result = profiler.profile(model, input_ids)

    # 2) PyTorch profiler 细粒度时间分解
    logger.info(f"  [torch.profiler] 细粒度时间分解")
    time_detail = profiler.profile_time(model, adapter=None, input_data=input_ids)

    # 3) 内存分解
    logger.info(f"  [Memory profiling] 逐层内存分解")
    mem_detail = profiler.profile_memory(model, adapter=None)

    # 4) FLOPs 估计
    logger.info(f"  [FLOPs estimation] 计算量估计")
    flops = profiler.estimate_flops(model, adapter=None, input_shape=(batch_size, input_length, 512))

    # 汇总结果
    result = {
        "model_name": model_name,
        "input_length": input_length,
        "device": str(device),
        # Hook-based profiling
        "total_latency_ms": profile_result.total_latency_ms,
        "peak_memory_mb": profile_result.peak_memory_mb,
        "time_breakdown": profile_result.time_breakdown,
        "bottleneck_layers": profile_result.bottleneck_layers,
        "per_layer_latency_ms": dict(
            sorted(profile_result.per_layer_latency_ms.items(), key=lambda x: -x[1])[:20]
        ),
        # torch.profiler 细粒度
        "torch_profiler_breakdown_ms": time_detail["breakdown_ms"],
        "torch_profiler_percent": time_detail["percent"],
        # 内存
        "memory_breakdown": mem_detail,
        # FLOPs
        "flops": flops,
    }

    logger.info(
        f"  结果: {profile_result.total_latency_ms:.2f} ms, "
        f"{profile_result.peak_memory_mb:.1f} MB peak GPU mem"
    )
    return result


def run_profiling_experiment(
    model_names: List[str],
    input_lengths: List[int],
    gpu_id: Optional[int],
    warmup: int,
    runs: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """运行完整的 profiling 实验：多模型 × 多输入长度。

    Returns
    -------
    dict
        包含所有 profiling 结果的字典。
    """
    device = _get_device(gpu_id)
    logger.info(f"使用设备: {device}")

    all_results: Dict[str, Any] = {
        "experiment_config": {
            "models": model_names,
            "input_lengths": input_lengths,
            "device": str(device),
            "warmup": warmup,
            "runs": runs,
            "gpu_available": torch.cuda.is_available(),
        },
        "gpu_info": {},
        "profiling_results": {},
    }

    # GPU 信息
    if torch.cuda.is_available():
        all_results["gpu_info"] = {
            "name": torch.cuda.get_device_name(gpu_id or 0),
            "total_memory_mb": torch.cuda.get_device_properties(gpu_id or 0).total_memory / (1024 ** 2),
            "cuda_version": torch.version.cuda,
            "compute_capability": f"{torch.cuda.get_device_properties(gpu_id or 0).major}.{torch.cuda.get_device_properties(gpu_id or 0).minor}",
        }

    # 逐模型 × 逐输入长度 profiling
    for model_name in model_names:
        model_results = []
        for length in input_lengths:
            logger.info(f"\n{'='*60}")
            logger.info(f"Profiling: {model_name} @ {length} genes")
            logger.info(f"{'='*60}")

            model, display_name, is_synthetic = load_model_or_synthetic(
                model_name, device, n_genes=max(length, 512)
            )

            result = profile_single_model(
                model, display_name, length, device, warmup, runs
            )
            result["is_synthetic"] = is_synthetic
            model_results.append(result)

            # 清理 GPU 内存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results["profiling_results"][model_name] = model_results

    return all_results


# ===========================================================================
# 命令行接口
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="推理 Profiling 脚本 — 单细胞基础模型推理瓶颈分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/profile_models.py --model scgpt --input-lengths 128 512 2048 --gpu 0
  python scripts/profile_models.py --model all --output results/
  python scripts/profile_models.py --model scgpt geneformer gpt2 --input-lengths 128 512 2048 4096
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["scgpt"],
        help="要 profiling 的模型（scgpt, geneformer, scfoundation, uce, gpt2, all）",
    )
    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 4096],
        help="输入基因数量列表（默认: 128 512 2048 4096）",
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
        default=3,
        help="预热次数（默认: 3）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="profiling 运行次数（默认: 10）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="输出目录（默认: results/）",
    )
    return parser.parse_args()


def main() -> None:
    """运行推理 profiling 主入口。"""
    args = parse_args()

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

    # 展开 "all" 为全部模型
    model_names = args.model
    if "all" in model_names:
        model_names = ["scgpt", "geneformer", "scfoundation", "gpt2"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"推理 Profiling 开始")
    logger.info(f"模型: {model_names}")
    logger.info(f"输入长度: {args.input_lengths}")
    logger.info(f"输出目录: {output_dir}")

    # 运行 profiling 实验
    results = run_profiling_experiment(
        model_names=model_names,
        input_lengths=args.input_lengths,
        gpu_id=args.gpu,
        warmup=args.warmup,
        runs=args.runs,
        output_dir=output_dir,
    )

    # 保存 JSON 结果
    json_path = output_dir / "profiling_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n结果已保存到 {json_path}")

    # 打印摘要
    print("\n" + "=" * 70)
    print("Profiling 结果摘要")
    print("=" * 70)
    for model_name, model_results in results["profiling_results"].items():
        print(f"\n--- {model_name} ---")
        for r in model_results:
            length = r["input_length"]
            latency = r["total_latency_ms"]
            peak_mem = r["peak_memory_mb"]
            breakdown = r.get("time_breakdown", {})
            # 找出主要瓶颈
            dominant = max(breakdown, key=breakdown.get) if breakdown else "N/A"
            print(f"  {length:>5d} genes: {latency:>8.2f} ms | GPU mem: {peak_mem:>8.1f} MB | 主要瓶颈: {dominant}")
    print("=" * 70)


if __name__ == "__main__":
    main()
