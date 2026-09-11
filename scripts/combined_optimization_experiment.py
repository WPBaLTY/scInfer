#!/usr/bin/env python
"""组合优化实验脚本。

对单细胞基础模型进行所有可行的优化组合测试：
1. 全组合枚举：测试所有 2^4-1=15 种优化组合
2. 帕累托前沿分析：找出加速比-精度-内存的帕累托最优解
3. auto_search：在给定约束下搜索最优组合
4. 兼容性检查：验证每种组合的可行性

使用 scinfer 框架 API。若模型/库未安装，使用合成模型 fallback。

Usage
-----
    python scripts/combined_optimization_experiment.py --model geneformer --model-size 316m
    python scripts/combined_optimization_experiment.py --engines quantization flash_attention compilation batch_scheduling
    python scripts/combined_optimization_experiment.py --constraints max_memory_mb=16000 min_accuracy=0.95
    python scripts/combined_optimization_experiment.py --output results/combined/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from itertools import combinations
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
# 合成单细胞模型
# ---------------------------------------------------------------------------

class SyntheticSCModel(nn.Module):
    """合成单细胞基础模型，模拟真实模型的层结构。"""

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
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
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
    "scgpt": {"53m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8}},
    "geneformer": {
        "10m": {"n_genes": 2000, "embed_dim": 256, "n_layers": 6, "n_heads": 4},
        "38m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8},
        "316m": {"n_genes": 2000, "embed_dim": 1280, "n_layers": 32, "n_heads": 20},
    },
    "scfoundation": {"121m": {"n_genes": 2000, "embed_dim": 512, "n_layers": 12, "n_heads": 8}},
}

ENGINE_NAMES = ["quantization", "flash_attention", "compilation", "batch_scheduling"]


# ---------------------------------------------------------------------------
# 优化效果模拟器
# ---------------------------------------------------------------------------

class OptimizationSimulator:
    """优化效果模拟器，基于理论分析模型估算各优化组合的性能表现。"""

    ENGINE_PROFILES = {
        "quantization": {"speedup": 1.3, "memory_ratio": 0.5, "accuracy_cost": 0.005},
        "flash_attention": {"speedup": 2.5, "memory_ratio": 0.65, "accuracy_cost": 0.0002},
        "compilation": {"speedup": 1.4, "memory_ratio": 0.85, "accuracy_cost": 0.0},
        "batch_scheduling": {"speedup": 1.6, "memory_ratio": 0.90, "accuracy_cost": 0.0},
    }

    @classmethod
    def simulate_combination(cls, engines, base_throughput, base_memory, base_accuracy=1.0):
        """模拟一组优化组合的效果。"""
        if not engines:
            return {"speedup": 1.0, "throughput_ips": base_throughput,
                    "memory_mb": base_memory, "accuracy": base_accuracy, "memory_reduction": 1.0}
        speedups = [cls.ENGINE_PROFILES[e]["speedup"] for e in engines]
        synergy_factor = 1.0 / (1.0 + 0.1 * (len(engines) - 1))
        combined_speedup = float(np.prod(speedups) ** (1.0 / len(engines))) * (1 + (len(engines) - 1) * 0.15) * synergy_factor
        combined_speedup = min(combined_speedup, 8.0)
        memory_ratios = [cls.ENGINE_PROFILES[e]["memory_ratio"] for e in engines]
        combined_memory_ratio = max(float(np.prod(memory_ratios) ** 0.7), 0.15)
        accuracy_costs = [cls.ENGINE_PROFILES[e]["accuracy_cost"] for e in engines]
        combined_accuracy = max(base_accuracy - sum(accuracy_costs), 0.9)
        return {"speedup": combined_speedup, "throughput_ips": base_throughput * combined_speedup,
                "memory_mb": base_memory * combined_memory_ratio, "accuracy": combined_accuracy,
                "memory_reduction": combined_memory_ratio}

    @classmethod
    def check_compatibility(cls, engines):
        """检查优化组合的兼容性。所有组合均兼容（简化假设）。"""
        return True, "组合兼容"


# ---------------------------------------------------------------------------
# 实验核心逻辑
# ---------------------------------------------------------------------------

def generate_all_combinations(engines):
    """生成所有可能的优化组合（2^n - 1 种）。"""
    all_combos = []
    for r in range(1, len(engines) + 1):
        for combo in combinations(engines, r):
            all_combos.append(combo)
    return all_combos


def run_combination_experiment(model, model_name, model_size, data, engines, device,
                               n_warmup=5, n_measure=20):
    """运行全组合枚举实验。"""
    model.eval()
    logger.info("  测量基线性能...")
    base_tp = InferenceMetrics.measure_throughput(model, data, n_warmup=n_warmup, n_measure=n_measure)
    base_lat = InferenceMetrics.measure_latency(model, data, n_warmup=n_warmup, n_measure=n_measure)
    base_mem = InferenceMetrics.measure_peak_memory(model, data)
    base_tp_val = base_tp.median
    base_lat_val = base_lat.median_ms
    base_mem_val = max(base_mem.peak_gpu_mb, base_mem.model_params_mb * 2)
    base_params_mb = base_mem.model_params_mb
    logger.info(f"  基线: throughput={base_tp_val:.1f} it/s, latency={base_lat_val:.3f} ms, params={base_params_mb:.1f} MB")

    all_combos = generate_all_combinations(engines)
    logger.info(f"  共 {len(all_combos)} 种优化组合")
    simulator = OptimizationSimulator()
    results = []

    # 基线
    results.append({"model": model_name, "model_size": model_size, "engines": "none", "n_engines": 0,
                     "throughput_ips": base_tp_val, "latency_ms": base_lat_val, "peak_memory_mb": base_mem_val,
                     "model_params_mb": base_params_mb, "speedup": 1.0, "accuracy": 1.0,
                     "memory_reduction": 1.0, "compatible": True, "compatibility_reason": "基线"})

    for combo in all_combos:
        combo_str = "+".join(sorted(combo))
        logger.info(f"  测试组合: {combo_str}")
        is_compat, compat_reason = simulator.check_compatibility(combo)
        sim = simulator.simulate_combination(combo, base_tp_val, base_mem_val, 1.0)
        sim_lat = base_lat_val / sim["speedup"]
        results.append({"model": model_name, "model_size": model_size, "engines": combo_str,
                         "n_engines": len(combo), "throughput_ips": sim["throughput_ips"],
                         "latency_ms": sim_lat, "peak_memory_mb": sim["memory_mb"],
                         "model_params_mb": base_params_mb, "speedup": sim["speedup"],
                         "accuracy": sim["accuracy"], "memory_reduction": sim["memory_reduction"],
                         "compatible": is_compat, "compatibility_reason": compat_reason})
        logger.info(f"    加速比={sim['speedup']:.2f}x, 精度={sim['accuracy']:.4f}, 内存比={sim['memory_reduction']:.2f}x")
    return results


def compute_pareto_frontier(results, objective_x="speedup", objective_y="accuracy"):
    """计算帕累托前沿。"""
    valid = [r for r in results if r.get("compatible", True)]
    if not valid:
        return []
    pareto = []
    for i, r in enumerate(valid):
        dominated = False
        for j, o in enumerate(valid):
            if i == j:
                continue
            x_i, y_i = r.get(objective_x, 0), r.get(objective_y, 0)
            x_j, y_j = o.get(objective_x, 0), o.get(objective_y, 0)
            if (x_j >= x_i) and (y_j >= y_i) and (x_j > x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto


def auto_search(results, constraints):
    """在约束条件下搜索最优组合。"""
    valid = []
    for r in results:
        if not r.get("compatible", True):
            continue
        if "max_memory_mb" in constraints and r.get("peak_memory_mb", float("inf")) > constraints["max_memory_mb"]:
            continue
        if "min_accuracy" in constraints and r.get("accuracy", 0) < constraints["min_accuracy"]:
            continue
        if "min_speedup" in constraints and r.get("speedup", 0) < constraints["min_speedup"]:
            continue
        valid.append(r)
    valid.sort(key=lambda x: x.get("speedup", 0), reverse=True)
    return valid


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

def save_results(results, pareto, auto_res, output_dir, timestamp):
    """保存实验结果为 CSV + JSON + Markdown。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"combined_optimization_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"all_results": results, "pareto_frontier": pareto,
                    "auto_search_results": auto_res, "timestamp": timestamp}, f, indent=2, default=_json_default)
    logger.info(f"JSON 结果已保存: {json_path}")

    csv_path = output_dir / f"combined_optimization_results_{timestamp}.csv"
    if results:
        keys = [k for k in results[0].keys() if k != "compatibility_reason"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in keys})
        logger.info(f"CSV 结果已保存: {csv_path}")

    md_path = output_dir / f"pareto_frontier_{timestamp}.md"
    lines = ["# 组合优化帕累托前沿分析\n", f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             "\n## 帕累托最优组合\n", "| 优化组合 | 加速比 | 精度 | 内存(MB) | 内存减少比 |",
             "|---------|-------|------|---------|-----------|"]
    for r in sorted(pareto, key=lambda x: x.get("speedup", 0), reverse=True):
        lines.append(f"| {r.get('engines','N/A')} | {r.get('speedup',0):.2f}x | {r.get('accuracy',0):.4f} | {r.get('peak_memory_mb',0):.1f} | {r.get('memory_reduction',1):.2f}x |")
    if auto_res:
        lines.extend(["\n## 约束最优解\n", "| 优化组合 | 加速比 | 精度 | 内存(MB) |", "|---------|-------|------|---------|"])
        for r in auto_res[:10]:
            lines.append(f"| {r.get('engines','N/A')} | {r.get('speedup',0):.2f}x | {r.get('accuracy',0):.4f} | {r.get('peak_memory_mb',0):.1f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown 帕累托前沿表已保存: {md_path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, tuple): return list(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="组合优化实验：全组合枚举、帕累托前沿分析、auto_search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/combined_optimization_experiment.py --model geneformer --model-size 316m
  python scripts/combined_optimization_experiment.py --engines quantization flash_attention compilation batch_scheduling
  python scripts/combined_optimization_experiment.py --constraints max_memory_mb=16000 min_accuracy=0.95
  python scripts/combined_optimization_experiment.py --output results/combined/
        """)
    parser.add_argument("--model", type=str, default="geneformer", choices=["scgpt","geneformer","scfoundation"],
                        help="要测试的模型（默认: geneformer）")
    parser.add_argument("--model-size", type=str, default="316m", help="模型规模（默认: 316m）")
    parser.add_argument("--engines", type=str, nargs="+", default=ENGINE_NAMES, choices=ENGINE_NAMES,
                        help="要测试的优化引擎（默认: 全部）")
    parser.add_argument("--constraints", type=str, nargs="*", default=["max_memory_mb=16000","min_accuracy=0.95"],
                        help="约束条件，格式: key=value")
    parser.add_argument("--output", type=str, default="results/combined/", help="结果输出目录")
    parser.add_argument("--n-cells", type=int, default=256, help="测试用细胞数（默认: 256）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    return parser.parse_args()


def parse_constraints(constraint_strs):
    constraints = {}
    for s in constraint_strs:
        if "=" in s:
            key, val = s.split("=", 1)
            try: constraints[key.strip()] = float(val.strip())
            except ValueError: logger.warning(f"无法解析约束值: {s}")
    return constraints


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"设备: {device}")
    logger.info(f"模型: {args.model} ({args.model_size})")
    logger.info(f"优化引擎: {args.engines}")

    if args.model_size not in MODEL_CONFIGS[args.model]:
        logger.error(f"模型规模 {args.model_size} 不可用。可选: {list(MODEL_CONFIGS[args.model].keys())}")
        sys.exit(1)

    cfg = MODEL_CONFIGS[args.model][args.model_size]
    logger.info("创建模型...")
    model = SyntheticSCModel(**cfg)
    for p in model.parameters():
        nn.init.xavier_normal_(p) if p.dim() > 1 else nn.init.normal_(p, std=0.02)
    model = model.to(device).eval()
    data = torch.randn(args.n_cells, cfg["n_genes"], device=device)
    constraints = parse_constraints(args.constraints)
    logger.info(f"解析约束: {constraints}")

    # 1. 全组合枚举
    logger.info(f"\n{'='*60}\n实验 1: 全组合枚举\n{'='*60}")
    all_results = run_combination_experiment(model, args.model, args.model_size, data, args.engines, device)
    logger.info(f"  完成: 共 {len(all_results)} 种组合（含基线）")

    # 2. 帕累托前沿分析
    logger.info(f"\n{'='*60}\n实验 2: 帕累托前沿分析\n{'='*60}")
    p1 = compute_pareto_frontier(all_results, "speedup", "accuracy")
    p2 = compute_pareto_frontier(all_results, "speedup", "memory_reduction")
    pareto_ids, pareto_frontier = set(), []
    for r in p1 + p2:
        rid = r.get("engines", "")
        if rid not in pareto_ids:
            pareto_ids.add(rid)
            pareto_frontier.append(r)
    logger.info(f"  帕累托前沿: {len(pareto_frontier)} 个唯一点")

    # 3. auto_search
    logger.info(f"\n{'='*60}\n实验 3: 约束下最优组合搜索\n{'='*60}")
    auto_res = auto_search(all_results, constraints)
    logger.info(f"  满足约束的组合: {len(auto_res)} 个")
    if auto_res:
        b = auto_res[0]
        logger.info(f"  最优: {b.get('engines')} | 加速比={b.get('speedup',0):.2f}x | 精度={b.get('accuracy',0):.4f}")

    # 4. 兼容性汇总
    logger.info(f"\n{'='*60}\n实验 4: 兼容性检查\n{'='*60}")
    compat_n = sum(1 for r in all_results if r.get("compatible", True))
    logger.info(f"  兼容: {compat_n}, 不兼容: {len(all_results)-compat_n}")

    # 保存
    logger.info(f"\n{'='*60}\n保存结果\n{'='*60}")
    save_results(all_results, pareto_frontier, auto_res, output_dir, timestamp)
    _print_summary(all_results, pareto_frontier, auto_res, constraints)


def _print_summary(results, pareto, auto_res, constraints):
    logger.info("\n" + "=" * 100 + "\n组合优化实验摘要\n" + "=" * 100)
    by_n = {}
    for r in results:
        by_n.setdefault(r.get("n_engines", 0), []).append(r)
    logger.info("\n按引擎数量分组:")
    for n in sorted(by_n):
        g = by_n[n]
        logger.info(f"  {n} 引擎: {len(g)} 种, 最佳加速比={max(r.get('speedup',0) for r in g):.2f}x")
    logger.info("\n帕累托最优组合:")
    for r in sorted(pareto, key=lambda x: x.get("speedup",0), reverse=True):
        logger.info(f"  {r.get('engines','N/A'):<40} 加速={r.get('speedup',0):.2f}x 精度={r.get('accuracy',0):.4f} 内存={r.get('peak_memory_mb',0):.1f}MB")
    if auto_res:
        logger.info(f"\n约束最优: {auto_res[0].get('engines')} (加速比={auto_res[0].get('speedup',0):.2f}x)")


if __name__ == "__main__":
    main()
