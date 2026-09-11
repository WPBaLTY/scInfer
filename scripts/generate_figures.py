#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scInfer 论文图表生成脚本
生成 Fig.1 - Fig.7 的所有论文级别图表，符合 Nature Methods 规范。

用法:
    python scripts/generate_figures.py --data-dir results/ --output-dir figures/ --format pdf --dpi 300
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# ============================================================
# 全局样式配置 (Nature Methods 规范)
# ============================================================
NATURE_SINGLE_COL_MM = 86    # 单栏宽度 86mm
NATURE_DOUBLE_COL_MM = 178   # 双栏宽度 178mm
MM_PER_INCH = 25.4

# Nature Methods 推荐字体
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # TrueType 字体嵌入
    "ps.fonttype": 42,
})

# 色盲友好调色板 (基于 viridis/cividis)
CB_COLORS = {
    "scgpt": "#440154",
    "geneformer": "#31688e",
    "scfoundation": "#35b779",
    "uce": "#fde725",
    "scinfer": "#e74c3c",
    "baseline": "#95a5a6",
    "optimized": "#2ecc71",
}
PALETTE = ["#440154", "#31688e", "#35b779", "#fde725", "#e74c3c", "#f39c12"]


def mm_to_inch(mm):
    """毫米转英寸"""
    return mm / MM_PER_INCH


def setup_panel_label(ax, label, x=-0.08, y=1.05):
    """添加面板标签 (a), (b), (c)..."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def save_figure(fig, name, output_dir, fmt="pdf", dpi=350):
    """保存图表到文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {path}")


# ============================================================
# 合成数据生成 (当 results/ 目录无数据时使用)
# ============================================================
def generate_synthetic_data():
    """生成所有图表所需的合成实验数据"""
    np.random.seed(42)
    data = {}

    # --- Fig.1 数据 ---
    models = ["scGPT", "Geneformer", "scFoundation", "UCE"]
    data["fig1"] = {
        "models": models,
        "baseline_speed": np.array([1200, 980, 1500, 800]),       # cells/sec
        "optimized_speed": np.array([4800, 3920, 6000, 3200]),    # cells/sec
        "techniques": ["Quantization", "FlashAttn", "Compile", "Distillation", "Batching", "Mixed Prec"],
        "applicability": np.random.uniform(0.3, 1.0, (4, 6)),
    }

    # --- Fig.2 数据: Profiling ---
    data["fig2"] = {
        "models": models,
        "components": ["Embedding", "Attention", "FFN", "LayerNorm", "Sampling"],
        "time_pct": np.array([
            [15, 35, 40, 5, 5],
            [18, 30, 42, 5, 5],
            [12, 38, 40, 5, 5],
            [20, 28, 42, 5, 5],
        ]),
        "memory_gb": np.array([8.2, 6.5, 10.1, 5.8]),
        "llm_metrics": np.array([4.2, 3.8, 4.5, 3.5, 4.0, 3.9]),  # radar
        "sc_metrics": np.array([2.8, 4.5, 3.2, 1.5, 3.8, 2.1]),
    }

    # --- Fig.3 数据: 量化 ---
    data["fig3"] = {
        "methods": ["FP32", "FP16", "INT8", "INT4 (GPTQ)", "INT4 (AWQ)"],
        "speed": np.array([1.0, 1.8, 2.5, 3.2, 3.0]),
        "cosine_sim": np.array([
            [1.00, 0.99, 0.97, 0.94, 0.95],
            [0.99, 1.00, 0.98, 0.96, 0.96],
            [0.97, 0.98, 1.00, 0.97, 0.97],
            [0.94, 0.96, 0.97, 1.00, 0.98],
            [0.95, 0.96, 0.97, 0.98, 1.00],
        ]),
        "downstream_ari": np.array([0.82, 0.81, 0.79, 0.75, 0.76]),
        "downstream_nmi": np.array([0.78, 0.77, 0.75, 0.71, 0.72]),
        "memory_gb": np.array([16.0, 8.0, 4.0, 2.5, 2.5]),
    }

    # --- Fig.4 数据: FlashAttention ---
    data["fig4"] = {
        "versions": ["PyTorch SDPA", "FlashAttn v1", "FlashAttn v2", "FlashAttn v3"],
        "speedup": np.array([1.0, 1.3, 1.8, 2.1]),
        "seq_lengths": np.array([128, 256, 512, 1024, 2048, 4096]),
        "latency_ms": np.array([
            [2.1, 4.5, 9.8, 21.0, 45.0, 95.0],   # standard
            [1.8, 3.2, 6.5, 13.0, 27.0, 55.0],    # flashv2
        ]),
        "memory_mb": np.array([
            [120, 480, 1920, 7680, 30720, 122880],
            [80, 200, 520, 1400, 4200, 14000],
        ]),
        "model_sizes": ["Small (6L)", "Medium (12L)", "Large (24L)"],
        "small_speedup": np.array([1.2, 1.5, 1.9]),
        "large_speedup": np.array([1.6, 2.2, 3.1]),
    }

    # --- Fig.5 数据: 知识蒸馏 ---
    data["fig5"] = {
        "teacher": "scGPT (47M)",
        "student": "scGPT-distill (12M)",
        "configs": [("47M→12M", 0.78, 2.8), ("47M→6M", 0.72, 3.5),
                    ("47M→3M", 0.65, 4.2), ("24M→6M", 0.74, 3.2)],
        "metrics": ["ARI", "NMI", "F1", "PCC", "Speed"],
        "teacher_scores": np.array([0.85, 0.82, 0.80, 0.88, 1.0]),
        "student_scores": np.array([0.79, 0.76, 0.75, 0.84, 2.8]),
        "direct_scores": np.array([0.68, 0.65, 0.63, 0.72, 2.8]),
        "pareto_speed": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
        "pareto_ari": np.array([0.85, 0.84, 0.82, 0.79, 0.76, 0.70]),
    }

    # --- Fig.6 数据: 组合优化 + 生物学验证 ---
    data["fig6"] = {
        "pareto_x": np.array([1.0, 1.8, 2.5, 3.2, 4.0, 5.2]),
        "pareto_y": np.array([0.85, 0.84, 0.82, 0.79, 0.76, 0.71]),
        "radar_labels": ["ARI", "NMI", "F1", "Speed", "Memory", "PCC"],
        "radar_baseline": np.array([0.85, 0.82, 0.80, 1.0, 1.0, 0.88]),
        "radar_optimized": np.array([0.79, 0.76, 0.75, 4.2, 5.5, 0.84]),
        "bio_tasks": ["Cell Type", "Gene Perturbation", "Trajectory", "Drug Response"],
        "bio_pcc": np.array([0.92, 0.87, 0.85, 0.81]),
    }

    # --- Fig.7 数据: 实际应用 ---
    data["fig7"] = {
        "cell_counts": np.array([10000, 50000, 100000, 500000, 1000000]),
        "baseline_time": np.array([5, 25, 50, 250, 500]),
        "optimized_time": np.array([1.5, 7, 14, 65, 125]),
        "memory_curve_gb": np.array([2.1, 4.5, 8.0, 16.0, 28.0]),
        "optimized_memory_gb": np.array([0.8, 1.5, 2.5, 5.0, 8.5]),
        "gpus": ["T4", "V100", "A100", "H100", "RTX 4090"],
        "gpu_speedup": np.array([1.0, 2.1, 3.8, 5.2, 3.2]),
        "gpu_memory": np.array([16, 32, 80, 80, 24]),
    }

    return data


def load_or_synthesize(data_dir):
    """尝试从 results/ 加载数据，否则使用合成数据"""
    data_path = Path(data_dir)
    if data_path.exists() and (data_path / "synthetic_data.npz").exists():
        try:
            loaded = dict(np.load(data_path / "synthetic_data.npz", allow_pickle=True))
            print(f"[INFO] 从 {data_dir} 加载实验数据")
            return loaded
        except Exception:
            pass
    print("[INFO] 使用合成数据生成示例图表")
    return generate_synthetic_data()


# ============================================================
# Fig.1: 模型推理速度对比 + 技术适用性 + 框架示意
# ============================================================
def generate_fig1(data, output_dir, fmt, dpi):
    """Fig.1: (a)推理速度对比 (b)技术适用性热力图 (c)框架概览"""
    print("\n[Fig.1] 生成概览图...")
    d = data["fig1"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.45))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1], wspace=0.4)

    # (a) 推理速度对比柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d["models"]))
    bar_w = 0.35
    bars1 = ax1.bar(x - bar_w/2, d["baseline_speed"], bar_w, color="#95a5a6", label="Baseline", edgecolor="none")
    bars2 = ax1.bar(x + bar_w/2, d["optimized_speed"], bar_w, color="#e74c3c", label="scInfer", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["models"], rotation=30, ha="right", fontsize=5.5)
    ax1.set_ylabel("Throughput (cells/sec)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0, 7000)
    setup_panel_label(ax1, "(a)")

    # (b) LLM优化技术适用性热力图
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(d["applicability"], cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(d["techniques"])))
    ax2.set_xticklabels(d["techniques"], rotation=45, ha="right", fontsize=5)
    ax2.set_yticks(range(len(d["models"])))
    ax2.set_yticklabels(d["models"], fontsize=5.5)
    for i in range(len(d["models"])):
        for j in range(len(d["techniques"])):
            val = d["applicability"][i, j]
            color = "white" if val < 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Applicability Score")
    setup_panel_label(ax2, "(b)")

    # (c) 框架概览示意图 (简化)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    boxes = [
        (1, 8.5, "API Layer", "#3498db"),
        (1, 6.5, "Optimization\nCombiner", "#e74c3c"),
        (0.2, 4.5, "Quant.", "#e67e22"),
        (3.5, 4.5, "Flash\nAttn", "#2ecc71"),
        (6.8, 4.5, "Compile", "#9b59b6"),
        (1, 2.5, "Model Adapters\nscGPT · Geneformer · scFoundation · UCE", "#31688e"),
        (1, 0.5, "Evaluation\nProfiler · Benchmark · Metrics", "#95a5a6"),
    ]
    for (bx, by, txt, clr) in boxes:
        bw = 8.5 if by in [2.5, 0.5] else (2.8 if by == 4.5 else 8.5)
        rect = FancyBboxPatch((bx, by), bw, 1.2, boxstyle="round,pad=0.1",
                              facecolor=clr, edgecolor="white", alpha=0.85)
        ax3.add_patch(rect)
        ax3.text(bx + bw/2, by + 0.6, txt, ha="center", va="center",
                 fontsize=5.5, color="white", fontweight="bold")
    # 连接箭头
    for y_start, y_end in [(8.5, 7.7), (6.5, 5.7), (4.5, 3.7), (2.5, 1.7)]:
        ax3.annotate("", xy=(5, y_end), xytext=(5, y_start),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    setup_panel_label(ax3, "(c)", x=-0.05, y=1.02)

    save_figure(fig, "fig1_overview", output_dir, fmt, dpi)


# ============================================================
# Fig.2: 推理瓶颈 Profiling
# ============================================================
def generate_fig2(data, output_dir, fmt, dpi):
    """Fig.2: (a)时间分解堆叠柱状图 (b)内存分解 (c)与LLM对比雷达图"""
    print("\n[Fig.2] 生成推理瓶颈 Profiling 图...")
    d = data["fig2"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.42))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.2, 0.8, 1], wspace=0.4)
    colors_stack = ["#440154", "#31688e", "#35b779", "#fde725", "#f39c12"]

    # (a) 时间分解堆叠柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d["models"]))
    bottom = np.zeros(len(d["models"]))
    for i, comp in enumerate(d["components"]):
        ax1.bar(x, d["time_pct"][:, i], bottom=bottom, label=comp,
                color=colors_stack[i], edgecolor="white", linewidth=0.3)
        bottom += d["time_pct"][:, i]
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["models"], rotation=30, ha="right", fontsize=5.5)
    ax1.set_ylabel("Time Fraction (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)
    setup_panel_label(ax1, "(a)")

    # (b) 内存分解
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(d["models"], d["memory_gb"], color=PALETTE[:4], edgecolor="none")
    ax2.set_xlabel("GPU Memory (GB)")
    for bar, val in zip(bars, d["memory_gb"]):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=5.5)
    setup_panel_label(ax2, "(b)")

    # (c) 与LLM对比雷达图
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    categories = ["Embed\nOverhead", "Attention\nCost", "FFN\nCost", "Memory\nEfficiency",
                  "Batch\nScaling", "Seq Len\nSensitivity"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    llm_vals = d["llm_metrics"].tolist() + [d["llm_metrics"][0]]
    sc_vals = d["sc_metrics"].tolist() + [d["sc_metrics"][0]]

    ax3.plot(angles, llm_vals, "o-", color="#31688e", linewidth=1.2, markersize=3, label="LLM (avg)")
    ax3.fill(angles, llm_vals, color="#31688e", alpha=0.15)
    ax3.plot(angles, sc_vals, "o-", color="#e74c3c", linewidth=1.2, markersize=3, label="scFoundation (avg)")
    ax3.fill(angles, sc_vals, color="#e74c3c", alpha=0.15)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=5)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=5)
    ax3.set_ylim(0, 5)
    setup_panel_label(ax3, "(c)", x=-0.1, y=1.15)

    save_figure(fig, "fig2_profiling", output_dir, fmt, dpi)


# ============================================================
# Fig.3: 量化系统评估
# ============================================================
def generate_fig3(data, output_dir, fmt, dpi):
    """Fig.3: (a)推理速度 (b)余弦相似度热图 (c)下游任务 (d)内存占用"""
    print("\n[Fig.3] 生成量化系统评估图...")
    d = data["fig3"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 推理速度对比
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(d["methods"], d["speed"], color=PALETTE[:5], edgecolor="none")
    ax1.set_ylabel("Speedup Factor (vs FP32)")
    ax1.set_xticklabels(d["methods"], rotation=35, ha="right", fontsize=5)
    for bar, val in zip(bars, d["speed"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.1f}x", ha="center", va="bottom", fontsize=5.5)
    ax1.set_ylim(0, 4.0)
    setup_panel_label(ax1, "(a)")

    # (b) Embedding 余弦相似度热图
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(d["cosine_sim"], cmap="cividis", vmin=0.9, vmax=1.0)
    ax2.set_xticks(range(len(d["methods"])))
    ax2.set_yticks(range(len(d["methods"])))
    ax2.set_xticklabels(d["methods"], rotation=45, ha="right", fontsize=4.5)
    ax2.set_yticklabels(d["methods"], fontsize=4.5)
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f"{d['cosine_sim'][i,j]:.3f}", ha="center", va="center",
                     fontsize=5, color="white" if d["cosine_sim"][i,j] < 0.95 else "black")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    setup_panel_label(ax2, "(b)")

    # (c) 下游任务影响
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(d["methods"]))
    ax3.bar(x - 0.18, d["downstream_ari"], 0.35, label="ARI", color="#31688e", edgecolor="none")
    ax3.bar(x + 0.18, d["downstream_nmi"], 0.35, label="NMI", color="#35b779", edgecolor="none")
    ax3.set_xticks(x)
    ax3.set_xticklabels(d["methods"], rotation=35, ha="right", fontsize=5)
    ax3.set_ylabel("Score")
    ax3.legend(frameon=False)
    ax3.set_ylim(0, 1.0)
    setup_panel_label(ax3, "(c)")

    # (d) 内存占用
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(range(len(d["methods"])), d["memory_gb"], alpha=0.3, color="#e74c3c")
    ax4.plot(range(len(d["methods"])), d["memory_gb"], "o-", color="#e74c3c", markersize=4, linewidth=1.2)
    ax4.set_xticks(range(len(d["methods"])))
    ax4.set_xticklabels(d["methods"], rotation=35, ha="right", fontsize=5)
    ax4.set_ylabel("GPU Memory (GB)")
    for i, v in enumerate(d["memory_gb"]):
        ax4.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=5.5)
    ax4.set_ylim(0, 20)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "fig3_quantization", output_dir, fmt, dpi)


# ============================================================
# Fig.4: FlashAttention 效果
# ============================================================
def generate_fig4(data, output_dir, fmt, dpi):
    """Fig.4: (a)版本对比 (b)序列长度影响 (c)内存影响 (d)小vs大模型"""
    print("\n[Fig.4] 生成 FlashAttention 效果图...")
    d = data["fig4"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 版本对比
    ax1 = fig.add_subplot(gs[0, 0])
    colors_v = ["#95a5a6", "#31688e", "#35b779", "#440154"]
    bars = ax1.bar(d["versions"], d["speedup"], color=colors_v, edgecolor="none")
    ax1.set_ylabel("Speedup Factor")
    ax1.set_xticklabels(d["versions"], rotation=30, ha="right", fontsize=5)
    for bar, val in zip(bars, d["speedup"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                 f"{val:.1f}x", ha="center", fontsize=5.5)
    ax1.set_ylim(0, 2.5)
    setup_panel_label(ax1, "(a)")

    # (b) 序列长度影响
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(d["seq_lengths"], d["latency_ms"][0], "o-", color="#95a5a6",
                 label="Standard Attention", markersize=3, linewidth=1.2)
    ax2.semilogy(d["seq_lengths"], d["latency_ms"][1], "s-", color="#e74c3c",
                 label="FlashAttention v2", markersize=3, linewidth=1.2)
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Latency (ms)")
    ax2.legend(frameon=False, fontsize=5)
    ax2.set_xticks(d["seq_lengths"])
    ax2.set_xticklabels([str(s) for s in d["seq_lengths"]], rotation=30, fontsize=5)
    setup_panel_label(ax2, "(b)")

    # (c) 内存影响
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(d["seq_lengths"], d["memory_mb"][0], "o-", color="#95a5a6",
               label="Standard", markersize=3, linewidth=1.2)
    ax3.loglog(d["seq_lengths"], d["memory_mb"][1], "s-", color="#2ecc71",
               label="FlashAttention v2", markersize=3, linewidth=1.2)
    ax3.set_xlabel("Sequence Length")
    ax3.set_ylabel("Memory (MB)")
    ax3.legend(frameon=False, fontsize=5)
    ax3.set_xticks(d["seq_lengths"])
    ax3.set_xticklabels([str(s) for s in d["seq_lengths"]], rotation=30, fontsize=5)
    setup_panel_label(ax3, "(c)")

    # (d) 小模型 vs 大模型
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(d["model_sizes"]))
    ax4.bar(x - 0.18, d["small_speedup"], 0.35, label="Small Models", color="#31688e")
    ax4.bar(x + 0.18, d["large_speedup"], 0.35, label="Large Models", color="#e74c3c")
    ax4.set_xticks(x)
    ax4.set_xticklabels(d["model_sizes"], fontsize=5)
    ax4.set_ylabel("Speedup Factor")
    ax4.legend(frameon=False, fontsize=5)
    ax4.set_ylim(0, 4.0)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "fig4_flash_attention", output_dir, fmt, dpi)


# ============================================================
# Fig.5: 知识蒸馏
# ============================================================
def generate_fig5(data, output_dir, fmt, dpi):
    """Fig.5: (a)Teacher-Student配置 (b)性能对比 (c)蒸馏vs直接训练 (d)帕累托曲线 (e)UMAP"""
    print("\n[Fig.5] 生成知识蒸馏图...")
    d = data["fig5"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.55))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5,
                           width_ratios=[1, 1, 1])

    # (a) Teacher-Student 配置示意图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    # Teacher
    t_rect = FancyBboxPatch((1, 6), 8, 3, boxstyle="round,pad=0.2",
                            facecolor="#440154", edgecolor="white", alpha=0.85)
    ax1.add_patch(t_rect)
    ax1.text(5, 7.5, "Teacher: scGPT (47M params)\n12 layers, d=512", ha="center", va="center",
             fontsize=5.5, color="white", fontweight="bold")
    # Student
    s_rect = FancyBboxPatch((2, 1), 6, 3, boxstyle="round,pad=0.2",
                            facecolor="#e74c3c", edgecolor="white", alpha=0.85)
    ax1.add_patch(s_rect)
    ax1.text(5, 2.5, "Student: scGPT-distill (12M)\n6 layers, d=256", ha="center", va="center",
             fontsize=5.5, color="white", fontweight="bold")
    # Arrow
    ax1.annotate("", xy=(5, 4.2), xytext=(5, 5.8),
                 arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.5))
    ax1.text(5.5, 5.0, "Knowledge\nDistillation", fontsize=5, color="#f39c12",
             ha="left", va="center", fontstyle="italic")
    setup_panel_label(ax1, "(a)", x=-0.05, y=1.02)

    # (b) 性能对比
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(d["metrics"]))
    ax2.bar(x - 0.25, d["teacher_scores"], 0.25, label="Teacher", color="#440154")
    ax2.bar(x, d["student_scores"], 0.25, label="Student (distill)", color="#e74c3c")
    ax2.bar(x + 0.25, d["direct_scores"], 0.25, label="Student (direct)", color="#95a5a6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["metrics"], fontsize=5.5)
    ax2.set_ylabel("Normalized Score")
    ax2.legend(frameon=False, fontsize=4.5)
    ax2.set_ylim(0, 3.5)
    setup_panel_label(ax2, "(b)")

    # (c) 蒸馏 vs 直接训练
    ax3 = fig.add_subplot(gs[0, 2])
    methods = ["Distillation", "Direct Train"]
    vals_ari = [0.79, 0.68]
    vals_nmi = [0.76, 0.65]
    x = np.arange(len(methods))
    ax3.bar(x - 0.18, vals_ari, 0.35, label="ARI", color="#31688e")
    ax3.bar(x + 0.18, vals_nmi, 0.35, label="NMI", color="#35b779")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=5.5)
    ax3.set_ylabel("Score")
    ax3.legend(frameon=False)
    ax3.set_ylim(0, 1.0)
    setup_panel_label(ax3, "(c)")

    # (d) 帕累托曲线
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(d["pareto_speed"], d["pareto_ari"], "o-", color="#e74c3c",
             markersize=4, linewidth=1.2, label="Distillation variants")
    ax4.fill_between(d["pareto_speed"], d["pareto_ari"], alpha=0.1, color="#e74c3c")
    ax4.set_xlabel("Speedup Factor")
    ax4.set_ylabel("ARI Score")
    ax4.legend(frameon=False, fontsize=5)
    setup_panel_label(ax4, "(d)")

    # (e) UMAP 可视化 (合成)
    ax5 = fig.add_subplot(gs[1, 1])
    np.random.seed(123)
    n_pts = 200
    # Teacher embeddings
    t_x = np.random.randn(n_pts) * 0.8 + np.array([np.sin(i*0.1)*2 for i in range(n_pts)])
    t_y = np.random.randn(n_pts) * 0.8 + np.array([np.cos(i*0.1)*2 for i in range(n_pts)])
    labels = np.array([i % 5 for i in range(n_pts)])
    cmap = plt.cm.viridis
    sc = ax5.scatter(t_x, t_y, c=labels, cmap=cmap, s=5, alpha=0.6, label="Teacher")
    # Student embeddings (shifted)
    s_x = t_x + np.random.randn(n_pts) * 0.3
    s_y = t_y + np.random.randn(n_pts) * 0.3
    ax5.scatter(s_x, s_y, c=labels, cmap=cmap, s=5, alpha=0.6, marker="x", label="Student")
    ax5.legend(frameon=False, fontsize=5, markerscale=2)
    ax5.set_xlabel("UMAP-1", fontsize=5.5)
    ax5.set_ylabel("UMAP-2", fontsize=5.5)
    ax5.tick_params(labelsize=5)
    setup_panel_label(ax5, "(e)")

    save_figure(fig, "fig5_distillation", output_dir, fmt, dpi)


# ============================================================
# Fig.6: 组合优化 + 生物学验证
# ============================================================
def generate_fig6(data, output_dir, fmt, dpi):
    """Fig.6: (a)帕累托前沿 (b)雷达图 (c)UMAP对比 (d)扰动预测"""
    print("\n[Fig.6] 生成组合优化与生物学验证图...")
    d = data["fig6"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 帕累托前沿
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(d["pareto_x"], d["pareto_y"], color="#e74c3c", s=20, zorder=5, edgecolors="white", linewidths=0.5)
    ax1.plot(d["pareto_x"], d["pareto_y"], "--", color="#e74c3c", alpha=0.5, linewidth=0.8)
    # 标注各配置点
    labels = ["Q", "Q+A", "Q+A+C", "Q+A+C+D", "Full", "Full+B"]
    for i, lbl in enumerate(labels):
        ax1.annotate(lbl, (d["pareto_x"][i], d["pareto_y"][i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=4.5)
    ax1.fill_between(d["pareto_x"], d["pareto_y"], alpha=0.08, color="#e74c3c")
    ax1.set_xlabel("Speedup Factor")
    ax1.set_ylabel("Biological Fidelity (ARI)")
    ax1.set_ylim(0.65, 0.90)
    setup_panel_label(ax1, "(a)")

    # (b) 雷达图
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    N = len(d["radar_labels"])
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    base_vals = d["radar_baseline"].tolist() + [d["radar_baseline"][0]]
    opt_vals = d["radar_optimized"].tolist() + [d["radar_optimized"][0]]
    # 归一化到0-1
    max_val = max(max(base_vals), max(opt_vals))
    base_norm = [v / max_val for v in base_vals]
    opt_norm = [v / max_val for v in opt_vals]
    ax2.plot(angles, base_norm, "o-", color="#95a5a6", linewidth=1.2, markersize=3, label="Baseline")
    ax2.fill(angles, base_norm, color="#95a5a6", alpha=0.1)
    ax2.plot(angles, opt_norm, "o-", color="#e74c3c", linewidth=1.2, markersize=3, label="scInfer")
    ax2.fill(angles, opt_norm, color="#e74c3c", alpha=0.1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(d["radar_labels"], fontsize=5)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=5)
    setup_panel_label(ax2, "(b)", x=-0.1, y=1.15)

    # (c) UMAP 对比 (合成)
    ax3 = fig.add_subplot(gs[1, 0])
    np.random.seed(456)
    n = 300
    clusters = 6
    for c in range(clusters):
        cx, cy = np.cos(c * np.pi / 3) * 3, np.sin(c * np.pi / 3) * 3
        ax3.scatter(np.random.randn(n//clusters) * 0.5 + cx,
                    np.random.randn(n//clusters) * 0.5 + cy,
                    c=plt.cm.cividis(c / clusters), s=4, alpha=0.6)
    ax3.set_xlabel("UMAP-1", fontsize=5.5)
    ax3.set_ylabel("UMAP-2", fontsize=5.5)
    ax3.tick_params(labelsize=5)
    ax3.text(0.02, 0.98, "Combined\nOptimization", transform=ax3.transAxes,
             fontsize=5.5, va="top", fontweight="bold")
    setup_panel_label(ax3, "(c)")

    # (d) 扰动预测
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.barh(d["bio_tasks"], d["bio_pcc"], color=["#440154", "#31688e", "#35b779", "#fde725"],
                    edgecolor="none")
    ax4.set_xlabel("Pearson Correlation (PCC)")
    ax4.set_xlim(0, 1.0)
    for bar, val in zip(bars, d["bio_pcc"]):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=5.5)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "fig6_combined_bio", output_dir, fmt, dpi)


# ============================================================
# Fig.7: 实际应用
# ============================================================
def generate_fig7(data, output_dir, fmt, dpi):
    """Fig.7: (a)百万细胞推理时间 (b)内存曲线 (c)端到端工作流 (d)GPU配置对比"""
    print("\n[Fig.7] 生成实际应用图...")
    d = data["fig7"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 百万细胞推理时间
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d["cell_counts"] / 1e6, d["baseline_time"], "o-", color="#95a5a6",
             label="Baseline", markersize=4, linewidth=1.2)
    ax1.plot(d["cell_counts"] / 1e6, d["optimized_time"], "s-", color="#e74c3c",
             label="scInfer", markersize=4, linewidth=1.2)
    ax1.fill_between(d["cell_counts"]/1e6, d["optimized_time"], d["baseline_time"],
                     alpha=0.1, color="#e74c3c")
    ax1.set_xlabel("Number of Cells (Millions)")
    ax1.set_ylabel("Inference Time (min)")
    ax1.legend(frameon=False)
    ax1.set_xticks([0, 0.2, 0.5, 1.0])
    setup_panel_label(ax1, "(a)")

    # (b) 内存曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(d["cell_counts"]/1e6, d["optimized_memory_gb"], d["memory_curve_gb"],
                     alpha=0.15, color="#31688e")
    ax2.plot(d["cell_counts"]/1e6, d["memory_curve_gb"], "o-", color="#31688e",
             label="Baseline", markersize=4, linewidth=1.2)
    ax2.plot(d["cell_counts"]/1e6, d["optimized_memory_gb"], "s-", color="#2ecc71",
             label="scInfer", markersize=4, linewidth=1.2)
    ax2.set_xlabel("Number of Cells (Millions)")
    ax2.set_ylabel("GPU Memory (GB)")
    ax2.legend(frameon=False)
    # 80GB 线
    ax2.axhline(y=80, color="#e74c3c", linestyle="--", linewidth=0.6, alpha=0.7, label="A100 80GB limit")
    ax2.set_xticks([0, 0.2, 0.5, 1.0])
    setup_panel_label(ax2, "(b)")

    # (c) 端到端工作流示意
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0, 8)
    ax3.axis("off")
    steps = [
        (0.5, 6, "Load\nData", "#3498db"),
        (3, 6, "Model\nSelection", "#9b59b6"),
        (5.5, 6, "Optimization\nPipeline", "#e74c3c"),
        (8, 6, "Inference", "#e67e22"),
        (10.5, 6, "Analysis", "#2ecc71"),
        (0.5, 2, "Scanpy\nAnnData", "#3498db"),
        (3, 2, "scGPT/GF/\nsf/UCE", "#9b59b6"),
        (5.5, 2, "Quant+FA+\nCompile+Dist", "#e74c3c"),
        (8, 2, "GPU\nAccelerated", "#e67e22"),
        (10.5, 2, "Clustering\nDEG/Trajectory", "#2ecc71"),
    ]
    for (sx, sy, txt, clr) in steps:
        rect = FancyBboxPatch((sx, sy), 2, 1.3, boxstyle="round,pad=0.1",
                              facecolor=clr, edgecolor="white", alpha=0.85)
        ax3.add_patch(rect)
        ax3.text(sx + 1, sy + 0.65, txt, ha="center", va="center",
                 fontsize=4.5, color="white", fontweight="bold")
    # 箭头
    for sx in [2.5, 5.0, 7.5, 10.0]:
        ax3.annotate("", xy=(sx + 0.5, 6.65), xytext=(sx, 6.65),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=0.6))
        ax3.annotate("", xy=(sx + 0.5, 2.65), xytext=(sx, 2.65),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=0.6))
    # 上下连接
    for sx in [1.5, 4.0, 6.5, 9.0, 11.5]:
        ax3.annotate("", xy=(sx, 5.8), xytext=(sx, 3.5),
                     arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4, linestyle="dotted"))
    setup_panel_label(ax3, "(c)", x=-0.02, y=1.02)

    # (d) GPU 配置对比
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(d["gpus"]))
    bars = ax4.bar(x, d["gpu_speedup"], color=PALETTE[:5], edgecolor="none")
    ax4.set_xticks(x)
    ax4.set_xticklabels(d["gpus"], fontsize=5.5)
    ax4.set_ylabel("Relative Speedup")
    # 在柱子上标注显存
    for bar, val, mem in zip(bars, d["gpu_speedup"], d["gpu_memory"]):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f"{val:.1f}x\n({mem}GB)", ha="center", va="bottom", fontsize=4.5)
    ax4.set_ylim(0, 6.5)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "fig7_application", output_dir, fmt, dpi)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="scInfer 论文图表生成 (Fig.1-7)")
    parser.add_argument("--data-dir", default="results/", help="实验数据目录 (默认: results/)")
    parser.add_argument("--output-dir", default="figures/", help="图表输出目录 (默认: figures/)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "eps", "tiff", "png"],
                        help="输出格式 (默认: pdf)")
    parser.add_argument("--dpi", type=int, default=350, help="输出分辨率 (默认: 350)")
    parser.add_argument("--figs", nargs="*", default=None,
                        help="指定生成哪些图 (如 --figs 1 3 5)，默认全部")
    args = parser.parse_args()

    print("=" * 60)
    print("  scInfer 论文图表生成器")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  格式: {args.format} | DPI: {args.dpi}")
    print("=" * 60)

    # 加载或生成数据
    data = load_or_synthesize(args.data_dir)

    # 图表生成函数映射
    fig_funcs = {
        1: generate_fig1,
        2: generate_fig2,
        3: generate_fig3,
        4: generate_fig4,
        5: generate_fig5,
        6: generate_fig6,
        7: generate_fig7,
    }

    # 生成指定图表
    figs_to_gen = args.figs if args.figs else list(range(1, 8))
    for fig_num in figs_to_gen:
        if fig_num in fig_funcs:
            fig_funcs[fig_num](data, args.output_dir, args.format, args.dpi)
        else:
            print(f"[WARN] 未知图表编号: Fig.{fig_num}")

    print("\n" + "=" * 60)
    print("  所有图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
