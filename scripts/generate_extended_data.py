#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scInfer Extended Data Figures 生成脚本
生成 Extended Data Fig.1-8 和 Extended Data Table 1-2。

用法:
    python scripts/generate_extended_data.py --data-dir results/ --output-dir figures/extended/ --format pdf
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import csv

# ============================================================
# 全局样式 (与 generate_figures.py 一致)
# ============================================================
NATURE_DOUBLE_COL_MM = 178
MM_PER_INCH = 25.4

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = ["#440154", "#31688e", "#35b779", "#fde725", "#e74c3c", "#f39c12"]


def mm_to_inch(mm):
    return mm / MM_PER_INCH


def setup_panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def save_figure(fig, name, output_dir, fmt="pdf", dpi=350):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {path}")


# ============================================================
# 合成数据
# ============================================================
def generate_synthetic_extended_data():
    """生成 Extended Data Figures 所需的合成数据"""
    np.random.seed(2024)
    data = {}

    # ED Fig.1: Profiling 详细设置
    data["ed1"] = {
        "gpu_configs": ["T4 16GB", "V100 32GB", "A100 80GB", "H100 80GB"],
        "batch_sizes": [1, 4, 16, 64, 256],
        "latency_ms": np.random.uniform(5, 200, (4, 5)),
        "throughput": np.random.uniform(100, 5000, (4, 5)),
        "memory_used": np.random.uniform(1, 70, (4, 5)),
        "profiling_phases": ["Data Loading", "Tokenization", "Forward Pass", "Post-processing"],
        "phase_times": np.array([5, 8, 75, 12]),  # 百分比
    }

    # ED Fig.2: 量化方法对比
    data["ed2"] = {
        "methods": ["FP32", "FP16", "INT8 (RTN)", "INT8 (GPTQ)", "INT8 (AWQ)",
                    "INT4 (RTN)", "INT4 (GPTQ)", "INT4 (AWQ)", "GGUF Q4", "GGUF Q8"],
        "speed": np.array([1.0, 1.8, 2.2, 2.5, 2.4, 3.0, 3.2, 3.0, 2.8, 2.1]),
        "ari": np.array([0.85, 0.84, 0.82, 0.81, 0.82, 0.75, 0.74, 0.76, 0.73, 0.80]),
        "nmi": np.array([0.82, 0.81, 0.79, 0.78, 0.79, 0.71, 0.70, 0.72, 0.69, 0.77]),
        "memory_gb": np.array([16.0, 8.0, 4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 2.8, 4.5]),
        "cosine_sim": np.array([1.0, 0.99, 0.97, 0.97, 0.98, 0.94, 0.94, 0.95, 0.93, 0.97]),
    }

    # ED Fig.3: 量化对不同基因集的影响
    data["ed3"] = {
        "gene_sets": ["All Genes", "HVG (2000)", "Marker Genes", "TFs", "Pathway Genes",
                      "Ribosomal", "Mitochondrial", "Cell Cycle"],
        "fp16_cosine": np.array([0.99, 0.99, 0.98, 0.99, 0.98, 0.99, 0.97, 0.98]),
        "int8_cosine": np.array([0.97, 0.97, 0.96, 0.97, 0.96, 0.98, 0.94, 0.96]),
        "int4_cosine": np.array([0.94, 0.94, 0.92, 0.93, 0.92, 0.95, 0.89, 0.91]),
    }

    # ED Fig.4: FlashAttention 不同 GPU 架构
    data["ed4"] = {
        "gpus": ["T4", "V100", "A100", "H100", "RTX 4090"],
        "compute_cap": ["7.5", "7.0", "8.0", "9.0", "8.9"],
        "standard_ms": np.array([12.5, 8.2, 4.1, 2.8, 6.5]),
        "flash_ms": np.array([8.5, 5.0, 2.2, 1.3, 4.0]),
        "speedup": np.array([1.47, 1.64, 1.86, 2.15, 1.63]),
        "memory_reduction": np.array([0.55, 0.58, 0.62, 0.68, 0.56]),
    }

    # ED Fig.5: 蒸馏训练收敛曲线
    data["ed5"] = {
        "epochs": np.arange(1, 101),
        "train_loss": np.exp(-np.arange(1, 101) * 0.05) * 2.5 + np.random.randn(100) * 0.02,
        "val_loss": np.exp(-np.arange(1, 101) * 0.05) * 2.8 + np.random.randn(100) * 0.05,
        "train_ari": 1 - np.exp(-np.arange(1, 101) * 0.04) * 0.4 + np.random.randn(100) * 0.01,
        "val_ari": 1 - np.exp(-np.arange(1, 101) * 0.04) * 0.45 + np.random.randn(100) * 0.02,
    }

    # ED Fig.6: 不同蒸馏损失函数对比
    data["ed6"] = {
        "loss_types": ["MSE", "KL Divergence", "Cosine Loss", "Attention Transfer", "Combined"],
        "ari": np.array([0.72, 0.76, 0.74, 0.78, 0.81]),
        "nmi": np.array([0.69, 0.73, 0.71, 0.75, 0.78]),
        "cosine_sim": np.array([0.93, 0.95, 0.96, 0.94, 0.97]),
        "speed": np.array([3.5, 3.2, 3.3, 3.0, 2.8]),
    }

    # ED Fig.7: 组合优化超参数敏感性
    data["ed7"] = {
        "quant_bits": [4, 8, 16, 32],
        "attention_type": ["Standard", "Flash v2"],
        "compile_mode": ["default", "reduce-overhead", "max-autotune"],
        "sensitivity": np.random.uniform(0.6, 0.95, (4, 2, 3)),
        "pareto_configs": ["Q4+FA+CO", "Q8+FA+RO", "Q16+FA+MO", "Q4+SDPA+CO",
                           "Q8+SDPA+RO", "Q4+FA+MO", "Q8+FA+CO", "Q16+SDPA+MO"],
        "pareto_speed": np.array([3.2, 2.5, 1.8, 2.8, 2.0, 3.0, 2.7, 1.9]),
        "pareto_ari": np.array([0.74, 0.79, 0.83, 0.72, 0.78, 0.76, 0.80, 0.82]),
    }

    # ED Fig.8: 额外生物学任务验证
    data["ed8"] = {
        "tasks": ["Cell Type Annotation", "Gene Perturbation", "Trajectory Inference",
                  "Drug Response", "Cell-Cell Interaction", "Spatial Deconvolution"],
        "baseline": np.array([0.85, 0.78, 0.82, 0.75, 0.70, 0.73]),
        "scinfer": np.array([0.82, 0.75, 0.80, 0.72, 0.68, 0.71]),
        "p_values": np.array([0.032, 0.045, 0.078, 0.091, 0.12, 0.15]),
    }

    # ED Table 1: 详细推理 Benchmark
    data["ed_table1"] = {
        "headers": ["Model", "Params", "FP32 (cells/s)", "FP16 (cells/s)", "INT8 (cells/s)",
                    "INT4 (cells/s)", "Memory FP32 (GB)", "Memory INT4 (GB)"],
        "rows": [
            ["scGPT", "47M", "1200", "2160", "3000", "3840", "16.0", "2.5"],
            ["Geneformer", "29M", "980", "1764", "2450", "3136", "10.5", "1.8"],
            ["scFoundation", "96M", "1500", "2700", "3750", "4800", "22.0", "3.5"],
            ["UCE", "15M", "800", "1440", "2000", "2560", "6.5", "1.2"],
        ]
    }

    # ED Table 2: 生物学 Benchmark 统计检验
    data["ed_table2"] = {
        "headers": ["Task", "Metric", "Baseline", "scInfer", "Δ", "p-value", "Significant"],
        "rows": [
            ["Cell Type (PBMC)", "ARI", "0.85", "0.82", "-0.03", "0.032", "Yes"],
            ["Cell Type (Pancreas)", "ARI", "0.81", "0.78", "-0.03", "0.041", "Yes"],
            ["Gene Perturbation", "PCC", "0.87", "0.85", "-0.02", "0.045", "Yes"],
            ["Trajectory (Monocle)", "Correlation", "0.91", "0.89", "-0.02", "0.078", "No"],
            ["Drug Response (GDSC)", "PCC", "0.75", "0.72", "-0.03", "0.091", "No"],
            ["Cell-Cell Interaction", "F1", "0.70", "0.68", "-0.02", "0.12", "No"],
            ["Spatial Deconvolution", "RMSE", "0.15", "0.16", "+0.01", "0.15", "No"],
        ]
    }

    return data


def load_or_synthesize(data_dir):
    """尝试加载数据，否则使用合成数据"""
    data_path = Path(data_dir)
    if data_path.exists() and (data_path / "extended_data.npz").exists():
        try:
            loaded = dict(np.load(data_path / "extended_data.npz", allow_pickle=True))
            print(f"[INFO] 从 {data_dir} 加载 Extended Data")
            return loaded
        except Exception:
            pass
    print("[INFO] 使用合成数据生成 Extended Data Figures")
    return generate_synthetic_extended_data()


# ============================================================
# ED Fig.1: Profiling 详细设置
# ============================================================
def generate_ed_fig1(data, output_dir, fmt, dpi):
    """ED Fig.1: 推理 Profiling 详细设置和原始数据"""
    print("\n[ED Fig.1] 推理 Profiling 详细数据...")
    d = data["ed1"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 不同GPU上不同batch size的延迟
    ax1 = fig.add_subplot(gs[0, 0])
    for i, gpu in enumerate(d["gpu_configs"]):
        ax1.plot(d["batch_sizes"], d["latency_ms"][i], "o-", color=PALETTE[i],
                 label=gpu, markersize=3, linewidth=1)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(d["batch_sizes"])
    ax1.set_xticklabels([str(b) for b in d["batch_sizes"]])
    ax1.legend(frameon=False, fontsize=5)
    setup_panel_label(ax1, "(a)")

    # (b) 吞吐量
    ax2 = fig.add_subplot(gs[0, 1])
    for i, gpu in enumerate(d["gpu_configs"]):
        ax2.plot(d["batch_sizes"], d["throughput"][i], "s-", color=PALETTE[i],
                 label=gpu, markersize=3, linewidth=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (cells/sec)")
    ax2.set_xticks(d["batch_sizes"])
    ax2.set_xticklabels([str(b) for b in d["batch_sizes"]])
    ax2.legend(frameon=False, fontsize=5)
    setup_panel_label(ax2, "(b)")

    # (c) 内存使用
    ax3 = fig.add_subplot(gs[1, 0])
    for i, gpu in enumerate(d["gpu_configs"]):
        ax3.plot(d["batch_sizes"], d["memory_used"][i], "^-", color=PALETTE[i],
                 label=gpu, markersize=3, linewidth=1)
    ax3.set_xscale("log", base=2)
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("GPU Memory (GB)")
    ax3.set_xticks(d["batch_sizes"])
    ax3.set_xticklabels([str(b) for b in d["batch_sizes"]])
    ax3.legend(frameon=False, fontsize=5)
    setup_panel_label(ax3, "(c)")

    # (d) Profiling 阶段分解
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ["#3498db", "#9b59b6", "#e74c3c", "#2ecc71"]
    ax4.bar(d["profiling_phases"], d["phase_times"], color=colors, edgecolor="none")
    ax4.set_ylabel("Time Fraction (%)")
    ax4.set_xticklabels(d["profiling_phases"], rotation=30, ha="right", fontsize=5)
    for i, v in enumerate(d["phase_times"]):
        ax4.text(i, v + 1, f"{v}%", ha="center", fontsize=5.5)
    ax4.set_ylim(0, 90)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "ed_fig1_profiling", output_dir, fmt, dpi)


# ============================================================
# ED Fig.2: 量化方法详细对比
# ============================================================
def generate_ed_fig2(data, output_dir, fmt, dpi):
    """ED Fig.2: 不同量化方法 (GPTQ vs AWQ vs GGUF) 详细对比"""
    print("\n[ED Fig.2] 量化方法详细对比...")
    d = data["ed2"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    x = np.arange(len(d["methods"]))

    # (a) 推理速度
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, d["speed"], color=PALETTE[:len(d["methods"])] * 2, edgecolor="none")
    # 按类别着色
    colors_q = ["#95a5a6", "#95a5a6", "#31688e", "#31688e", "#31688e",
                "#440154", "#440154", "#440154", "#e67e22", "#e67e22"]
    for bar, c in zip(bars, colors_q):
        bar.set_color(c)
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["methods"], rotation=45, ha="right", fontsize=4.5)
    ax1.set_ylabel("Speedup (vs FP32)")
    setup_panel_label(ax1, "(a)")

    # (b) ARI & NMI
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - 0.18, d["ari"], 0.35, label="ARI", color="#31688e")
    ax2.bar(x + 0.18, d["nmi"], 0.35, label="NMI", color="#35b779")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["methods"], rotation=45, ha="right", fontsize=4.5)
    ax2.set_ylabel("Score")
    ax2.legend(frameon=False, fontsize=5)
    ax2.set_ylim(0, 1.0)
    setup_panel_label(ax2, "(b)")

    # (c) 内存占用
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x, d["memory_gb"], color="#e74c3c", edgecolor="none")
    ax3.set_xticks(x)
    ax3.set_xticklabels(d["methods"], rotation=45, ha="right", fontsize=4.5)
    ax3.set_ylabel("GPU Memory (GB)")
    setup_panel_label(ax3, "(c)")

    # (d) 余弦相似度
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x, d["cosine_sim"], color="#fde725", edgecolor="none")
    ax4.set_xticks(x)
    ax4.set_xticklabels(d["methods"], rotation=45, ha="right", fontsize=4.5)
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_ylim(0.9, 1.01)
    setup_panel_label(ax4, "(d)")

    save_figure(fig, "ed_fig2_quantization", output_dir, fmt, dpi)


# ============================================================
# ED Fig.3: 量化对不同基因集的影响
# ============================================================
def generate_ed_fig3(data, output_dir, fmt, dpi):
    """ED Fig.3: 量化对不同基因集 embedding 的影响"""
    print("\n[ED Fig.3] 量化对不同基因集的影响...")
    d = data["ed3"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.38))
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d["gene_sets"]))
    width = 0.25
    ax.bar(x - width, d["fp16_cosine"], width, label="FP16", color="#31688e", edgecolor="none")
    ax.bar(x, d["int8_cosine"], width, label="INT8", color="#35b779", edgecolor="none")
    ax.bar(x + width, d["int4_cosine"], width, label="INT4", color="#e74c3c", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(d["gene_sets"], rotation=35, ha="right", fontsize=5)
    ax.set_ylabel("Cosine Similarity (vs FP32)")
    ax.set_ylim(0.85, 1.01)
    ax.legend(frameon=False, fontsize=5)
    ax.axhline(y=0.95, color="#e74c3c", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(len(d["gene_sets"]) - 0.5, 0.955, "Threshold (0.95)", fontsize=5,
            color="#e74c3c", ha="right")
    setup_panel_label(ax, "(a)")

    save_figure(fig, "ed_fig3_gene_sets", output_dir, fmt, dpi)


# ============================================================
# ED Fig.4: FlashAttention 不同 GPU
# ============================================================
def generate_ed_fig4(data, output_dir, fmt, dpi):
    """ED Fig.4: FlashAttention 在不同 GPU 架构上的性能"""
    print("\n[ED Fig.4] FlashAttention 不同 GPU 架构...")
    d = data["ed4"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.38))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    x = np.arange(len(d["gpus"]))

    # (a) 延迟对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - 0.18, d["standard_ms"], 0.35, label="Standard", color="#95a5a6")
    ax1.bar(x + 0.18, d["flash_ms"], 0.35, label="FlashAttn v2", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g}\n(cc {c})" for g, c in zip(d["gpus"], d["compute_cap"])], fontsize=5)
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(frameon=False, fontsize=5)
    setup_panel_label(ax1, "(a)")

    # (b) 加速比和内存节省
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - 0.18, d["speedup"], 0.35, label="Speedup", color="#31688e")
    ax2.bar(x + 0.18, d["memory_reduction"], 0.35, label="Memory Reduction", color="#2ecc71")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["gpus"], fontsize=5)
    ax2.set_ylabel("Factor")
    ax2.legend(frameon=False, fontsize=5)
    setup_panel_label(ax2, "(b)")

    save_figure(fig, "ed_fig4_flash_gpu", output_dir, fmt, dpi)


# ============================================================
# ED Fig.5: 蒸馏训练收敛曲线
# ============================================================
def generate_ed_fig5(data, output_dir, fmt, dpi):
    """ED Fig.5: 知识蒸馏训练收敛曲线"""
    print("\n[ED Fig.5] 蒸馏训练收敛曲线...")
    d = data["ed5"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.38))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    # (a) Loss 曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d["epochs"], d["train_loss"], color="#31688e", linewidth=1, label="Train Loss")
    ax1.plot(d["epochs"], d["val_loss"], color="#e74c3c", linewidth=1, linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False, fontsize=5)
    setup_panel_label(ax1, "(a)")

    # (b) ARI 收敛
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(d["epochs"], d["train_ari"], color="#31688e", linewidth=1, label="Train ARI")
    ax2.plot(d["epochs"], d["val_ari"], color="#e74c3c", linewidth=1, linestyle="--", label="Val ARI")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ARI Score")
    ax2.legend(frameon=False, fontsize=5)
    setup_panel_label(ax2, "(b)")

    save_figure(fig, "ed_fig5_convergence", output_dir, fmt, dpi)


# ============================================================
# ED Fig.6: 不同蒸馏损失函数对比
# ============================================================
def generate_ed_fig6(data, output_dir, fmt, dpi):
    """ED Fig.6: 不同蒸馏损失函数对比"""
    print("\n[ED Fig.6] 蒸馏损失函数对比...")
    d = data["ed6"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.38))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    x = np.arange(len(d["loss_types"]))

    # (a) ARI & NMI
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - 0.18, d["ari"], 0.35, label="ARI", color="#31688e")
    ax1.bar(x + 0.18, d["nmi"], 0.35, label="NMI", color="#35b779")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["loss_types"], rotation=30, ha="right", fontsize=5)
    ax1.set_ylabel("Score")
    ax1.legend(frameon=False, fontsize=5)
    ax1.set_ylim(0, 1.0)
    setup_panel_label(ax1, "(a)")

    # (b) 余弦相似度 & 速度
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    bars = ax2.bar(x - 0.18, d["cosine_sim"], 0.35, label="Cosine Sim", color="#fde725")
    line = ax2_twin.plot(x + 0.18, d["speed"], "o-", color="#e74c3c", markersize=4,
                         linewidth=1.2, label="Speedup")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["loss_types"], rotation=30, ha="right", fontsize=5)
    ax2.set_ylabel("Cosine Similarity", color="#b7950b")
    ax2_twin.set_ylabel("Speedup Factor", color="#e74c3c")
    ax2.set_ylim(0.9, 1.0)
    ax2.legend(frameon=False, fontsize=5, loc="upper left")
    ax2_twin.legend(frameon=False, fontsize=5, loc="upper right")
    setup_panel_label(ax2, "(b)")

    save_figure(fig, "ed_fig6_loss_functions", output_dir, fmt, dpi)


# ============================================================
# ED Fig.7: 超参数敏感性
# ============================================================
def generate_ed_fig7(data, output_dir, fmt, dpi):
    """ED Fig.7: 组合优化超参数敏感性"""
    print("\n[ED Fig.7] 超参数敏感性...")
    d = data["ed7"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.48))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # (a) 量化位数敏感性 (热力图)
    ax1 = fig.add_subplot(gs[0, 0])
    # 合并 attention_type 和 compile_mode 维度
    sensitivity_2d = d["sensitivity"].mean(axis=2)  # (4, 2)
    im = ax1.imshow(sensitivity_2d, cmap="viridis", aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(d["attention_type"], fontsize=5)
    ax1.set_yticks(range(len(d["quant_bits"])))
    ax1.set_yticklabels([f"INT{b}" for b in d["quant_bits"]], fontsize=5)
    ax1.set_xlabel("Attention Type")
    ax1.set_ylabel("Quantization Bits")
    for i in range(4):
        for j in range(2):
            ax1.text(j, i, f"{sensitivity_2d[i,j]:.3f}", ha="center", va="center",
                     fontsize=5.5, color="white" if sensitivity_2d[i,j] < 0.75 else "black")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    setup_panel_label(ax1, "(a)")

    # (b) 帕累托散点
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(d["pareto_speed"], d["pareto_ari"], s=30, c=[PALETTE[i % len(PALETTE)]
                for i in range(len(d["pareto_speed"]))], zorder=5, edgecolors="white", linewidths=0.5)
    for i, cfg in enumerate(d["pareto_configs"]):
        ax2.annotate(cfg, (d["pareto_speed"][i], d["pareto_ari"][i]),
                     textcoords="offset points", xytext=(4, 4), fontsize=4)
    ax2.set_xlabel("Speedup Factor")
    ax2.set_ylabel("ARI Score")
    setup_panel_label(ax2, "(b)")

    # (c) 不同 compile_mode 对比
    ax3 = fig.add_subplot(gs[1, 0])
    compile_modes = d["compile_mode"]
    for j, cm in enumerate(compile_modes):
        vals = d["sensitivity"][:, :, j].mean(axis=1)
        ax3.plot(d["quant_bits"], vals, "o-", color=PALETTE[j], label=cm, markersize=3)
    ax3.set_xlabel("Quantization Bits")
    ax3.set_ylabel("Mean ARI")
    ax3.set_xticks(d["quant_bits"])
    ax3.legend(frameon=False, fontsize=4.5)
    setup_panel_label(ax3, "(c)")

    # (d) 综合评分雷达
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    categories = ["Speed", "ARI", "NMI", "Memory", "Cosine"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    # 三组对比
    configs_scores = {
        "Q4+FA+CO": [3.2, 0.74, 0.71, 0.85, 0.94],
        "Q8+FA+CO": [2.7, 0.80, 0.77, 0.65, 0.97],
        "Q16+SDPA+MO": [1.9, 0.82, 0.79, 0.50, 0.99],
    }
    for idx, (name, scores) in enumerate(configs_scores.items()):
        vals = scores + [scores[0]]
        max_vals = [3.5, 1.0, 1.0, 1.0, 1.0, 3.5]
        norm_vals = [v / m for v, m in zip(vals, max_vals)]
        ax4.plot(angles, norm_vals, "o-", color=PALETTE[idx], linewidth=1, markersize=2.5, label=name)
        ax4.fill(angles, norm_vals, color=PALETTE[idx], alpha=0.08)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=5)
    ax4.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=4.5)
    setup_panel_label(ax4, "(d)", x=-0.1, y=1.15)

    save_figure(fig, "ed_fig7_sensitivity", output_dir, fmt, dpi)


# ============================================================
# ED Fig.8: 额外生物学任务验证
# ============================================================
def generate_ed_fig8(data, output_dir, fmt, dpi):
    """ED Fig.8: 额外生物学任务验证"""
    print("\n[ED Fig.8] 额外生物学任务验证...")
    d = data["ed8"]
    w = mm_to_inch(NATURE_DOUBLE_COL_MM)
    fig = plt.figure(figsize=(w, w * 0.38))
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d["tasks"]))
    ax.bar(x - 0.18, d["baseline"], 0.35, label="Baseline", color="#95a5a6")
    ax.bar(x + 0.18, d["scinfer"], 0.35, label="scInfer", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(d["tasks"], rotation=30, ha="right", fontsize=5)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=5)
    ax.set_ylim(0, 1.0)
    # 标注显著性
    for i, pval in enumerate(d["p_values"]):
        marker = "*" if pval < 0.05 else "ns"
        ax.text(i, max(d["baseline"][i], d["scinfer"][i]) + 0.02,
                marker, ha="center", fontsize=6, fontweight="bold",
                color="#e74c3c" if pval < 0.05 else "#95a5a6")
    setup_panel_label(ax, "(a)")

    save_figure(fig, "ed_fig8_bio_validation", output_dir, fmt, dpi)


# ============================================================
# Extended Data Tables
# ============================================================
def generate_ed_tables(data, output_dir):
    """生成 ED Table 1 和 ED Table 2 (Markdown + CSV)"""
    print("\n[ED Tables] 生成扩展数据表格...")
    os.makedirs(output_dir, exist_ok=True)

    for table_name, table_key in [("ed_table1_benchmark", "ed_table1"),
                                   ("ed_table2_bio_stats", "ed_table2")]:
        d = data[table_key]
        # CSV
        csv_path = os.path.join(output_dir, f"{table_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(d["headers"])
            writer.writerows(d["rows"])
        print(f"  [OK] {csv_path}")

        # Markdown
        md_path = os.path.join(output_dir, f"{table_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"| {' | '.join(d['headers'])} |\n")
            f.write(f"| {' | '.join(['---'] * len(d['headers']))} |\n")
            for row in d["rows"]:
                f.write(f"| {' | '.join(row)} |\n")
        print(f"  [OK] {md_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="scInfer Extended Data Figures 生成")
    parser.add_argument("--data-dir", default="results/", help="实验数据目录")
    parser.add_argument("--output-dir", default="figures/extended/", help="输出目录")
    parser.add_argument("--format", default="pdf", choices=["pdf", "eps", "tiff", "png"])
    parser.add_argument("--dpi", type=int, default=350, help="输出分辨率")
    parser.add_argument("--figs", nargs="*", default=None, help="指定生成哪些图 (1-8 + tables)")
    args = parser.parse_args()

    print("=" * 60)
    print("  scInfer Extended Data Figures 生成器")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  格式: {args.format} | DPI: {args.dpi}")
    print("=" * 60)

    data = load_or_synthesize(args.data_dir)

    fig_funcs = {
        1: generate_ed_fig1,
        2: generate_ed_fig2,
        3: generate_ed_fig3,
        4: generate_ed_fig4,
        5: generate_ed_fig5,
        6: generate_ed_fig6,
        7: generate_ed_fig7,
        8: generate_ed_fig8,
    }

    figs_to_gen = args.figs if args.figs else list(range(1, 9))

    for fig_num in figs_to_gen:
        if fig_num == "tables" or fig_num == "table":
            generate_ed_tables(data, args.output_dir)
        elif isinstance(fig_num, int) and fig_num in fig_funcs:
            fig_funcs[fig_num](data, args.output_dir, args.format, args.dpi)
        else:
            print(f"[WARN] 未知图表编号: {fig_num}")

    # 始终生成表格
    if "tables" not in figs_to_gen and "table" not in figs_to_gen:
        generate_ed_tables(data, args.output_dir)

    print("\n" + "=" * 60)
    print("  所有 Extended Data Figures 生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
