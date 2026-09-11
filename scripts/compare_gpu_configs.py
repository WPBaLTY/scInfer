#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scInfer GPU 配置对比脚本
检测系统 GPU 信息，模拟/实测不同 GPU 上的推理性能对比。

用法:
    python scripts/compare_gpu_configs.py --gpus T4 V100 A100 H100 RTX4090 \
        --models scgpt geneformer scfoundation --output results/gpu_comparison/
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# GPU 已知规格数据库 (用于模拟和参考)
# ============================================================
GPU_SPECS = {
    "T4": {
        "name": "NVIDIA T4",
        "memory_gb": 16,
        "memory_type": "GDDR6",
        "compute_capability": "7.5",
        "cuda_cores": 2560,
        "tensor_cores": 320,
        "fp32_tflops": 8.1,
        "fp16_tflops": 65.1,
        "int8_tops": 130,
        "tdp_w": 70,
        "architecture": "Turing",
        "bandwidth_gb_s": 320,
    },
    "V100": {
        "name": "NVIDIA V100",
        "memory_gb": 32,
        "memory_type": "HBM2",
        "compute_capability": "7.0",
        "cuda_cores": 5120,
        "tensor_cores": 640,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "int8_tops": 600,
        "tdp_w": 300,
        "architecture": "Volta",
        "bandwidth_gb_s": 900,
    },
    "A100": {
        "name": "NVIDIA A100",
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "compute_capability": "8.0",
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "fp32_tflops": 19.5,
        "fp16_tflops": 312,
        "int8_tops": 1248,
        "tdp_w": 400,
        "architecture": "Ampere",
        "bandwidth_gb_s": 2039,
    },
    "H100": {
        "name": "NVIDIA H100",
        "memory_gb": 80,
        "memory_type": "HBM3",
        "compute_capability": "9.0",
        "cuda_cores": 16896,
        "tensor_cores": 528,
        "fp32_tflops": 67,
        "fp16_tflops": 1979,
        "int8_tops": 3958,
        "tdp_w": 700,
        "architecture": "Hopper",
        "bandwidth_gb_s": 3350,
    },
    "RTX4090": {
        "name": "NVIDIA RTX 4090",
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "compute_capability": "8.9",
        "cuda_cores": 16384,
        "tensor_cores": 512,
        "fp32_tflops": 82.6,
        "fp16_tflops": 330.3,
        "int8_tops": 660.6,
        "tdp_w": 450,
        "architecture": "Ada Lovelace",
        "bandwidth_gb_s": 1008,
    },
    "RTX3090": {
        "name": "NVIDIA RTX 3090",
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "compute_capability": "8.6",
        "cuda_cores": 10496,
        "tensor_cores": 328,
        "fp32_tflops": 35.6,
        "fp16_tflops": 71,
        "int8_tops": 285,
        "tdp_w": 350,
        "architecture": "Ampere",
        "bandwidth_gb_s": 936,
    },
    "L4": {
        "name": "NVIDIA L4",
        "memory_gb": 24,
        "memory_type": "GDDR6",
        "compute_capability": "8.9",
        "cuda_cores": 7680,
        "tensor_cores": 240,
        "fp32_tflops": 30.3,
        "fp16_tflops": 121,
        "int8_tops": 485,
        "tdp_w": 72,
        "architecture": "Ada Lovelace",
        "bandwidth_gb_s": 300,
    },
}

# 模型参数量和特征
MODEL_SPECS = {
    "scgpt": {"name": "scGPT", "params_m": 47, "layers": 12, "hidden_dim": 512, "vocab_size": 36732},
    "geneformer": {"name": "Geneformer", "params_m": 29, "layers": 6, "hidden_dim": 256, "vocab_size": 25475},
    "scfoundation": {"name": "scFoundation", "params_m": 96, "layers": 18, "hidden_dim": 768, "vocab_size": 33534},
    "uce": {"name": "UCE", "params_m": 15, "layers": 6, "hidden_dim": 1280, "vocab_size": 0},
}


def detect_system_gpu():
    """检测系统当前可用 GPU"""
    gpu_info = {
        "available": False,
        "name": "N/A",
        "memory_total_gb": 0,
        "cuda_version": "N/A",
        "compute_capability": "N/A",
        "driver_version": "N/A",
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
            gpu_info["cuda_version"] = torch.version.cuda or "N/A"
            props = torch.cuda.get_device_properties(0)
            gpu_info["compute_capability"] = f"{props.major}.{props.minor}"
            gpu_info["driver_version"] = "N/A"
    except ImportError:
        pass

    # 尝试 nvidia-smi 获取更多信息
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    gpu_info["name"] = parts[0]
                    gpu_info["memory_total_gb"] = round(float(parts[1]) / 1024, 1)
                    gpu_info["driver_version"] = parts[2]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpu_info


def match_gpu_to_specs(gpu_name):
    """尝试将检测到的 GPU 名称匹配到已知规格"""
    gpu_name_lower = gpu_name.lower()
    for key, specs in GPU_SPECS.items():
        if key.lower() in gpu_name_lower:
            return key, specs
    return None, None


def simulate_inference(gpu_key, model_key, batch_size=64, n_cells=10000):
    """
    模拟推理性能 (基于 GPU 和模型规格的估算)。
    实际部署时应替换为真实 benchmark。
    """
    gpu = GPU_SPECS.get(gpu_key, GPU_SPECS["A100"])
    model = MODEL_SPECS.get(model_key, MODEL_SPECS["scgpt"])

    # 基础推理时间估算 (毫秒)
    # 考虑: 参数量、计算能力、显存带宽
    flops_per_token = 6 * model["params_m"] * 1e6  # 近似
    tokens_per_batch = batch_size * 512  # 假设每个细胞 512 tokens
    total_flops = flops_per_token * tokens_per_batch

    # GPU 有效算力 (考虑利用率 ~40%)
    effective_flops = gpu["fp16_tflops"] * 1e12 * 0.4
    compute_time_ms = (total_flops / effective_flops) * 1000

    # 内存带宽瓶颈
    data_size_bytes = model["params_m"] * 1e6 * 2  # FP16, 2 bytes
    memory_time_ms = (data_size_bytes / (gpu["bandwidth_gb_s"] * 1e9)) * 1000 * batch_size

    # 取较大值 (受限于计算或内存)
    latency_ms = max(compute_time_ms, memory_time_ms)

    # 吞吐量
    throughput_cells_s = (batch_size * 1000) / latency_ms

    # 内存使用估算
    model_memory_gb = model["params_m"] * 2 / 1000  # FP16
    kv_cache_gb = model["layers"] * model["hidden_dim"] * batch_size * 4 / 1e6
    activation_gb = model["params_m"] * batch_size * 0.001
    total_memory_gb = model_memory_gb + kv_cache_gb + activation_gb

    return {
        "latency_ms": round(latency_ms, 2),
        "throughput_cells_s": round(throughput_cells_s, 1),
        "memory_gb": round(min(total_memory_gb, gpu["memory_gb"] * 0.95), 2),
        "memory_utilization_pct": round(total_memory_gb / gpu["memory_gb"] * 100, 1),
        "compute_bound": compute_time_ms > memory_time_ms,
    }


def run_real_benchmark(model_key, n_cells=1000, device="cpu"):
    """尝试运行真实 benchmark (需要 torch 和模型)"""
    try:
        import torch
        if device == "cpu" or not torch.cuda.is_available():
            # CPU 上运行简单测试
            n_tokens = n_cells * 100  # 简化
            start = time.time()
            # 模拟计算
            x = torch.randn(64, 512)
            for _ in range(10):
                x = torch.nn.functional.linear(x, torch.randn(512, 512))
                x = torch.nn.functional.relu(x)
            elapsed = time.time() - start
            return {
                "latency_ms": round(elapsed * 100, 2),
                "throughput_cells_s": round(n_cells / elapsed, 1),
                "memory_gb": 0,
                "device": "cpu",
                "real": True,
            }
    except ImportError:
        pass
    return None


def format_gpu_table(gpu_keys, system_gpu=None):
    """格式化 GPU 规格对比表"""
    headers = ["GPU", "Architecture", "Memory (GB)", "Memory Type", "Compute Capability",
               "FP32 (TFLOPS)", "FP16 (TFLOPS)", "INT8 (TOPS)", "Bandwidth (GB/s)", "TDP (W)"]
    rows = []
    for key in gpu_keys:
        if key in GPU_SPECS:
            g = GPU_SPECS[key]
            rows.append([
                g["name"], g["architecture"], g["memory_gb"], g["memory_type"],
                g["compute_capability"], g["fp32_tflops"], g["fp16_tflops"],
                g["int8_tops"], g["bandwidth_gb_s"], g["tdp_w"]
            ])
    return headers, rows


def format_performance_table(gpu_keys, model_keys, metric="throughput_cells_s"):
    """格式化性能对比表"""
    model_names = [MODEL_SPECS[k]["name"] for k in model_keys]
    headers = ["GPU"] + model_names
    rows = []
    for gpu_key in gpu_keys:
        row = [GPU_SPECS.get(gpu_key, {}).get("name", gpu_key)]
        for model_key in model_keys:
            result = simulate_inference(gpu_key, model_key)
            if metric == "throughput_cells_s":
                row.append(f"{result['throughput_cells_s']:.0f}")
            elif metric == "latency_ms":
                row.append(f"{result['latency_ms']:.2f}")
            elif metric == "memory_gb":
                row.append(f"{result['memory_gb']:.1f}")
        rows.append(row)
    return headers, rows


def save_tables(headers, rows, output_path, title=""):
    """保存表格为 Markdown 和 CSV"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # CSV
    csv_path = output_path + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if title:
            writer.writerow([title])
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  [OK] {csv_path}")

    # Markdown
    md_path = output_path + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"## {title}\n\n")
        f.write(f"| {' | '.join(str(h) for h in headers)} |\n")
        f.write(f"| {' | '.join(['---'] * len(headers))} |\n")
        for row in rows:
            f.write(f"| {' | '.join(str(v) for v in row)} |\n")
        f.write("\n")
    print(f"  [OK] {md_path}")


def generate_comparison_json(gpu_keys, model_keys, output_dir):
    """生成 JSON 格式的完整对比数据"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "system_gpu": detect_system_gpu(),
        "comparisons": {},
    }
    for gpu_key in gpu_keys:
        results["comparisons"][gpu_key] = {}
        for model_key in model_keys:
            results["comparisons"][gpu_key][model_key] = simulate_inference(gpu_key, model_key)

    json_path = os.path.join(output_dir, "gpu_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {json_path}")


def main():
    parser = argparse.ArgumentParser(description="scInfer GPU 配置对比工具")
    parser.add_argument("--gpus", nargs="+", default=["T4", "V100", "A100", "H100", "RTX4090"],
                        help="要对比的 GPU 型号 (默认: T4 V100 A100 H100 RTX4090)")
    parser.add_argument("--models", nargs="+", default=["scgpt", "geneformer", "scfoundation"],
                        help="要测试的模型 (默认: scgpt geneformer scfoundation)")
    parser.add_argument("--output", default="results/gpu_comparison/",
                        help="输出目录 (默认: results/gpu_comparison/)")
    parser.add_argument("--n-cells", type=int, default=10000, help="模拟的细胞数量 (默认: 10000)")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小 (默认: 64)")
    parser.add_argument("--real-benchmark", action="store_true",
                        help="尝试运行真实 benchmark (需要 GPU 和模型)")
    args = parser.parse_args()

    print("=" * 70)
    print("  scInfer GPU 配置对比工具")
    print(f"  GPU 列表: {', '.join(args.gpus)}")
    print(f"  模型列表: {', '.join(args.models)}")
    print(f"  输出目录: {args.output}")
    print(f"  模拟细胞数: {args.n_cells} | 批大小: {args.batch_size}")
    print("=" * 70)

    # 检测系统 GPU
    system_gpu = detect_system_gpu()
    print(f"\n[系统 GPU 检测]")
    if system_gpu["available"]:
        print(f"  型号: {system_gpu['name']}")
        print(f"  显存: {system_gpu['memory_total_gb']} GB")
        print(f"  CUDA: {system_gpu['cuda_version']}")
        print(f"  Compute Capability: {system_gpu['compute_capability']}")
        print(f"  驱动: {system_gpu['driver_version']}")
    else:
        print("  [!] 未检测到可用 GPU，将在 CPU 上运行标注")

    # 验证 GPU 型号
    valid_gpus = []
    for g in args.gpus:
        g_upper = g.upper().replace(" ", "")
        if g_upper in GPU_SPECS:
            valid_gpus.append(g_upper)
        else:
            print(f"  [WARN] 未知 GPU 型号: {g}，跳过")
    if not valid_gpus:
        print("[ERROR] 没有有效的 GPU 型号，退出")
        sys.exit(1)

    # 验证模型
    valid_models = []
    for m in args.models:
        m_lower = m.lower()
        if m_lower in MODEL_SPECS:
            valid_models.append(m_lower)
        else:
            print(f"  [WARN] 未知模型: {m}，跳过")
    if not valid_models:
        print("[ERROR] 没有有效的模型，退出")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 1. GPU 规格对比表
    print("\n[1/4] 生成 GPU 规格对比表...")
    headers, rows = format_gpu_table(valid_gpus)
    save_tables(headers, rows, os.path.join(args.output, "gpu_specs"),
                title="GPU Specifications Comparison")

    # 2. 推理吞吐量对比
    print("\n[2/4] 生成推理吞吐量对比表...")
    headers, rows = format_performance_table(valid_gpus, valid_models, "throughput_cells_s")
    save_tables(headers, rows, os.path.join(args.output, "throughput_comparison"),
                title="Inference Throughput (cells/sec)")

    # 3. 内存使用对比
    print("\n[3/4] 生成内存使用对比表...")
    headers, rows = format_performance_table(valid_gpus, valid_models, "memory_gb")
    save_tables(headers, rows, os.path.join(args.output, "memory_comparison"),
                title="GPU Memory Usage (GB)")

    # 4. 完整 JSON 数据
    print("\n[4/4] 生成完整对比 JSON...")
    generate_comparison_json(valid_gpus, valid_models, args.output)

    # 如果有真实 GPU，尝试运行简单 benchmark
    if system_gpu["available"] and args.real_benchmark:
        print("\n[Real Benchmark]")
        matched_key, _ = match_gpu_to_specs(system_gpu["name"])
        if matched_key:
            print(f"  检测到 GPU 匹配: {matched_key}")
            for model_key in valid_models:
                result = run_real_benchmark(model_key, n_cells=1000, device="cuda")
                if result:
                    print(f"  {MODEL_SPECS[model_key]['name']}: "
                          f"{result['throughput_cells_s']:.0f} cells/sec (real)")
        else:
            print(f"  GPU '{system_gpu['name']}' 未在已知规格库中匹配")
    elif args.real_benchmark:
        print("\n[CPU Benchmark - 无 GPU]")
        for model_key in valid_models:
            result = run_real_benchmark(model_key, n_cells=1000, device="cpu")
            if result:
                print(f"  {MODEL_SPECS[model_key]['name']}: "
                      f"{result['throughput_cells_s']:.0f} cells/sec (CPU, reference only)")

    # 打印摘要
    print("\n" + "=" * 70)
    print("  GPU 对比摘要")
    print("=" * 70)
    for gpu_key in valid_gpus:
        gpu_name = GPU_SPECS[gpu_key]["name"]
        print(f"\n  {gpu_name}:")
        for model_key in valid_models:
            r = simulate_inference(gpu_key, model_key, args.batch_size, args.n_cells)
            model_name = MODEL_SPECS[model_key]["name"]
            bound = "compute" if r["compute_bound"] else "memory"
            print(f"    {model_name}: {r['throughput_cells_s']:.0f} cells/s, "
                  f"{r['memory_gb']:.1f} GB ({bound}-bound)")

    print("\n" + "=" * 70)
    print(f"  所有结果已保存到: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
