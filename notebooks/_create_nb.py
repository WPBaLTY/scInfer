import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# scInfer: Accelerating Single-Cell Foundation Model Inference\n",
                "\n",
                "**Paper Figures Collection**\n",
                "\n",
                "This notebook generates all figures for the manuscript."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import sys\n",
                "sys.path.insert(0, \"..\")\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from scripts.generate_figures import generate_all_figures"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 1: Motivation and Framework Overview"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig1 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig1\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 2: Inference Bottleneck Profiling"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig2 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig2\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 3: Quantization Systematic Evaluation"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig3 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig3\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 4: FlashAttention Upgrade"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig4 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig4\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 5: Knowledge Distillation"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig5 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig5\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 6: Combined Optimization & Biological Validation"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig6 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig6\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Figure 7: Real-world Application Demo"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig7 = generate_all_figures(data_dir=\"../results\", output_dir=\"../figures/figs\", figs=[\"fig7\"])"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Extended Data Figures"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from scripts.generate_extended_data import generate_all_extended_data\n",
                "generate_all_extended_data(data_dir=\"../results\", output_dir=\"../figures/extended\")"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Summary\n",
                "\n",
                "All 7 main figures and 10 extended data figures have been generated."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/07_paper_figures.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {len(nb['cells'])} cells")
