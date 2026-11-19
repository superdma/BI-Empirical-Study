"""Environment schematic: the two stacks share hardware, not versions.

All values come from docs/02_environment.md and docs/12_sglang_results.md.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import C, WIDTH, export, setup  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")

LAYERS = [
    ("引擎", "vLLM 0.23.0", "SGLang 0.5.6"),
    ("深度学习框架", "PyTorch 2.11.0+cu129", "PyTorch 2.9.1+cu128"),
    ("CUDA", "12.9 运行时", "12.8 运行时 / 12.8 nvcc（JIT）"),
    ("驱动", "550.120", "550.120"),
]


def box(ax, x, y, w, h, text, face, edge, size=7, weight="normal", color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge,
                                linewidth=0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.12"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight=weight, color=color or C["ink"])


def main():
    setup()
    fig = plt.figure(figsize=(WIDTH, 2.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")

    box(ax, 1, 0.6, 98, 4.6,
        "4 × RTX A6000 48 GB（SM86 / Ampere，PCIe，无 InfiniBand）\n"
        "没有 TMA、没有 FP8 tensor core；官方 tuned matmul 表只覆盖 Ada / Hopper / Blackwell",
        "#eef0f4", C["muted"], size=7)
    box(ax, 20, 29.0, 60, 2.4, "同一台机器，两套互不覆盖的环境",
        "white", C["ink"], size=7.6, weight="bold")

    for i, (layer, left, right) in enumerate(LAYERS):
        y = 23.4 - i * 3.8
        ax.text(10.0, y + 1.5, layer, ha="right", va="center", fontsize=7, color=C["muted"])
        box(ax, 14.0, y, 34.0, 3.0, left, "#eef1f8", C["vllm"], size=6.7)
        box(ax, 52.0, y, 34.0, 3.0, right, "#eaf4f2", C["sgl"], size=6.7)

    box(ax, 14.0, 8.6, 34.0, 2.6, "VLLM_BATCH_INVARIANT=1", "white", C["vllm"],
        size=6.7, color=C["vllm"])
    box(ax, 52.0, 8.6, 34.0, 2.6, "--enable-deterministic-inference", "white", C["sgl"],
        size=6.7, color=C["sgl"])

    ax.text(31.0, 27.2, "vLLM", ha="center", fontsize=7.5, fontweight="bold", color=C["vllm"])
    ax.text(69.0, 27.2, "SGLang", ha="center", fontsize=7.5, fontweight="bold", color=C["sgl"])
    ax.text(50.0, 7.0, "两套环境的 Python 均为 3.12，模型权重共用", ha="center",
            fontsize=6.4, color=C["muted"])
    export(fig, "fig_env", OUT, multipanel=False)


if __name__ == "__main__":
    main()
