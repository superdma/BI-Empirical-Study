"""Schematic of the mechanism argued in chapter 7.

The chain is: request shape -> kernel/tile selection -> floating-point
reduction order -> last bits of logits -> (sometimes) a flipped argmax. The
two brackets underneath separate what batch invariance fixes from what it
leaves alone; the bottom row carries the four observations that constrain the
explanation. Nothing here is a measurement: the numbers quoted in the evidence
row come from chapters 4-6 and 12.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import C, WIDTH, export, setup

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")


def box(ax, x, y, w, h, text, face, edge, size=6.6, weight="normal", color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge,
                                linewidth=0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.6"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color or C["ink"], linespacing=1.35)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.7, color=C["muted"],
                                 shrinkA=0, shrinkB=0))


def main():
    setup()
    fig = plt.figure(figsize=(WIDTH, 2.42))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    ax.text(1, 36.4, "一个自洽的解释：形状先决定 kernel，再决定末位数值",
            fontsize=8.2, fontweight="bold")

    y, h = 27.0, 6.0
    stages = [(1.0, 15.5, "请求形状\nbatch / 每序列长度"),
              (21.5, 19.0, "kernel 与 tile 选择\nmatmul · attention split\nMoE tile · 跨卡归约"),
              (45.5, 17.0, "浮点归约顺序\n在 tile / split 之间变化"),
              (67.5, 15.5, "logits 末位\n差 1e-7 到 1e-1"),
              (88.0, 11.0, "贪心 argmax\n平局时翻 token")]
    for (x, w, text) in stages:
        face = "white" if x != 21.5 else "#f2f4f6"
        box(ax, x, y, w, h, text, face, C["muted"])
    for x0, x1 in ((16.7, 21.3), (40.7, 45.3), (62.7, 67.3), (83.2, 87.8)):
        arrow(ax, x0, x1, y + h / 2)

    # what BI fixes / leaves alone, hung under the kernel-selection stage
    box(ax, 21.5, 16.6, 31.0, 5.0,
        "BI 已经固定\nmatmul 归约顺序 · attention split · MoE tile",
        "#eaf4f2", C["pass"], size=6.3)
    box(ax, 54.5, 16.6, 30.0, 5.0,
        "BI 没有覆盖\nTP all-reduce 的算法选择 · 编译产物",
        "#fbeae5", C["fail"], size=6.3)
    ax.plot([31.0, 31.0], [26.8, 24.4], color=C["muted"], linewidth=0.6)
    ax.plot([37.0, 69.5], [24.4, 24.4], color=C["muted"], linewidth=0.6)
    ax.plot([31.0, 37.0], [24.4, 24.4], color=C["pass"], linewidth=0.6)
    ax.plot([37.0, 37.0], [24.4, 21.8], color=C["pass"], linewidth=0.6)
    ax.plot([69.5, 69.5], [24.4, 21.8], color=C["fail"], linewidth=0.6)

    # evidence row
    ax.text(1, 12.6, "支撑这一解释的观测（强度 A：都是重复过的现象）",
            fontsize=6.8, color=C["muted"])
    items = [
        (1.0, 23.0, "256 附近有边界\n255 → 0.0，256 → 8.805e-02"),
        (25.0, 23.5, "换 attention 后端、关 chunked prefill\n不改变 eager 的失败"),
        (49.5, 23.0, "TP=4 失败、PP=4 通过\n（30B 与 0.6B 都是）"),
        (73.5, 25.5, "compiled 在全部形状上通过\n换编译产物又会变"),
    ]
    for (x, w, text) in items:
        box(ax, x, 5.6, w, 5.6, text, "white", C["grid"], size=6.0)

    ax.text(1, 2.8, "强度：现象为 A；第 9 章逐项测过 tile 与 split 都没有变，"
            "差异位置在某一层 norm 的输出（A），触发点尚未定位（C）。",
            fontsize=6.4, color=C["muted"])
    ax.text(1, 0.7, "这一链条同时覆盖跨 attention 后端、跨 MoE/稠密模型、跨并行方式的失败。",
            fontsize=6.4, color=C["muted"])
    export(fig, "fig_mechanism", OUT, multipanel=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
