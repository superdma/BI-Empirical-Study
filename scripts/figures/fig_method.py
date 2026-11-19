"""Schematic: what "bitwise identical" means in this report.

Every number in the figure comes from one eager run of the 32-prompt test:
prompt0 (the 5-token target) is the passing example, prompt1 is a failing one
whose third logprob differs in the low bits.
"""
import json
import os
import struct

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import C, WIDTH, export, setup

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")


def bits(x):
    return struct.pack(">d", float(x)).hex()


def chip(ax, x, y, w, h, text, face, edge, color=None, size=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge,
                                linewidth=0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.05"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color or C["ink"])


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.7, color=C["muted"],
                                 shrinkA=1, shrinkB=1))


def main():
    setup()
    run = json.load(open(f"{ROOT}/results/extra/bi_batchall_offline_eager.json"))
    solo, batch = run["solo"], run["batch"]
    diag = json.load(open(f"{ROOT}/results/extra/bi_diag_bs256_pc0.json"))["cases"]
    d_single, d_batch = diag["single"]["logprobs"], diag["batch"]["logprobs"]
    window = (4, 5, 6)

    fig = plt.figure(figsize=(WIDTH, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    ax.text(1, 28.4, "同一个 prompt，两条执行路径，逐位比较 logprob",
            fontsize=8.4, fontweight="bold")
    chip(ax, 1, 23.0, 16, 2.8, "prompt #0\n（5-token target）", "#f2f4f6", C["muted"], size=6.1)
    chip(ax, 21.0, 24.9, 14.5, 2.5, "单独请求 bs=1", "white", C["vllm"], size=6.3)
    chip(ax, 21.0, 21.2, 14.5, 2.5, "批内请求 bs=32", "white", C["sgl"], size=6.3)
    arrow(ax, (17.2, 24.6), (20.7, 26.1))
    arrow(ax, (17.2, 24.1), (20.7, 22.4))

    for i in range(2):
        chip(ax, 38.5 + i * 13.0, 24.9, 12.4, 2.6,
             f"{solo[0]['logprobs'][i]:.4f}\n{bits(solo[0]['logprobs'][i])[:12]}…",
             "white", C["muted"], size=5.7)
        chip(ax, 38.5 + i * 13.0, 21.2, 12.4, 2.6,
             f"{batch[0]['logprobs'][i]:.4f}\n{bits(batch[0]['logprobs'][i])[:12]}…",
             "white", C["muted"], size=5.7)
    ax.text(65.0, 26.2, "64 位模式相同", fontsize=6.3, color=C["pass"])
    ax.text(65.0, 22.5, "判为一致", fontsize=6.3, color=C["pass"])

    ax.plot([1, 72], [18.6, 18.6], color=C["grid"], linewidth=0.7)
    ax.text(1, 17.1, "eager bs=256：前 5 个 logprob 位级相同，从第 6 个（i=5）开始不同",
            fontsize=7, color=C["muted"])

    for label, vals, y in (("单独", d_single, 13.0), ("批内", d_batch, 8.8)):
        ax.text(30.0, y + 1.35, label, ha="right", va="center", fontsize=6.3)
        for n, j in enumerate(window):
            same_bit = bits(vals[j]) == bits((d_batch if label == "单独" else d_single)[j])
            face, edge = "white", C["muted"]
            color = C["ink"]
            if not same_bit:
                face, edge, color = "#fbe9e4", C["fail"], C["fail"]
            chip(ax, 32.0 + n * 15.2, y, 14.2, 2.7,
                 f"i={j}  {vals[j]:.6f}\n{bits(vals[j])}", face, edge, color=color, size=5.4)
    ax.text(79.5, 11.4, "i=4 相同", fontsize=6.3, color=C["pass"])
    ax.text(79.5, 9.4, "i=5、6 位模式不同", fontsize=6.3, color=C["fail"])
    ax.text(79.5, 7.4, "判为不一致", fontsize=6.3, color=C["fail"])

    ax.text(1, 2.4, "口径：32 个生成 token 的 logprob 逐个转成 IEEE-754 位模式比较；"
            "token id 相同但 logprob 不同，同样算失败。",
            fontsize=6.4, color=C["muted"])
    export(fig, "fig_method", OUT, multipanel=False)


if __name__ == "__main__":
    main()
