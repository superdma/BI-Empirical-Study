"""What BI pins, and where a bit difference can still come from.

Panel a: the M-tile each compute path picks as the batch grows.  The BI
linear and the BI MoE return one constant; the MoE heuristic without BI
steps at 33/97/513, none of which is 256.  Panel b: the 255-vs-256 all-reduce
comparison through the engine's pynccl communicator, with and without the
NCCL environment BI installs.

Data: results/tier2/bi_bi_coverage.json, bi_pynccl_*.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier2")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

COMMS = [
    ("BI 环境：tree + 单通道", "bi_pynccl_bi_env.json", C["pass"]),
    ("tree + 8 通道", "bi_pynccl_ch8.json", C["warn"]),
    ("NCCL 自动选择", "bi_pynccl_auto.json", C["fail"]),
]


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def pair_255_256(data):
    for p in data["pairs"]:
        if p["a"] == 255 and p["b"] == 256:
            return p["n_diff"], p["total"]
    return None, None


def main():
    setup()
    rows = []
    cov = load("bi_bi_coverage.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    fig.subplots_adjust(left=0.135, right=0.975, top=0.84, bottom=0.21, wspace=0.52)

    # a: BLOCK_SIZE_M against batch size
    ax = axes[0]
    series = [
        ("MoE，BI 关", [(r["M"], r["config"]["BLOCK_SIZE_M"])
                     for r in cov["moe_bi_off"]["rows"]], C["fail"], "--"),
        ("MoE，BI 开", [(r["M"], r["config"]["BLOCK_SIZE_M"])
                     for r in cov["moe_bi_on"]["rows"]], C["sgl"], "-"),
        ("BI 线性层", [(r["M"], r["config"]["BLOCK_SIZE_M"])
                   for r in cov["matmul"]["rows"]
                   if "shape" not in r], C["vllm"], "-"),
    ]
    for label, pts, color, style in series:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                linewidth=1.1, linestyle=style, label=label, zorder=3)
        rows.append(["a", label, len(pts), sorted({p[1] for p in pts})])
    ax.axvline(256, color=C["muted"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
    ax.text(256, 137, " 256", fontsize=6.4, color=C["muted"], va="top")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 32, 64, 128])
    ax.set_xlabel("批大小 M")
    ax.set_ylabel("所选 M 方向 tile")
    ax.set_title("配置随 M 的切换点", loc="left", fontsize=7.6, pad=6)
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.6,
              handletextpad=0.5, borderpad=0.2, labelspacing=0.35,
              bbox_to_anchor=(1.0, 0.02))
    tidy(ax)
    panel_label(ax, "a", dx=-0.22, dy=1.03)

    # b: 255 vs 256 through the engine communicator
    ax = axes[1]
    for i, (label, fname, color) in enumerate(COMMS):
        data = load(fname)
        n, total = pair_255_256(data)
        y = len(COMMS) - 1 - i
        ax.barh(y, n, color=color, height=0.55, zorder=2)
        text = "0" if n == 0 else f"{n}"
        ax.text(n + 25, y, f"{text} / {total}", va="center", fontsize=6.4,
                color=C["ink"] if n == 0 else C["fail"])
        rows.append(["b", label, n, total])
    ax.set_yticks(range(len(COMMS)))
    ax.set_yticklabels([c[0] for c in COMMS][::-1], fontsize=6.4)
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 500, 1000])
    ax.set_xlabel("批 255 与 256 之间改位的元素数")
    ax.set_title("同一个 all-reduce 的 255 与 256", loc="left", fontsize=7.6, pad=6)
    ax.set_ylim(-0.6, len(COMMS) - 0.4)
    tidy(ax, "x")
    panel_label(ax, "b", dx=-0.36, dy=1.03)

    with open(os.path.join(OUT, "fig_bi_coverage.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "item", "value"])
        writer.writerows(rows)

    export(fig, "fig_bi_coverage", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
