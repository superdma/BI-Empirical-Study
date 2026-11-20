"""Where the replay lands, and how the boundary behaves past 256.

Panel a: the same three inputs replayed with and without batch invariance.
With BI off, every input reproduces the 255 run; with BI on each input
reproduces the run it came from.

Panel b: decode batch sizes from 255 to 512 against the 255 baseline. The
failure count steps up at 256 and then stays flat.

Data: results/tier4/probe_bi0.json, probe_bi1.json, bi_boundary_sweep.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier4")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

CASES = ["x256", "x255", "xfill256"]
CASE_LABELS = {"x256": "256 的输入", "x255": "255 的输入", "xfill256": "第 0 行\n复制 256 行"}


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def main():
    setup()
    bi0 = load("probe_bi0.json")
    bi1 = load("probe_bi1.json")
    sweep = load("bi_boundary_sweep.json")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.82, bottom=0.235, wspace=0.42)
    rows = []

    # a: 4 x 3 matrix of differences against each engine run
    ax = axes[0]
    by_case = {b: {c["label"]: c for c in data["cases"]}
               for b, data in (("0", bi0), ("1", bi1))}
    # Rows run bottom-up in matplotlib; each block lists 255 under 256.
    grid = [[by_case["0"][c]["vs_engine255"] for c in CASES],
            [by_case["0"][c]["vs_engine256"] for c in CASES],
            [by_case["1"][c]["vs_engine255"] for c in CASES],
            [by_case["1"][c]["vs_engine256"] for c in CASES]]
    row_names = ["BI 关 · 对 255", "BI 关 · 对 256",
                 "BI 开 · 对 255", "BI 开 · 对 256"]
    for i in range(4):
        for j, case in enumerate(CASES):
            value = grid[i][j]
            color = C["fail"] if value else C["pass"]
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, facecolor=color,
                                       alpha=0.16 if value else 0.12,
                                       edgecolor="white", linewidth=1.2, zorder=2))
            ax.text(j, i, str(value), ha="center", va="center", fontsize=7.0,
                    color=C["ink"], zorder=3)
            rows.append(["a", row_names[i], case, value])
    ax.set_xlim(-0.6, len(CASES) - 0.4)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], fontsize=5.9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(row_names, fontsize=6.2)
    ax.set_title("与引擎输出的差异元素（第 0 行）", loc="left", fontsize=7.4, pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, "a", dx=-0.30, dy=1.04)

    # b: changed prompts against decode batch size
    ax = axes[1]
    sizes = [int(k) for k in sweep["vs_255"]]
    changed = [sweep["vs_255"][str(n)]["changed"] for n in sizes]
    ax.step(sizes, changed, where="mid", color=C["vllm"], linewidth=1.3, zorder=3)
    ax.scatter(sizes, changed, s=14, color=C["vllm"], zorder=4)
    ax.axvline(256, color=C["muted"], linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate("256", xy=(259, 8), ha="left", va="bottom", fontsize=6.2,
                color=C["muted"])
    ax.set_xlabel("decode 批大小", labelpad=2)
    ax.set_ylabel("改变的 prompt 条数（共 255）", labelpad=2)
    ax.set_xlim(240, 525)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("过 256 之后停在同一个水平", loc="left", fontsize=7.4, pad=6)
    tidy(ax)
    panel_label(ax, "b", dx=-0.20, dy=1.04)

    for n, value in zip(sizes, changed):
        rows.append(["b", "changed", str(n), value])

    with open(os.path.join(OUT, "fig_tier4_replay.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "item", "value"])
        writer.writerows(rows)

    export(fig, "fig_tier4_replay", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
