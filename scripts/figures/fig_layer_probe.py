"""Where the 255-vs-256 difference first becomes visible.

Panel a is a step-by-layer map of how many of the 2048 elements in row 0
change bits; panel b counts, per decode step, how many of the 48 layers are
already affected.  Both come from the per-layer dump that bi_layer_probe.py
writes from inside the four workers.

Data: results/tier2/bi_layer_map.json.
"""
import csv
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier2")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

FIRST_STEP, FIRST_LAYER = 4, 42


def main():
    setup()
    with open(os.path.join(RES, "bi_layer_map.json")) as handle:
        grid = np.array(json.load(handle)["grid"], dtype=float)
    steps, layers = grid.shape
    rows = []

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.20, wspace=0.34)

    ax = axes[0]
    shown = np.where(grid > 0, grid, np.nan)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "bi", ["#f6d9cf", C["fail"]])
    cmap.set_bad("#eef0f2")
    im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0, vmax=grid.max(),
                   interpolation="nearest", origin="lower")
    ax.set_xlabel("层号")
    ax.set_ylabel("解码步")
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.set_yticks(range(steps))
    ax.set_title("行 0 中改位的元素数（灰 = 0）", loc="left", fontsize=7.4, pad=6)
    ax.add_patch(plt.Rectangle((FIRST_LAYER - 0.5, FIRST_STEP - 0.5), 1, 1,
                               fill=False, edgecolor=C["ink"], linewidth=0.9))
    ax.annotate("第 5 步\n第 42 层", xy=(FIRST_LAYER, FIRST_STEP),
                xytext=(FIRST_LAYER - 19, FIRST_STEP - 2.6), fontsize=6.2,
                color=C["ink"],
                arrowprops=dict(arrowstyle="-", color=C["ink"], linewidth=0.7))
    for s in range(steps):
        for l in range(layers):
            rows.append(["a", s, l, int(grid[s, l])])
    panel_label(ax, "a", dx=-0.10, dy=1.04)

    ax = axes[1]
    counts = (grid > 0).sum(axis=1)
    ax.bar(range(steps), counts, color=C["fail"], width=0.62, zorder=2)
    for s, v in enumerate(counts):
        if v:
            ax.text(s, v + 1.5, str(int(v)), ha="center", fontsize=6.2,
                    color=C["ink"])
    ax.set_xticks(range(steps))
    ax.set_xlabel("解码步")
    ax.set_ylabel("已有差异的层数")
    ax.set_ylim(0, 56)
    ax.set_yticks([0, 24, 48])
    ax.set_title("差异扩散到多少层", loc="left", fontsize=7.4, pad=6)
    ax.grid(False)
    for s, v in enumerate(counts):
        rows.append(["b", s, "layers", int(v)])
    panel_label(ax, "b", dx=-0.30, dy=1.04)

    with open(os.path.join(OUT, "fig_layer_probe.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "x", "y", "value"])
        writer.writerows(rows)

    export(fig, "fig_layer_probe", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
