"""SGLang online results: PP=4 passes all ninety-six comparisons, at a cost.

Panel a counts how many of the 32 concurrent requests reproduce the sequential
baseline for each scenario; panel b shows the wall-clock time of one 400-token
request, the price of turning TP=4 into PP=4.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "sglang", "results")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

CONFIGS = [
    ("TP=4，确定性", "sgl_det_http_http", C["fail"]),
    ("TP=4，关闭确定性", "sgl_nodet_http_http", C["zero"]),
    ("PP=4，确定性", "sgl_det_pp4_http_http", C["pass"]),
]
SCENES = [
    ("顺序 vs 并发", "concurrent"),
    ("相同 prompt 并发", "identical"),
    ("顺序 vs 突发", "burst"),
]


def load(stem):
    with open(os.path.join(RES, stem + ".json")) as handle:
        return json.load(handle)


def pass_count(data, suffix):
    got = [v for k, v in data["cmp"].items()
           if ("seq_vs_" + suffix) in k or k.startswith(suffix)]
    return sum(1 for v in got if v["bitwise"]), len(got)


def main():
    setup()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.78, bottom=0.20, wspace=0.44)

    # a: 3x3 count matrix, scenarios by configuration
    ax = axes[0]
    grid = []
    for i, (scene, suffix) in enumerate(SCENES):
        row = []
        for j, (label, stem, color) in enumerate(CONFIGS):
            good, n = pass_count(load(stem), suffix)
            row.append((good, n))
            rows.append(["a", label, scene, good, n])
        grid.append(row)
    for i, row in enumerate(grid):
        for j, (good, n) in enumerate(row):
            if good == n:
                face, edge = "#cfe4e0", C["pass"]
            elif good == 0:
                face, edge = "#f7e4de", C["fail"]
            else:
                face, edge = "#f0c9bd", C["fail"]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                       edgecolor="white", linewidth=1.4))
            ax.text(j, i, f"{good}", ha="center", va="center", fontsize=7.4,
                    color=C["ink"])
    ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
    ax.set_ylim(len(SCENES) - 0.5, -0.5)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([c[0].replace("，", "\n") for c in CONFIGS], fontsize=6.4)
    ax.set_yticks(range(len(SCENES)))
    ax.set_yticklabels([s[0] for s in SCENES], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("每场景 32 条中位级一致的条数", loc="left", fontsize=7.2, pad=8)
    panel_label(ax, "a", dx=-0.30, dy=1.03)

    # b: latency of one 400-token request
    ax = axes[1]
    for i, (label, stem, color) in enumerate(CONFIGS):
        sec = load(stem)["env"]["long_sec"]
        y = len(CONFIGS) - 1 - i
        ax.barh(y, sec, color=color, height=0.55, zorder=2)
        ax.text(sec + 0.35, y, f"{sec:.2f} s", va="center", fontsize=6.8)
        rows.append(["b", label, "long_sec", sec, None])
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels([c[0] for c in CONFIGS][::-1], fontsize=7)
    ax.set_xlim(0, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("单个 400-token 请求的墙钟时间")
    ax.set_title("PP=4 的代价", loc="left", fontsize=7.6, pad=6)
    ax.set_ylim(-0.6, len(CONFIGS) - 0.4)
    ax.grid(False)
    panel_label(ax, "b", dx=-0.30, dy=1.03)

    with open(os.path.join(OUT, "fig_sgl_online.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "configuration", "item", "value", "total"])
        writer.writerows(rows)

    export(fig, "fig_sgl_online", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
