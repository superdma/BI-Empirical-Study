"""What changes when the batch changes, and where the kernel switches.

Panel a: bitwise-equal counts for four ways of touching a 255/256 batch.
Order and a transient extra request sit at the noise floor; only the number
of active sequences moves a quarter of the prompts.

Panel b: block size of the fused add+norm kernel against the number of rows,
measured twice: from the engine traces and from standalone calls. Both switch
at exactly 256 rows.

Data: results/tier3/bi_composition.json, bi_launch_blocks.json,
bi_block_probe.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier3")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

CASES = [
    ("c255_vs_c256", "255 条\nvs 256 条", C["fail"]),
    ("c255_vs_c255b", "255 条\n重复", C["pass"]),
    ("s255_vs_c255", "255 条 + 1 条\n短请求", C["warn"]),
    ("r256_vs_c256", "256 条\n逆序", C["warn"]),
]


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def main():
    setup()
    comp = load("bi_composition.json")
    blocks = load("bi_launch_blocks.json")
    probe = load("bi_block_probe.json")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.24, wspace=0.30)
    rows = []

    # a: bitwise-equal counts
    ax = axes[0]
    labels, vals, totals, colors = [], [], [], []
    for key, label, color in CASES:
        v = comp["cmp"][key]
        labels.append(label)
        vals.append(v["bitwise_equal"] / v["n"] * 100)
        totals.append((v["bitwise_equal"], v["n"]))
        colors.append(color)
        rows.append(["a", key, v["bitwise_equal"], v["n"]])
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.58, zorder=2)
    for i, (bar, (eq, total)) in enumerate(zip(bars, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f"{eq}/{total}", ha="center", fontsize=6.3, color=C["ink"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.1)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("位级一致的条数（%）")
    ax.set_title("改什么才会改变结果", loc="left", fontsize=7.6, pad=6)
    tidy(ax, "y")
    panel_label(ax, "a", dx=-0.16, dy=1.03)

    # b: block size against rows
    ax = axes[1]
    pts = []
    for n, value in probe["rows_to_launch"].items():
        pts.append((int(n), value[1][0], "probe"))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C["vllm"],
            linewidth=1.1, marker="o", markersize=3.2, zorder=3,
            label="独立调用")
    engine = []
    for tag, label, color in (("p255", "255 条（引擎）", C["sgl"]),
                              ("p256", "256 条（引擎）", C["fail"])):
        for pair in blocks[tag]["pairs"]:
            if pair["grid"] in (255, 256):
                engine.append((pair["grid"], pair["block"], label, color))
    for x, y, label, color in engine:
        ax.scatter([x], [y], s=26, color=color, zorder=4, marker="s",
                   edgecolor="white", linewidth=0.5)
        rows.append(["b", label, x, y])
    ax.axvline(256, color=C["muted"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
    ax.annotate("256", xy=(256, 640), xytext=(-5, 0), textcoords="offset points",
                fontsize=6.2, color=C["muted"], ha="right", va="center")
    ax.set_xscale("log", base=2)
    ax.set_xlim(4, 5000)
    ax.set_ylim(180, 1100)
    ax.set_yticks([256, 1024])
    ax.set_xlabel("行数（对数轴）")
    ax.set_ylabel("block 大小")
    ax.set_title("fused add+norm 的 block 在 256 处切换", loc="left", fontsize=7.6, pad=6)
    ax.text(6, 1035, "独立调用", fontsize=5.9, color=C["vllm"], va="bottom")
    ax.text(246, 985, "引擎 255 条", fontsize=5.9, color=C["sgl"], ha="right", va="top")
    ax.text(268, 300, "引擎 256 条", fontsize=5.9, color=C["fail"], va="bottom")
    tidy(ax, "x")
    panel_label(ax, "b", dx=-0.16, dy=1.03)

    with open(os.path.join(OUT, "fig_tier3_composition.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "item", "value"])
        writer.writerows(rows)

    export(fig, "fig_tier3_composition", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
