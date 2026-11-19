"""How far the cross-GPU reduction itself depends on message size.

Panel a walks the batch size one row at a time and counts how many of the
unchanged rows change bits; panel b compares bs=1 against much larger batches
and adds an all-positive payload as a control for cancellation.  Data comes
from the model-free NCCL probe in bi_allreduce_bench.py.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier1")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

THRESH = [
    ("TP=4，自动选择", "bi_ar_thresh_w4.json", C["fail"], "o"),
    ("TP=4，allreduce:tree", "bi_ar_thresh_w4_tree.json", C["warn"], "s"),
    ("TP=2，allreduce:ring", "bi_ar_thresh_w2.json", C["pass"], "^"),
]
SPARSE = [
    ("TP=4，自动选择", "bi_ar_sparse_w4.json", C["fail"], "o"),
    ("TP=4，allreduce:tree", "bi_ar_sparse_w4_tree.json", C["warn"], "s"),
]


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def rows_of(data, dtype, test):
    return [r for r in data["rows"] if r["dtype"] == dtype and r.get("test", "reshape") == test]


def main():
    setup()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(left=0.175, right=0.985, top=0.86, bottom=0.20, wspace=0.34)

    ax = axes[0]
    counts = []
    for i, (label, fname, color, marker) in enumerate(THRESH):
        data = load(fname)
        pts = [(r["pair"][1], r["n_diff"]) for r in rows_of(data, "bfloat16", "threshold")]
        bad = sum(1 for _, v in pts if v)
        y = len(THRESH) - 1 - i
        ax.barh(y, bad, color=color, height=0.6, zorder=2)
        ax.text(bad + 5, y, f"{bad} / {len(pts)}", va="center", fontsize=6.8,
                color=C["ink"] if bad == 0 else C["fail"])
        rows.append(["a", label, len(pts), bad])
        counts.append(bad)
    ax.set_yticks(range(len(THRESH)))
    ax.set_yticklabels([t[0] for t in THRESH][::-1], fontsize=6.8)
    ax.set_xlim(0, 340)
    ax.set_xticks([0, 100, 200, 299])
    ax.set_xlabel("299 步里出现位差的步数")
    ax.set_title("批大小每加 1，未变的行还不变吗", loc="left", fontsize=7.6, pad=6)
    ax.set_ylim(-0.6, len(THRESH) - 0.4)
    tidy(ax, "x")
    panel_label(ax, "a", dx=-0.30, dy=1.04)

    ax = axes[1]
    for label, fname, color, marker in SPARSE:
        data = load(fname)
        pts = [(r["pair"][1], r["n_diff"]) for r in rows_of(data, "bfloat16", "sparse")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=0.8,
                marker=marker, markersize=3.2, markerfacecolor="white",
                markeredgecolor=color, label=label, zorder=3)
        rows.append(["b", label, len(pts), sum(1 for _, v in pts if v)])
    pos = rows_of(load("bi_ar_pos_w4.json"), "bfloat16", "sparse_pos")
    ax.plot([r["pair"][1] for r in pos], [r["n_diff"] for r in pos], color=C["vllm"],
            linewidth=0.8, linestyle=(0, (3, 2)), marker="D", markersize=2.8,
            markerfacecolor="white", markeredgecolor=C["vllm"],
            label="TP=4，自动选择，全正数据", zorder=3)
    rows.append(["b", "TP=4，自动选择，全正数据", len(pos), sum(1 for r in pos if r["n_diff"])])
    ax.axhline(0, color=C["pass"], linewidth=0.9)
    ax.text(140, -170, "TP=2 的全部比较都落在 0（0 / 2048）", fontsize=6.2,
          color=C["pass"])
    ax.set_xlabel("与 bs=1 同批的请求数")
    ax.set_ylabel("行 0 中位不同的元素数")
    ax.set_xlim(0, 310)
    ax.set_ylim(-260, 1150)
    ax.set_title("bs=1 与更大批次的行 0 比较", loc="left", fontsize=7.6, pad=6)
    ax.legend(loc="upper right", fontsize=6.0, handlelength=1.8, handletextpad=0.5,
              borderpad=0.2, labelspacing=0.3)
    tidy(ax)
    panel_label(ax, "b", dx=-0.115, dy=1.04)

    with open(os.path.join(OUT, "fig_ar_sensitivity.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "steps", "steps_with_diff"])
        writer.writerows(rows)

    export(fig, "fig_ar_sensitivity", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
