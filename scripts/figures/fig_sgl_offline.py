"""SGLang offline results: which parallelism keeps bitwise equality.

Both panels count the 32 prompts whose logprobs match a solo run bit for bit.
Panel a varies the parallel layout and attention backend on the 30B MoE model;
panel b repeats the layout sweep on the dense 0.6B model, where TP=1 and TP=2
pass and TP=4 does not. Values come from sglang/results/*.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "sglang", "results")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

R30 = [
    ("TP=4，确定性", "sgl_det_batchall"),
    ("TP=4，关 CUDA graph", "sgl_det_nograph_batchall"),
    ("TP=4，关 custom all-reduce", "sgl_det_dcar_batchall"),
    ("TP=4，+ triton attention", "sgl_det_triton_batchall"),
    ("TP=4，关闭确定性", "sgl_nodet_batchall"),
    ("PP=4，确定性", "sgl_det_pp4_batchall"),
]
R06 = [
    ("TP=1", "smoke_det_batchall"),
    ("TP=2", "smoke_det_tp2_batchall"),
    ("TP=4", "smoke_det_tp4_batchall"),
    ("TP=4，+ triton", "smoke_det_tp4_triton_batchall"),
    ("PP=4", "smoke_det_pp4_batchall"),
]


def pass_count(stem):
    with open(os.path.join(RES, stem + ".json")) as handle:
        s = json.load(handle)["summary"]["solo_vs_batch"]
    return s["n"] - s["bad"], s["n"]


def main():
    setup()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0))
    fig.subplots_adjust(left=0.20, right=0.90, top=0.86, bottom=0.22, wspace=0.55)

    panels = [
        (axes[0], R30, "a", "30B（MoE）", -0.72),
        (axes[1], R06, "b", "0.6B（稠密）", -0.52),
    ]
    for ax, entries, letter, title, dx in panels:
        n_total = 32
        for i, (label, stem) in enumerate(entries):
            good, n = pass_count(stem)
            y = len(entries) - 1 - i
            ax.barh(y, good, color=C["pass"], height=0.62, zorder=2)
            ax.barh(y, n - good, left=good, color=C["zero"], height=0.62, zorder=2)
            ax.text(n_total + 0.8, y, f"{good}/{n}", va="center", fontsize=6.8,
                    color=C["ink"] if good == n else C["muted"])
            rows.append([letter, label, good, n])
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([e[0] for e in entries][::-1], fontsize=6.8)
        ax.set_xlim(0, n_total + 8)
        ax.set_xticks([0, 16, 32])
        ax.set_xlabel("32 条 prompt 中位级一致的条数")
        ax.set_title(title, loc="left", fontsize=7.4, pad=8)
        ax.set_ylim(-0.6, len(entries) - 0.4)
        tidy(ax, "x")
        panel_label(ax, letter, dx=dx, dy=1.04)

    with open(os.path.join(OUT, "fig_sgl_offline.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "pass", "total"])
        writer.writerows(rows)

    export(fig, "fig_sgl_offline", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
