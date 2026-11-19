"""v1 offline results as a pass matrix: ten configurations by four tests.

A cell is green when the logprobs match bit for bit; otherwise it carries the
largest absolute logprob difference. Values come straight from results/*.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figstyle import C, WIDTH, export, setup

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")

ROWS = [
    ("compiled 8192 / PC1（冷）", "bi_results_chunk8192_pc1.json"),
    ("compiled 8192 / PC1（warm #2）", "bi_compiled_warm_2.json"),
    ("compiled 8192 / PC1（warm #3）", "bi_compiled_warm_3.json"),
    ("compiled 2048 / PC1（冷）", "bi_results_chunk2048_pc1.json"),
    ("compiled 8192 / PC0（冷）", "bi_results_chunk8192_pc0.json"),
    ("eager 8192 / PC1 #1", "bi_eager_1.json"),
    ("eager 8192 / PC1 #2", "bi_eager_2.json"),
    ("eager 2048 / PC1", "bi_eager_chunk2048.json"),
    ("eager 8192 / PC0", "bi_eager_pc0.json"),
    ("BI=0 对照（compiled 8192 / PC1）", "bi_off_chunk8192_pc1.json"),
    ("TP=1 Qwen3-0.6B 冒烟", "bi_smoke_qwen3_06b.json"),
]
COLS = [
    ("bs=1 vs bs=32", "bs1_vs_bs32"),
    ("逆序", "bs1_vs_reverse_order"),
    ("前缀缓存命中", "bs1_vs_repeat_prefix_cache"),
    ("decode vs re-feed", "decode_vs_refeed"),
]


def cell(path, key):
    with open(path) as handle:
        v = json.load(handle)["comparisons"][key]
    ok = v.get("logprobs_bitwise_equal", v.get("all_bitwise_equal"))
    mx = v.get("max_abs_logprob_diff", v.get("max_abs_diff")) or 0.0
    return bool(ok), float(mx)


def main():
    setup()
    grid = [[cell(os.path.join(ROOT, "results", f), key) for _, key in COLS] for _, f in ROWS]

    with open(os.path.join(OUT, "fig_v1_matrix.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration"] + [name for name, _ in COLS])
        for (label, _), row in zip(ROWS, grid):
            writer.writerow([label] + [("pass" if ok else f"{mx:.3e}") for ok, mx in row])

    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    ok_mask = np.array([[ok for ok, _ in row] for row in grid])
    ax.imshow(ok_mask.astype(float), cmap=ListedColormap(["#f4d6cd", "#cfe4e0"]),
              vmin=-0.5, vmax=1.5, aspect="auto")
    for i, row in enumerate(grid):
        for j, (ok, mx) in enumerate(row):
            ax.text(j, i, "0.0" if ok else f"{mx:.3e}", ha="center", va="center",
                    fontsize=6.6 if ok else 6.2,
                    color=C["ink"] if ok else C["fail"],
                    fontweight="normal" if ok else "bold")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([name for name, _ in COLS], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{chr(65 + i)}. {label}" if False else label for i, (label, _) in enumerate(ROWS)],
                       fontsize=6.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False, length=0)
    ax.tick_params(which="major", length=0)
    ax.set_xlabel("绿色：logprob 逐位相同；浅红：给出最大 |Δlogprob|", fontsize=6.6,
                  color=C["muted"], labelpad=8)
    export(fig, "fig_v1_matrix", OUT, multipanel=False)


if __name__ == "__main__":
    main()
