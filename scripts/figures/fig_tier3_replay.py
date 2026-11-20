"""Replaying the first divergent norm, and what TP=2 changes.

Panel a: the engine's own 256-row input, replayed outside the engine under
four arrangements (intact, cut to 255, the 255 input, 255 padded back to 256).
Every one of them lands on the 255 output; none reproduces the 256 output.

Panel b: the 32-prompt test at TP=4 and TP=2, plus the repeated batch.

Data: results/tier3/bi_replay_rmsnorm.json, bi_tp2_matrix.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier3")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

LABELS = {
    "engine256_full": "256 的输入\n原样",
    "engine256_first255": "256 的输入\n截到 255",
    "engine255_full": "255 的输入\n原样",
    "engine255_pad256": "255 的输入\n补到 256",
}


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def main():
    setup()
    replay = load("bi_replay_rmsnorm.json")
    tp2 = load("bi_tp2_matrix.json")
    tp4 = {"solo_vs_batch_32": (1, 32), "batch_repeat_32": (32, 32)}

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.24, wspace=0.34)
    rows = []

    # a: replay cases as a small matrix
    ax = axes[0]
    cases = replay["cases"]
    data = [[c["vs_engine256"] for c in cases], [c["vs_engine255"] for c in cases]]
    for i in range(2):
        for j in range(len(cases)):
            color = C["fail"] if data[i][j] else C["pass"]
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                                       facecolor=color, alpha=0.16 if data[i][j] else 0.12,
                                       edgecolor="white", linewidth=1.2, zorder=2))
            ax.text(j, i, str(data[i][j]), ha="center", va="center", fontsize=7.0,
                    color=C["ink"], zorder=3)
            rows.append(["a", "vs_engine256" if i == 0 else "vs_engine255",
                         cases[j]["label"], data[i][j]])
    ax.set_xlim(-0.6, len(cases) - 0.4)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([LABELS.get(c["label"], c["label"]) for c in cases], fontsize=5.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["与 256\n引擎输出", "与 255\n引擎输出"], fontsize=5.9)
    ax.set_title("同一份输入，回放落在哪一边", loc="left", fontsize=7.6, pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, "a", dx=-0.12, dy=1.03)

    # b: TP=4 vs TP=2 as a matrix
    ax = axes[1]
    groups = [("32 条：单独 vs 批内", "solo_vs_batch_32"),
              ("32 条：批请求重复", "batch_repeat_32")]
    tp2c = tp2["cmp"]
    mat = [[tp4["solo_vs_batch_32"][0] / 32 * 100, tp4["batch_repeat_32"][0] / 32 * 100],
           [tp2c["solo_vs_batch_32"]["bitwise_equal"] / tp2c["solo_vs_batch_32"]["n"] * 100,
            tp2c["batch_repeat_32"]["bitwise_equal"] / tp2c["batch_repeat_32"]["n"] * 100]]
    for i in range(2):
        for j in range(2):
            value = mat[i][j]
            color = C["pass"] if value > 99 else (C["fail"] if value < 60 else C["warn"])
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=color,
                                       alpha=0.16, edgecolor="white", linewidth=1.2, zorder=2))
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=7.0,
                    color=C["ink"], zorder=3)
            rows.append(["b", "TP=4" if i == 0 else "TP=2", groups[j][1], round(value, 1)])
    ax.set_xlim(-0.6, 1.4)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TP=4", "TP=2"], fontsize=6.4)
    ax.set_title("32 条 prompt：TP=4 与 TP=2", loc="left", fontsize=7.6, pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, "b", dx=-0.12, dy=1.03)

    with open(os.path.join(OUT, "fig_tier3_replay.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "item", "value"])
        writer.writerows(rows)

    export(fig, "fig_tier3_replay", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
