"""v1 online results: how many of the six comparisons pass per configuration.

Bars are split into passing (teal) and failing (brick) comparisons; the note on
the right names the worst failing value. Data comes from results/http/*.json.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")

CONFIGS = [
    ("eager 8192 / PC1 #1", "bi_http_eager_8192_pc1_run1.json"),
    ("eager 8192 / PC1 #2", "bi_http_eager_8192_pc1_run2.json"),
    ("eager 2048 / PC1", "bi_http_eager_2048_pc1.json"),
    ("eager 8192 / PC0", "bi_http_eager_8192_pc0.json"),
    ("BI=0（eager 8192 / PC1）", "bi_http_bi0_eager_8192_pc1.json"),
    ("compiled 8192 / PC1", "bi_http_compiled_8192_pc1.json"),
]
KEYS = [
    ("bs=1 vs bs=32", "server_bs1_vs_bs32"),
    ("逆序", "server_bs1_vs_reverse_order"),
    ("短 prompt 缓存", "server_short_fresh_vs_cache_hit"),
    ("长 prompt bs=1 vs 4", "server_long_bs1_vs_bs4_fresh"),
    ("长 prompt 缓存命中", "server_long_fresh_vs_cache_hit"),
    ("decode vs re-feed", "server_decode_vs_refeed"),
]


def main():
    setup()
    rows = []
    for label, name in CONFIGS:
        with open(os.path.join(ROOT, "results", "http", name)) as handle:
            comp = json.load(handle)["comparisons"]
        cells = []
        for short, key in KEYS:
            v = comp.get(key, {})
            ok = v.get("token_logprobs_bitwise_equal", v.get("all_bitwise_equal"))
            mx = v.get("max_abs_logprob_diff", v.get("max_abs_diff")) or 0.0
            cells.append((short, bool(ok), float(mx)))
        rows.append((label, cells))

    with open(os.path.join(OUT, "fig_v1_online.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration"] + [k for k, _ in KEYS])
        for label, cells in rows:
            writer.writerow([label] + [("pass" if ok else f"{mx:.3e}") for _, ok, mx in cells])

    fig, ax = plt.subplots(figsize=(WIDTH, 1.95))
    fig.subplots_adjust(left=0.30, right=0.72, top=0.90, bottom=0.20)

    for i, (label, cells) in enumerate(rows):
        y = len(rows) - 1 - i
        n_pass = sum(1 for _, ok, _ in cells if ok)
        n_fail = len(cells) - n_pass
        ax.barh(y, n_pass, color=C["pass"], height=0.6, zorder=2)
        ax.barh(y, n_fail, left=n_pass, color=C["fail"], height=0.6, zorder=2)
        ax.text(len(cells) + 0.15, y, f"{n_pass}/{len(cells)}", va="center",
                fontsize=7, color=C["ink"])
        if n_fail:
            worst = max(mx for _, ok, mx in cells if not ok)
            val = f"{worst:.3f}" if worst >= 1e-3 else f"{worst:.2e}"
            if n_fail == 1:
                note = f"失败项最大 {val}"
            else:
                note = f"{n_fail} 项失败，最大 {val}"
            ax.text(len(cells) + 0.95, y, note, va="center", fontsize=6.3,
                    color=C["fail"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows][::-1], fontsize=7)
    ax.set_xlim(0, 6)
    ax.set_xticks(range(7))
    ax.set_xlabel("6 项对比中通过的数量（绿）与失败的数量（红）", fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    tidy(ax, "x")
    export(fig, "fig_v1_online", OUT, multipanel=False)


if __name__ == "__main__":
    main()
