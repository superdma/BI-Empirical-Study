"""Homepage figure: how many of 32 prompts stay bitwise identical per engine.

Both panels read the same definition of a pass: token ids equal and every
logprob bitwise equal. Offline counts come from the 32-prompt solo-vs-batch
files; online counts come from the sequential-vs-concurrent pairs.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")


def passes(path, key_filter=None):
    with open(path) as handle:
        cmp = json.load(handle)["cmp"]
    if key_filter:
        cmp = {k: v for k, v in cmp.items() if key_filter in k}
    good = sum(1 for v in cmp.values() if v.get("ids") and v.get("bitwise"))
    return good, len(cmp)


def main():
    setup()
    rows = [
        ("vLLM compiled", "vllm",
         passes(f"{ROOT}/results/http/concurrency/bi_http_batchall_compiled_pc1.json"),
         passes(f"{ROOT}/results/http/concurrency/bi_http_conc_compiled_pc1.json", "seq_vs_concurrent")),
        ("vLLM eager", "vllm_light",
         passes(f"{ROOT}/results/http/concurrency/bi_http_batchall_eager_pc0.json"),
         passes(f"{ROOT}/results/http/concurrency/bi_http_conc_eager_pc0.json", "seq_vs_concurrent")),
        ("SGLang TP=4", "sgl_light",
         passes(f"{ROOT}/sglang/results/sgl_det_batchall.json"),
         passes(f"{ROOT}/sglang/results/sgl_det_http_http.json", "seq_vs_concurrent")),
        ("SGLang PP=4", "sgl",
         passes(f"{ROOT}/sglang/results/sgl_det_pp4_batchall.json"),
         passes(f"{ROOT}/sglang/results/sgl_det_pp4_http_http.json", "seq_vs_concurrent")),
    ]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fig_overview.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "offline_pass", "offline_total",
                         "online_pass", "online_total"])
        for name, _, off, on in rows:
            writer.writerow([name, off[0], off[1], on[0], on[1]])

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.95), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    titles = ["离线：32 条 prompt，单独 vs 批内", "在线：32 路并发 vs 顺序基线"]
    for ax, idx, letter, title in zip(axes, (2, 3), "ab", titles):
        labels = [r[0] for r in rows]
        vals = [r[idx][0] for r in rows]
        totals = [r[idx][1] for r in rows]
        colors = [C[r[1]] for r in rows]
        y = range(len(rows))
        ax.barh(y, totals, color=C["zero"], height=0.62, zorder=1)
        ax.barh(y, vals, color=colors, height=0.62, zorder=2)
        for i, (v, t) in enumerate(zip(vals, totals)):
            ax.text(t + 0.6, i, f"{v}/{t}", va="center", ha="left",
                    fontsize=7.5, color=C["ink"])
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_xlabel("位级一致的请求数")
        ax.set_title(title, loc="left", pad=6)
        tidy(ax, "x")
        panel_label(ax, letter, dx=-0.30 if letter == "a" else -0.10, dy=1.06)
    axes[0].invert_yaxis()
    export(fig, "fig_overview", OUT)


if __name__ == "__main__":
    main()
