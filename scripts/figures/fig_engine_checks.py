"""Three engine-level checks that decide which mechanism explanations survive.

Panel a pins the NCCL algorithm and reruns the 32-prompt comparison; panel b
sweeps the cached tail length; panel c walks batch sizes across CUDA-graph
bucket edges in compiled and eager mode.  Data comes from the tier-1 runs.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "tier1")
OUT = os.path.join(ROOT, "docs", "assets", "figs")

NCCL = [("不指定", "bi_nccl_default.json"),
        ("allreduce:ring", "bi_nccl_ring.json"),
        ("allreduce:tree", "bi_nccl_tree.json")]
BUCKET = [("compiled", "bi_bucket_scan.json", C["pass"], "s"),
          ("eager", "bi_bucket_scan_eager.json", C["fail"], "o")]


def load(name):
    with open(os.path.join(RES, name)) as handle:
        return json.load(handle)


def main():
    setup()
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22, wspace=0.42)

    # a: pinning NCCL_ALGO barely moves the engine
    ax = axes[0]
    for i, (label, fname) in enumerate(NCCL):
        d = load(fname)
        cmp = d["cmp"]
        good = sum(1 for v in cmp.values() if v["ids"] and v["bitwise"])
        y = len(NCCL) - 1 - i
        ax.barh(y, good, color=C["pass"], height=0.6, zorder=2)
        ax.barh(y, len(cmp) - good, left=good, color=C["zero"], height=0.6, zorder=2)
        mx = max((v["max"] or 0.0) for v in cmp.values())
        ax.text(len(cmp) + 0.8, y, f"{good}/{len(cmp)}", va="center", fontsize=6.6)
        rows.append(["a", label, good, len(cmp), mx])
    ax.set_yticks(range(len(NCCL)))
    ax.set_yticklabels([n[0] for n in NCCL][::-1], fontsize=6.8)
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 16, 32])
    ax.set_xlabel("32 条中位级一致的条数")
    ax.set_title("钉死 NCCL 算法", loc="left", fontsize=7.6, pad=6)
    ax.set_ylim(-0.6, len(NCCL) - 0.4)
    tidy(ax, "x")
    panel_label(ax, "a", dx=-0.30, dy=1.04)

    # b: tail length is not monotone
    ax = axes[1]
    tail = load("bi_tail_scan.json")
    xs = [r["tail"] for r in tail["rows"]]
    ys = [max(r["max_abs"] or 0.0, 1e-9) for r in tail["rows"]]
    for r, x, y in zip(tail["rows"], xs, ys):
        ok = r["bitwise"]
        ax.scatter([x], [y], s=30, zorder=3,
                   facecolor=C["pass"] if ok else C["fail"],
                   edgecolor="white" if ok else C["fail"])
        rows.append(["b", x, int(ok), y if not ok else 0.0, r["first_mismatch"]])
    ax.axhline(1e-9, color=C["grid"], linewidth=0.6)
    ax.text(20, 6e-8, "其余 tail 全部为 0.0（位级一致）", fontsize=6.2, color=C["pass"])
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_ylim(-1e-9, 3e-6)
    ax.set_xlim(0, 540)
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.set_xlabel("tail token 数")
    ax.set_ylabel("最大 |Δlogprob|")
    ax.set_title("缓存命中 vs fresh", loc="left", fontsize=7.6, pad=6)
    tidy(ax)
    panel_label(ax, "b", dx=-0.20, dy=1.04)

    # c: bucket edges are not the boundary
    ax = axes[2]
    for label, fname, color, marker in BUCKET:
        d = load(fname)
        xs = [r["bs"] for r in d["rows"]]
        ys = [max(r["max_abs"] or 0.0, 1e-9) for r in d["rows"]]
        ax.plot(xs, ys, color=color, linewidth=0.8, marker=marker, markersize=3.4,
                markerfacecolor="white", markeredgecolor=color, label=label, zorder=3)
        for r in d["rows"]:
            rows.append(["c", label, r["bs"], int(bool(r["bitwise"])),
                         r["max_abs"] or 0.0])
    ax.axvline(255.5, color=C["muted"], linestyle=(0, (2, 2)), linewidth=0.7)
    ax.text(252, 2.2e-8, "255 | 256", ha="right", va="center", fontsize=6.2,
            color=C["muted"])
    ax.annotate("8.805e-02", xy=(256, 8.805e-2), xytext=(150, 4e-2), fontsize=6.2,
                color=C["fail"], ha="right",
                arrowprops=dict(arrowstyle="-", color=C["fail"], linewidth=0.5,
                                shrinkB=3))
    ax.text(28, 3e-7, "compiled 全部为 0.0", fontsize=6.2, color=C["pass"])
    ax.set_yscale("log")
    ax.set_ylim(5e-10, 1.6)
    ax.set_xlim(20, 300)
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xlabel("批大小")
    ax.set_ylabel("最大 |Δlogprob|")
    ax.set_title("bucket 边界 vs 256", loc="left", fontsize=7.6, pad=6)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.2,
              handlelength=1.6, handletextpad=0.5, borderpad=0.2, labelspacing=0.3)
    tidy(ax)
    panel_label(ax, "c", dx=-0.20, dy=1.04)

    with open(os.path.join(OUT, "fig_engine_checks.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "x", "pass", "value"])
        writer.writerows(rows)

    export(fig, "fig_engine_checks", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
