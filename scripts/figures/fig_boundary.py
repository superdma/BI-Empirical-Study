"""Where eager batch invariance breaks: batch size, prefill shape, re-feed length.

Panel a scans the decode batch size against one 5-token target, panel b plots
in-batch prefill shapes as pass fraction, panel c sweeps how many decoded
tokens are re-fed as a prompt.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from figstyle import C, WIDTH, export, panel_label, setup, tidy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets", "figs")

BS_FILES = [(128, "bi_diag_bs128_pc0"), (192, "bi_diag_bs192_pc0"),
            (224, "bi_diag_bs224_pc0"), (240, "bi_diag_bs240_pc0"),
            (255, "bi_diag_bs255_pc0"), (256, "bi_diag_bs256_pc0"),
            (257, "bi_diag_bs257_mns512_pc0")]
LEN_TAGS = ["n32_l5", "n32_l8", "n32_l9", "n16_l30", "n8_l40"]


def comparison_value(v):
    ok = v.get("tok_logprobs_ok", v.get("logprobs_bitwise_equal"))
    if ok is None:
        ok = v.get("token_ids_equal") and v.get("logprobs_bitwise_equal")
    mx = v.get("max_abs_logprob_diff", v.get("max")) or 0.0
    return bool(ok), float(mx)


def main():
    setup()
    rows = []

    sweep = []
    for bs, stem in BS_FILES:
        with open(os.path.join(ROOT, "results", "extra", stem + ".json")) as handle:
            v = json.load(handle)["comparisons"]["single_vs_batch"]
        sweep.append((bs, bool(v["logprobs_bitwise_equal"]), float(v["max_abs_logprob_diff"])))
        rows.append(["sweep", bs, int(bool(v["logprobs_bitwise_equal"])), f"{v['max_abs_logprob_diff']:.3e}"])

    len_points = {"eager": [], "compiled": []}
    for tag in LEN_TAGS:
        for mode in ("eager", "compiled"):
            path = os.path.join(ROOT, "results", "http", "concurrency", f"bi_http_len_{mode}_{tag}.json")
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                data = json.load(handle)
            env = data["env"]
            good = sum(1 for v in data["cmp"].values() if v.get("ids") and v.get("bitwise"))
            n = len(data["cmp"])
            len_points[mode].append((env["total_prompt_tokens"], good / n, n,
                                     env.get("L", tag)))
            rows.append(["len", mode, tag, env["total_prompt_tokens"], good, n])

    refeed = {"eager": [], "compiled": []}
    for mode in ("eager", "compiled"):
        for mt in (64, 128, 192, 256):
            path = os.path.join(ROOT, "results", "extra", f"bi_refeed_{mode}_pc0_mt{mt}.json")
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                pairs = json.load(handle)["refeeds"]["1"]["pairs"]
            bad = sum(1 for p in pairs if not p["bitwise_equal"])
            refeed[mode].append((mt, bad / len(pairs)))
            rows.append(["refeed", mode, mt, bad, len(pairs)])

    with open(os.path.join(OUT, "fig_boundary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "x", "value_a", "value_b", "value_c"])
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.15))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.20, wspace=0.42)

    # a: decode batch size
    ax = axes[0]
    xs = [s[0] for s in sweep]
    ys = [max(s[2], 1e-8) if not s[1] else 1e-8 for s in sweep]
    ax.set_yscale("log")
    for (bs, ok, mx) in sweep:
        y = mx if mx > 0 else 1e-8
        ax.scatter([bs], [y], s=26, zorder=3,
                   facecolor=C["pass"] if ok else C["fail"],
                   edgecolor="white" if ok else C["fail"])
    ax.axvline(255.5, color=C["muted"], linestyle=(0, (2, 2)), linewidth=0.7)
    ax.text(252.5, 3e-7, "255 | 256", ha="right", va="center", fontsize=6.2, color=C["muted"])
    ax.annotate("8.805e-02", xy=(256, 8.805e-2), xytext=(198, 1.5e-1),
                fontsize=6.2, color=C["fail"], ha="left",
                arrowprops=dict(arrowstyle="-", color=C["fail"], linewidth=0.5))
    ax.text(150, 2.2e-8, "0.0（位级一致）", fontsize=6.2, color=C["pass"])
    ax.set_ylim(5e-9, 6e-1)
    ax.set_xlim(110, 300)
    ax.set_xticks([128, 192, 256])
    ax.set_xticklabels(["128", "192", "256"])
    ax.set_xlabel("decode 批大小")
    ax.set_ylabel("最大 |Δlogprob|")
    ax.set_title("批大小扫描（eager，target-only）", loc="left", fontsize=7.6, pad=6)
    tidy(ax)

    # b: in-batch prefill shapes
    ax = axes[1]
    for mode, color, label, marker in (("eager", C["fail"], "eager", "o"),
                                       ("compiled", C["pass"], "compiled", "s")):
        pts = len_points[mode]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=38 if mode == "compiled" else 15, marker=marker,
                   facecolor="white", edgecolor=color, linewidth=0.9,
                   label=label, zorder=3 if mode == "compiled" else 4)
    offsets = {(160, 1.0): (-16, 0.10, "left"), (253, 1.0): (0, -0.17, "center"),
               (284, 0.03): (10, -0.14, "left"), (455, 0.125): (-8, 0.14, "right"),
               (285, 1.0): (18, 0.16, "left")}
    for x, y, n, L in len_points["eager"]:
        dx, dy, ha = offsets.get((x, y), (6, 0.08, "left"))
        ax.annotate(f"N={n}, L={L}", (x, y), xytext=(x + dx, y + dy), fontsize=5.8,
                    ha=ha,
                    color=C["ink"],
                    arrowprops=dict(arrowstyle="-", color=C["muted"], linewidth=0.4,
                                    shrinkA=0, shrinkB=3) if abs(dy) > 0.11 else None)
    ax.axhline(1.0, color=C["grid"], linewidth=0.6)
    ax.set_xlim(120, 510)
    ax.set_ylim(-0.14, 1.28)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([160, 300, 450])
    ax.set_xticklabels(["160", "300", "450"])
    ax.set_xlabel("批内 prefill 总 token")
    ax.set_ylabel("通过比例")
    ax.set_title("批内形状（32/16/8 条）", loc="left", fontsize=7.6, pad=6)
    ax.legend(loc="center right", bbox_to_anchor=(1.02, 0.56), fontsize=6.3, handlelength=1.2, handletextpad=0.6,
              markerscale=0.55, borderpad=0.3)
    tidy(ax)

    # c: re-feed length
    ax = axes[2]
    for mode, color, label in (("eager", C["fail"], "eager"), ("compiled", C["pass"], "compiled")):
        pts = refeed[mode]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=0.9,
                marker="s", markersize=3.6, markerfacecolor="white",
                markeredgecolor=color, label=label, zorder=3)
    ax.annotate("48/256 通过", xy=(256, 0.8125), xytext=(170, 0.62), fontsize=6.2,
                color=C["fail"], arrowprops=dict(arrowstyle="-", color=C["fail"], linewidth=0.5))
    ax.set_ylim(-0.06, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([64, 128, 192, 256])
    ax.set_xlabel("生成后被拼回 prompt 的 token 数")
    ax.set_ylabel("不一致比例")
    ax.set_title("decode vs re-feed", loc="left", fontsize=7.6, pad=6)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.4)
    tidy(ax)

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter, dx=-0.26, dy=1.05)
    export(fig, "fig_boundary", OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
