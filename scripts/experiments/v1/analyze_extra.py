#!/usr/bin/env python3
"""Build Markdown tables for the boundary experiments (results/extra)."""
import json, os, struct, glob

def _repo_root():
    """Repository root, or the deployed layout root when there is no .git."""
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        if (os.path.isdir(os.path.join(path, "scripts"))
                and os.path.isdir(os.path.join(path, "results"))):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.dirname(os.path.abspath(__file__))
        path = parent


EXTRA = os.path.join(_repo_root(), "results", "extra")


def bits(x):
    return struct.pack(">d", float(x)).hex()


def cmp_cases(a, b, allow_none=False):
    if a is None or b is None:
        return {"ids": None, "bitwise": None, "first": None, "max": None}
    la, lb = a["logprobs"], b["logprobs"]
    n = min(len(la), len(lb))
    first = next((i for i in range(n)
                  if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i])), None)
    if first is None and len(la) != len(lb):
        first = n
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {"ids": a["token_ids"] == b["token_ids"],
            "bitwise": first is None,
            "first": first,
            "max": max(diffs) if diffs else None}


def load(name):
    p = os.path.join(EXTRA, name)
    with open(p) as f:
        return json.load(f)


def from_raw(v):
    return {"ids": v.get("token_ids_equal"),
            "bitwise": v.get("logprobs_bitwise_equal"),
            "first": v.get("first_logprob_mismatch"),
            "max": v.get("max_abs_logprob_diff")}


def cell(c):
    if c.get("ids") is None:
        return "—"
    if c["ids"] and c["bitwise"]:
        return "0.0 ✅"
    s = "{:.3e} ❌".format(c["max"])
    if not c["ids"]:
        s += " (ids ✗)"
    return s


def main():
    out = ["# 边界实验汇总表（results/extra）", ""]

    # Table I: batch-size sweep, eager PC0, greedy 32
    sweep = [("bs128", "bi_diag_bs128_pc0.json"),
             ("bs192", "bi_diag_bs192_pc0.json"),
             ("bs224", "bi_diag_bs224_pc0.json"),
             ("bs240", "bi_diag_bs240_pc0.json"),
             ("bs255", "bi_diag_bs255_pc0.json"),
             ("bs256", "bi_diag_bs256_pc0.json"),
             ("bs257", "bi_diag_bs257_mns512_pc0.json")]
    out += ["## 表 I：eager / FLASH_ATTN / PC=0 的批大小扫描（greedy，32 token）", "",
            "| 批大小 | bs1 vs batch | 逆序 | 首次不一致 | 最大 \\|Δlogprob\\| | token id | decode vs re-feed |",
            "|---|---|---|---|---|---|---|"]
    for label, fn in sweep:
        d = load(fn)
        c = from_raw(d["comparisons"]["single_vs_batch"])
        r = from_raw(d["comparisons"]["single_vs_reverse"])
        rf = d["comparisons"]["decode_vs_refeed"]
        rf_txt = "—" if not rf.get("available") else ("0.0 ✅" if rf.get("all_bitwise_equal")
                                                       else "{:.3e} ❌".format(rf.get("max_abs_diff")))
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label, cell(c), cell(r),
            "—" if c["first"] is None else c["first"],
            "—" if c["max"] is None else "{:.3e}".format(c["max"]),
            "✅" if c["ids"] else "❌", rf_txt))

    # Table J: bs=256 boundary conditions
    boundary = [
        ("eager PC1 bs256", "bi_diag_bs256_pc1.json"),
        ("eager PC0 bs256", "bi_diag_bs256_pc0.json"),
        ("eager PC0 bs256 + ignore_eos", "bi_diag_bs256_pc0_ie.json"),
        ("eager PC0 bs256 全 target", "bi_diag_bs256_pc0_allt.json"),
        ("eager PC0 bs256 max_num_seqs=512", "bi_diag_bs256_mns512_pc0.json"),
        ("eager PC0 bs256 max_num_seqs=128", "bi_diag_bs256_mns128_pc0.json"),
        ("compiled PC0 bs256", "bi_diag_bs256_comp_pc0.json"),
        ("eager PC0 bs256 temp=0.7", "bi_diag_bs256_temp07_pc0.json"),
        ("eager PC0 bs256 temp=0.7 max_num_seqs=128", "bi_diag_bs256_temp07_mns128_pc0.json"),
        ("eager PC0 bs256 生成 256 token", "bi_diag_bs256_gen256_pc0.json"),
        ("eager PC0 bs256 生成 256 token max_num_seqs=128", "bi_diag_bs256_gen256_mns128_pc0.json"),
        ("compiled PC0 bs256 生成 256 token", "bi_diag_bs256_comp_gen256_pc0.json"),
    ]
    out += ["", "## 表 J：bs=256 的边界条件", "",
            "| 配置 | bs1 vs batch | 逆序 | sampled single vs batch | 首次不一致 | 最大 \\|Δlogprob\\| | decode vs re-feed |",
            "|---|---|---|---|---|---|---|"]
    for label, fn in boundary:
        d = load(fn)
        c = from_raw(d["comparisons"]["single_vs_batch"])
        r = from_raw(d["comparisons"]["single_vs_reverse"])
        s = d["comparisons"].get("sampled_single_vs_batch")
        s = from_raw(s) if s else None
        rf = d["comparisons"]["decode_vs_refeed"]
        rf_txt = "—" if not rf.get("available") else ("0.0 ✅" if rf.get("all_bitwise_equal")
                                                       else "{:.3e} ❌".format(rf.get("max_abs_diff")))
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label, cell(c), cell(r), cell(s) if s else "—",
            "—" if c["first"] is None else c["first"],
            "—" if c["max"] is None else "{:.3e}".format(c["max"]), rf_txt))

    # Table K: orthogonal configurations
    ortho = [
        ("TP=4 基线 bs128", "bi_extra_tp4_eager_bs128.json"),
        ("TP=2 bs32", "bi_extra_tp2_eager_bs32.json"),
        ("TP=2 重复进程", "bi_extra_tp2_eager_bs32_repeat.json"),
        ("TRITON_ATTN bs32", "bi_extra_tp4_tritonattn.json"),
        ("chunk=2048 bs128", "bi_extra_tp4_eager_chunk2048_bs128.json"),
        ("temperature=0.7 bs32", "bi_extra_tp4_eager_temp07.json"),
        ("生成 256 token bs32", "bi_extra_tp4_eager_gen256.json"),
        ("长 prompt 8180 PC0", "bi_extra_tp4_pc0_long8180.json"),
        ("长 prompt 8180 PC1", "bi_extra_tp4_pc1_long8180.json"),
    ]
    out += ["", "## 表 K：正交配置（进程内比较）", "",
            "| 配置 | bs1 vs batch | 逆序 | 长 prompt | decode vs re-feed |",
            "|---|---|---|---|---|"]
    for label, fn in ortho:
        d = load(fn)
        c = from_raw(d["comparisons"]["single_vs_batch"])
        r = from_raw(d["comparisons"]["single_vs_reverse"])
        lg = from_raw(d["comparisons"]["long_batch_vs_single"])
        rf = d["comparisons"]["decode_vs_refeed"]
        rf_txt = "—" if not rf.get("available") else ("0.0 ✅" if rf.get("all_bitwise_equal")
                                                       else "{:.3e} ❌".format(rf.get("max_abs_diff")))
        out.append("| {} | {} | {} | {} | {} |".format(label, cell(c), cell(r), cell(lg), rf_txt))

    # Table L: cross-config comparisons of the same target request
    base = load("bi_extra_tp4_eager_bs128.json")["cases"]["single"]
    tp2a = load("bi_extra_tp2_eager_bs32.json")["cases"]["single"]
    tp2b = load("bi_extra_tp2_eager_bs32_repeat.json")["cases"]["single"]
    comp = load("bi_diag_bs256_comp_pc0.json")["cases"]["single"]
    eager128 = load("bi_diag_bs128_pc0.json")["cases"]["single"]
    cross = [
        ("TP=2 vs TP=4（短）", base, tp2a),
        ("TP=2 进程 1 vs 进程 2", tp2a, tp2b),
        ("TRITON_ATTN vs FLASH_ATTN", base, load("bi_extra_tp4_tritonattn.json")["cases"]["single"]),
        ("compiled vs eager（均 PC=0）", eager128, comp),
    ]
    out += ["", "## 表 L：跨配置比较（同一个短 prompt 的 `single` 输出）", "",
            "| 比较 | token id | logprob | 首次不一致 | 最大 \\|Δlogprob\\| |",
            "|---|---|---|---|---|"]
    for label, a, b in cross:
        c = cmp_cases(a, b)
        out.append("| {} | {} | {} | {} | {} |".format(
            label, "✅" if c["ids"] else "❌", "✅" if c["bitwise"] else "❌",
            "—" if c["first"] is None else c["first"],
            "—" if c["max"] is None else "{:.3e}".format(c["max"])))

    # Table Q: offline 32-prompt confirmation
    off = load("bi_batchall_offline_eager.json")
    bad = [k for k, v in off["cmp"].items() if not (v["ids"] and v["bitwise"])]
    rep = [k for k, v in off["repeat"].items() if not (v["ids"] and v["bitwise"])]
    out += ["", "## 表 Q：离线 32-prompt 复核（eager / PC=0 / greedy）", "",
            "| 运行 | 通过/总数 | 不一致的 prompt | 重复批不一致 |",
            "|---|---|---|---|"]
    out.append("| LLM.generate：32 条单独 vs 批内 | {}/{} | {} | {} |".format(
        len(off["cmp"]) - len(bad), len(off["cmp"]),
        ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""),
        len(rep)))

    text = "\n".join(out) + "\n"
    p = os.path.join(EXTRA, "analysis_tables.md")
    with open(p, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
