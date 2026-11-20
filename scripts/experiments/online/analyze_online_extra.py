#!/usr/bin/env python3
"""Build Markdown tables for the online concurrency, batch-all, length and refeed experiments."""
import json, os, glob

def _repo_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(path, ".git")):
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        path = parent
    return path


ROOT = _repo_root()
C = os.path.join(ROOT, "results", "http", "concurrency")
E = os.path.join(ROOT, "results", "extra")


def load(p):
    with open(p) as f:
        return json.load(f)


def find(*names):
    """First existing file in results/http/concurrency; None if none match."""
    for name in names:
        p = os.path.join(C, name)
        if os.path.exists(p):
            return p
    return None


def bad_count(cmp, pred=lambda v: v["ids"] and v["bitwise"]):
    return sum(1 for v in cmp.values() if not pred(v))


def main():
    out = ["# 补充实验汇总表（在线并发、批内长度、re-feed）", ""]

    out += ["## 表 M：HTTP 到达方式与并发", "",
            "| 场景 | 模式 | N | 比较对象 | 通过/总数 | 最大 \\|Δlogprob\\| |",
            "|---|---|---|---|---|---|"]
    rows = []
    for tag, label in [("eager_pc0", "eager PC0"), ("eager_pc1", "eager PC1"),
                       ("compiled_pc1", "compiled PC1")]:
        p = os.path.join(C, f"bi_http_conc_{tag}.json")
        if os.path.exists(p):
            d = load(p)
            prompt = {k: v for k, v in d["cmp"].items() if k.startswith("prompt")}
            rows.append((f"独立并发请求（短 prompt）", label, d["env"]["n"],
                         "逐 prompt：单独 vs 并发", len(prompt) - bad_count(prompt), len(prompt),
                         max((v["max"] for v in prompt.values()), default=0)))
    for tag, label in [("eager_pc0", "eager PC0"), ("compiled_pc1", "compiled PC1")]:
        p = find(f"bi_http_burst_{tag}.json", f"burst_{tag}.json")
        if p:
            d = load(p)
            for key, name in [("cmp_during", "突发 32 请求（另一请求正在解码）"),
                              ("cmp_alone", "突发 32 请求（无背景解码）")]:
                rows.append((name, label, 32, "逐 prompt：单独 vs 突发",
                             len(d[key]) - bad_count(d[key]), len(d[key]),
                             max((v["max"] for v in d[key].values()), default=0)))
    for tag, label in [("eager_pc0", "eager PC0"), ("compiled_pc1", "compiled PC1")]:
        p = find(f"batchall_{tag}.json", f"bi_http_batchall_{tag}.json")
        if p:
            d = load(p)
            rows.append(("单次批量请求（32 个 prompt 一起提交）", label, 32,
                         "逐 prompt：单独 vs 批内", len(d["cmp"]) - bad_count(d["cmp"]),
                         len(d["cmp"]),
                         max((v["max"] for v in d["cmp"].values()), default=0)))
    for tag, label in [("eager_chunked_pc0", "eager PC0"), ("eager_nochunk_pc0", "eager 无 chunked"),
                       ("compiled_chunked_pc0", "compiled PC0")]:
        p = find(f"mixed_{tag}.json", f"bi_http_mixed_{tag}.json")
        if p:
            d = load(p)
            ok = sum(1 for c in d["cases"] if c["cmp"]["ids"] and c["cmp"]["bitwise"])
            rows.append(("单个请求插入解码中的批次（4 个延迟）", label, 1,
                         "单独 vs 混合", ok, len(d["cases"]),
                         max((c["cmp"]["max"] for c in d["cases"]), default=0)))
    for name, label, n, what, ok, tot, mx in rows:
        out.append("| {} | {} | {} | {} | {}/{} | {:.3e} |".format(
            name, label, n, what, ok, tot, mx))

    out += ["", "## 表 N：批内 prompt 长度扫描（eager，FLASH_ATTN，PC=0）", "",
            "| N | 每个 filler 长度 | 总 prefill token | 通过/总数 | 最大 \\|Δlogprob\\| | 重复批一致 |",
            "|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(C, "bi_http_len_eager_*.json"))):
        d = load(p); e = d["env"]
        bad = bad_count(d["cmp"])
        rep = bad_count(d["repeat"])
        out.append("| {} | {} | {} | {}/{} | {:.3e} | {} |".format(
            e["n"], e["L"], e["total_prompt_tokens"], len(d["cmp"]) - bad, len(d["cmp"]),
            max((v["max"] for v in d["cmp"].values()), default=0),
            "是" if rep == 0 else f"否（{rep}）"))
    out += ["", "### 对照：compiled 模式", "",
            "| N | 每个 filler 长度 | 总 prefill token | 通过/总数 | 最大 \\|Δlogprob\\| |",
            "|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(C, "bi_http_len_compiled_*.json"))):
        d = load(p); e = d["env"]
        bad = bad_count(d["cmp"])
        out.append("| {} | {} | {} | {}/{} | {:.3e} |".format(
            e["n"], e["L"], e["total_prompt_tokens"], len(d["cmp"]) - bad, len(d["cmp"]),
            max((v["max"] for v in d["cmp"].values()), default=0)))

    out += ["", "## 表 O：decode vs re-feed 的生成长度扫描（eager，PC=0）", "",
            "| 生成 token 数 | 拼回后总长度 | 通过/比较数 | 首个不一致 | 最大 \\|Δlogprob\\| |",
            "|---|---|---|---|---|"]
    for mt in (64, 128, 192, 256):
        p = os.path.join(E, f"bi_refeed_eager_pc0_mt{mt}.json")
        if not os.path.exists(p):
            continue
        d = load(p)
        v = d["refeeds"]["1"]
        out.append("| {} | {} | {}/{} | {} | {} |".format(
            mt, len(d["env"]["prompt_ids"]) + mt,
            v["n_compared"] - v["n_mismatch"], v["n_compared"],
            "—" if v["first_mismatch"] is None else v["first_mismatch"],
            "—" if v["max_abs_diff"] is None else "{:.3e}".format(v["max_abs_diff"])))
    p = os.path.join(E, "bi_refeed_compiled_pc0_mt256.json")
    if os.path.exists(p):
        d = load(p); v = d["refeeds"]["1"]
        out.append("| 256（compiled） | {} | {}/{} | {} | {} |".format(
            len(d["env"]["prompt_ids"]) + 256,
            v["n_compared"] - v["n_mismatch"], v["n_compared"],
            "—" if v["first_mismatch"] is None else v["first_mismatch"],
            "—" if v["max_abs_diff"] is None else "{:.3e}".format(v["max_abs_diff"])))

    out += ["", "## 表 P：模型 / attention / MoE 后端对照（单次批量请求，32 prompt）", "",
            "| 配置 | 通过/总数 | 批重复一致 | 说明 |",
            "|---|---|---|---|"]
    controls = [
        ("Qwen3-30B-A3B, eager, FLASH, chunked", "bi_http_batchall_eager_pc0.json", "基线"),
        ("Qwen3-30B-A3B, eager, TRITON_ATTN", "bi_http_batchall_eager_triton_pc0.json", "换 attention 后端"),
        ("Qwen3-30B-A3B, eager, FLASH, 无 chunked", "bi_http_batchall_eager_flash_nochunk_pc0.json", "关 chunked prefill"),
        ("Qwen3-30B-A3B, eager, moe=triton（显式）", "bi_http_batchall_triton_explicit.json", "默认 MoE 后端"),
        ("Qwen3-0.6B 稠密模型, eager, FLASH", "bi_http_batchall_dense06_eager_pc0.json", "排除 MoE 因素"),
        ("Qwen3-30B-A3B, compiled, FLASH", "bi_http_batchall_compiled_pc1.json", "torch.compile + CUDA graphs"),
    ]
    for label, fn, note in controls:
        p = os.path.join(C, fn)
        if not os.path.exists(p):
            continue
        d = load(p)
        bad = bad_count(d["cmp"])
        rep = bad_count(d.get("repeat", {}))
        out.append("| {} | {}/{} | {} | {} |".format(
            label, len(d["cmp"]) - bad, len(d["cmp"]),
            "是" if rep == 0 else f"否（{rep}）", note))

    text = "\n".join(out) + "\n"
    p = os.path.join(C, "analysis_tables.md")
    with open(p, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
