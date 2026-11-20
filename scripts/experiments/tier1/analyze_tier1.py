"""Render the four tier-1 mechanism experiments into Markdown tables.

    python3 analyze_tier1.py [results/tier1]

Reads bi_allreduce_*.json, bi_ar_thresh_*.json, bi_nccl_*.json,
bi_tail_scan.json and bi_bucket_scan*.json and writes analysis_tables.md
next to them.
"""
import glob
import json
import os
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def stem(path):
    return os.path.basename(path).replace(".json", "")


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return "0.0" if v == 0 else f"{v:.3e}"
    return str(v)


def label(path):
    d = load(path)
    return f"world={d['world']}，NCCL_ALGO={d['tag']}，H={d.get('hidden', '—')}"


def allreduce_table(paths):
    lines = [r"### 表 T1：纯 NCCL all-reduce 的形状敏感性（不加载模型）", "",
             r"| 运行 | 测试 | 数据类型 | 比较对象 | 位不同的元素 | 最大差 | 位级一致 |",
             "|---|---|---|---|---|---|---|"]
    for path in paths:
        d = load(path)
        tag = label(path)
        for row in d["rows"]:
            test = row.get("test", "reshape")
            key = row.get("pair") or row.get("shape") or row.get("against")
            if isinstance(key, list):
                key = " → ".join(str(v) for v in key)
            lines.append(f"| {tag} | {test} | {row['dtype']} | {key} | "
                         f"{row['n_diff']} / {row['total']} | {fmt(row['max_abs'])} | "
                         f"{fmt(row['bitwise'])} |")
    return lines


def threshold_table(paths):
    lines = [r"### 表 T1b：批大小每加 1，未变的那些行是否还是原来的位", "",
             "比较的是 `[b,H]` 与 `[b+1,H]` 的前 b 行（H=2048，与 30B-A3B 的 hidden 一致），",
             "b 从 1 走到 299。下表统计有多少个 b→b+1 出现了位差，并列出最早几个。", "",
             "| 运行 | 数据类型 | 出现位差的步数 | 最早的几个 b→b+1 |",
             "|---|---|---|---|"]
    for path in paths:
        d = load(path)
        tag = label(path)
        for dtype in ("bfloat16", "float32"):
            rows = [r for r in d["rows"] if r["dtype"] == dtype and r["n_diff"]]
            firsts = ", ".join(f"{r['pair'][0]}→{r['pair'][1]}" for r in rows[:8])
            lines.append(f"| {tag} | {dtype} | {len(rows)} / {len([r for r in d['rows'] if r['dtype']==dtype])} | "
                         f"{firsts or '无'} |")
    return lines


def nccl_table(paths):
    lines = [r"### 表 T2：vLLM eager TP=4 固定 NCCL 算法后的 32 条 prompt", "",
             r"| NCCL_ALGO | 位级一致 | token id 变化 | 最大 \|Δlogprob\| | 重复批不一致 | 批量耗时 |",
             "|---|---|---|---|---|---|"]
    for path in paths:
        d = load(path)
        cmp = d["cmp"]
        good = sum(1 for v in cmp.values() if v["ids"] and v["bitwise"])
        ids_bad = sum(1 for v in cmp.values() if not v["ids"])
        mx = max((v["max"] or 0.0) for v in cmp.values())
        rep_bad = sum(1 for v in d["repeat"].values() if not (v["ids"] and v["bitwise"]))
        lines.append(f"| {d['env']['tag']} | {good}/{len(cmp)} | {ids_bad} | {fmt(mx)} | "
                     f"{rep_bad} | {d['env'].get('batch_sec', 0):.2f} s |")
    return lines


def tail_table(path):
    d = load(path)
    lines = ["### 表 T3：前缀缓存命中与 fresh prefill 随 tail 长度的变化", "",
             f"基座 prompt {d['env']['base_tokens']} token，比较前 "
             f"{d['env']['max_tokens']} 个生成 token。", "",
             r"| tail token | 位级一致 | 首个不一致 | 最大 \|Δlogprob\| | token id 相同 |",
             "|---|---|---|---|---|"]
    for row in d["rows"]:
        if "error" in row:
            lines.append(f"| {row['tail']} | 报错 | — | — | — |")
            continue
        lines.append(f"| {row['tail']} | {fmt(row['bitwise'])} | {fmt(row['first_mismatch'])} | "
                     f"{fmt(row['max_abs'])} | {fmt(row['token_ids_equal'])} |")
    return lines


def bucket_table(path):
    d = load(path)
    mode = "eager" if d["env"]["eager"] else "compiled"
    lines = [f"### 表 T4（{mode}）：CUDA graph bucket 边界附近的批大小扫描", "",
             r"| 批大小 | 位级一致 | 首个不一致 | 最大 \|Δlogprob\| | 该批耗时 |",
             "|---|---|---|---|---|"]
    for row in d["rows"]:
        lines.append(f"| {row['bs']} | {fmt(row['bitwise'])} | {fmt(row['first_mismatch'])} | "
                     f"{fmt(row['max_abs'])} | {row['sec']:.2f} s |")
    return lines


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/tier1"
    out = []
    ar = (sorted(glob.glob(os.path.join(root, "bi_allreduce_*.json")))
            + sorted(glob.glob(os.path.join(root, "bi_ar_sparse_*.json")))
            + sorted(glob.glob(os.path.join(root, "bi_ar_pos_*.json"))))
    if ar:
        out += allreduce_table(ar) + [""]
    th = sorted(glob.glob(os.path.join(root, "bi_ar_thresh_*.json")))
    if th:
        out += threshold_table(th) + [""]
    nc = sorted(glob.glob(os.path.join(root, "bi_nccl_*.json")))
    if nc:
        out += nccl_table(nc) + [""]
    tail = os.path.join(root, "bi_tail_scan.json")
    if os.path.exists(tail):
        out += tail_table(tail) + [""]
    for bucket in sorted(glob.glob(os.path.join(root, "bi_bucket_scan*.json"))):
        out += bucket_table(bucket) + [""]
    text = "\n".join(out)
    with open(os.path.join(root, "analysis_tables.md"), "w") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
