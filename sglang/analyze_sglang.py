"""Render SGLang experiment JSON files as Markdown tables.

Usage:
  python3 sglang/analyze_sglang.py [results_dir]
"""
import glob
import json
import os
import sys


def cell(c):
    if not c:
        return "—"
    if c.get("ids") and c.get("bitwise"):
        return "0.0 ✅"
    mx = c.get("max")
    s = "n/a ❌" if mx is None else f"{mx:.3e} ❌"
    if c.get("ids") is False:
        s += " (ids ✗)"
    return s


def load(pattern):
    """Collect one table's runs; the tag is the file name without the test suffix.

    Tags themselves contain underscores (sgl_det_nograph), so splitting on the
    first underscore would collapse different configurations into one column.
    """
    suffix = os.path.basename(pattern).lstrip("*")
    out = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        tag = name[: -len(suffix)] if name.endswith(suffix) else name
        with open(path) as handle:
            out[tag] = json.load(handle)
    return out


def table_batchall(runs):
    print("| 配置 | solo vs batch | batch vs batch | n |")
    print("|---|---|---|---|")
    for tag, r in runs.items():
        s = r["summary"]["solo_vs_batch"]
        rep = r["summary"]["batch_repeat"]
        print(f"| {tag} | {s['bad']}/{s['n']} 不一致 | {rep['bad']}/{rep['n']} 不一致 | {s['n']} |")


def table_bssweep(runs):
    sizes = None
    for r in runs.values():
        if sizes is None:
            sizes = [row["bs"] for row in r["rows"]]
    print("| bs | " + " | ".join(runs) + " |")
    print("|---" * (len(runs) + 1) + "|")
    for i, bs in enumerate(sizes or []):
        cells = []
        for r in runs.values():
            rows = [x for x in r["rows"] if x["bs"] == bs]
            cells.append(cell(rows[0]) if rows else "—")
        print(f"| {bs} | " + " | ".join(cells) + " |")


def table_len(runs):
    print("| 配置 | solo vs batch | batch vs batch | 总 token |")
    print("|---|---|---|---|")
    for tag, r in runs.items():
        s = r["summary"]["solo_vs_batch"]
        rep = r["summary"]["batch_repeat"]
        print(f"| {tag} | {s['bad']}/{s['n']} 不一致 | {rep['bad']}/{rep['n']} 不一致 | "
              f"{r['env']['total_prompt_tokens']} |")


def table_refeed(runs):
    print("| 配置 | 比较数 | 不一致 | 首个不一致位置 | 最大 |Δlogprob| |")
    print("|---|---|---|---|---|")
    for tag, r in runs.items():
        mx = r["max_abs_diff"]
        print(f"| {tag} | {r['n_compared']} | {r['n_mismatch']} | "
              f"{r['first_mismatch']} | {'—' if mx is None else f'{mx:.3e}'} |")


def table_long(runs):
    print("| 配置 | fresh vs cache hit | hit vs hit2 |")
    print("|---|---|---|")
    for tag, r in runs.items():
        print(f"| {tag} | {cell(r['fresh_vs_hit'])} | {cell(r['hit_vs_hit2'])} |")


def table_http(runs):
    groups = {}
    for tag, r in runs.items():
        for k, v in r["cmp"].items():
            if "concurrent" in k:
                groups.setdefault("seq vs concurrent", []).append(v)
            elif "identical" in k:
                groups.setdefault("identical vs baseline", []).append(v)
            elif "burst" in k:
                groups.setdefault("seq vs burst", []).append(v)
    print("| 场景 | 比较数 | 不一致 |")
    print("|---|---|---|")
    for name, vals in groups.items():
        bad = sum(1 for v in vals if not (v["ids"] and v["bitwise"]))
        print(f"| {name} | {len(vals)} | {bad} |")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
    print("## batchall\n")
    table_batchall(load(os.path.join(root, "*_batchall.json")))
    print("\n## bssweep\n")
    table_bssweep(load(os.path.join(root, "*_bssweep.json")))
    print("\n## len\n")
    table_len(load(os.path.join(root, "*_len.json")))
    print("\n## refeed\n")
    table_refeed(load(os.path.join(root, "*_refeed.json")))
    print("\n## long\n")
    table_long(load(os.path.join(root, "*_long.json")))
    print("\n## http\n")
    table_http(load(os.path.join(root, "*_http.json")))


if __name__ == "__main__":
    main()
