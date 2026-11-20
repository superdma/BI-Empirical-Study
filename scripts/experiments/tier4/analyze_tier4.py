"""Render the fourth-round results as the tables used by chapter 10.

Usage:
    python3 analyze_tier4.py results/tier4
"""
import json
import os
import sys


def load(root, name):
    with open(os.path.join(root, name)) as handle:
        return json.load(handle)


def short_kernel(name):
    if "<c10::BFloat16, 8>" in name:
        return "`<bf16, 8>`（向量化）"
    if "<c10::BFloat16, 0>" in name:
        return "`<bf16, 0>`（标量）"
    return "?"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/tier4"
    inproc = load(root, "bi_inproc_replay.json")
    probe = {b: load(root, f"probe_bi{b}.json") for b in ("0", "1")}
    sweep = load(root, "bi_boundary_sweep.json")
    shapes = {k: load(root, f"bi_shapes_{k}.json")
              for k in ("eager", "no_bcg", "compiled")}
    control = load(root, "bi_prompt_control.json")

    out = ["# 第四档：原始结果的汇总表", ""]

    out += ["## T17　同一份输入，BI 开与关各跑一遍", "",
            "| 运行 | 输入 | 行数 | 与 256 引擎输出差 | 与 255 引擎输出差 | 内核 | grid | block |",
            "|---|---|---|---|---|---|---|---|"]
    labels = {"x256": "256 的输入", "x255": "255 的输入",
              "xfill256": "第 0 行复制 256 行"}
    for b, title in (("0", "BI 关"), ("1", "BI 开")):
        for case in probe[b]["cases"]:
            launch = case["launches"][0]
            out.append(
                f"| {title} | {labels.get(case['label'], case['label'])} | {case['rows']} | "
                f"{case['vs_engine256']} | {case['vs_engine255']} | "
                f"{short_kernel(launch['name'])} | {launch['grid'][0]} | {launch['block'][0]} |")
    out += ["", "引擎进程内回放（`BI_INPROC_SEQ=2548`）：", "",
            "| 运行 | 行数 | 与引擎输出差（第 0 行） | 与引擎输出差（全量） | 与引擎 residual 差（第 0 行） |",
            "|---|---|---|---|---|"]
    for rec in inproc["records"]:
        out.append(f"| {rec['tag']} | {rec['rows']} | {rec['replay_vs_engine_row0']} | "
                   f"{rec['replay_vs_engine_all']} | {rec.get('replay_vs_engine_res_row0', '-')} |")
    out += ["", "## T18　decode 批大小从 255 扫到 512", "",
            "| 批大小 | 比较条数 | 位级一致 | 改变 | 首 token 就不同 | 最大 \\|Δlogprob\\| | 墙钟 |",
            "|---|---|---|---|---|---|---|"]
    for size, row in sweep["vs_255"].items():
        out.append(f"| {size} | {row['n']} | {row['bitwise_equal']} | {row['changed']} | "
                   f"{row['first_token_bad']} | {row['max_abs']:.4g} | {row['seconds']} s |")

    out += ["", "## T19　同形状矩阵：eager、关 breakable graph、compiled", "",
            "| 配置 | 32 条单独 vs 批内 | decode 256 vs 255 | decode 448 vs 255 | decode 512 vs 255 |",
            "|---|---|---|---|---|"]
    names = {"eager": "eager（默认）", "no_bcg": "eager + `VLLM_USE_BREAKABLE_CUDAGRAPH=0`",
             "compiled": "compiled"}
    for key in ("eager", "no_bcg", "compiled"):
        comp = shapes[key]["comp"]
        def cell(name):
            row = comp[name]
            return f"{row['bitwise_equal']}/{row['n']}"
        out.append(f"| {names[key]} | {cell('solo_vs_batch_32')} | {cell('decode_256_vs_255')} | "
                   f"{cell('decode_448_vs_255')} | {cell('decode_512_vs_255')} |")

    out += ["", "## T20　eager 的失败率随 prompt 集与生成长度变", "",
            "| 配置 | prompt 集 | max_tokens | ignore_eos | 位级一致 | 最大 \\|Δlogprob\\| |",
            "|---|---|---|---|---|---|"]
    label_text = {"old_prompts_16": ("target + 31 条 filler", "16", "否"),
                  "new_prompts_16": ("32 条问答 prompt", "16", "是"),
                  "old_prompts_8": ("target + 31 条 filler", "8", "是")}
    for key, row in control["runs"].items():
        pset, mt, ie = label_text.get(key, (key, str(row["max_tokens"]), "-"))
        out.append(f"| {key} | {pset} | {mt} | {ie} | {row['bitwise_equal']}/{row['n']} | "
                   f"{row['max_abs']:.4g} |")

    text = "\n".join(out) + "\n"
    with open(os.path.join(root, "analysis_tables.md"), "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
