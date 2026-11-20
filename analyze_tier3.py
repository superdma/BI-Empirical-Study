"""Render the tier-3 results as Markdown tables.

Usage:
    python3 analyze_tier3.py results/tier3
"""
import json
import os
import sys

T = []


def load(folder, name):
    with open(os.path.join(folder, name)) as handle:
        return json.load(handle)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    T.append("\n".join(out) + "\n")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/tier3"
    comp = load(folder, "bi_composition.json")
    replay = load(folder, "bi_replay_rmsnorm.json")
    blocks = load(folder, "bi_launch_blocks.json")
    probe = load(folder, "bi_block_probe.json")
    tp2 = load(folder, "bi_tp2_matrix.json")

    T.append("### 表 T12　请求组成：改什么才会改变结果\n")
    names = [("c255_vs_c256", "255 条 vs 256 条"),
             ("c255_vs_c255b", "255 条重复一次"),
             ("s255_vs_c255", "255 条 + 1 条只生成 1 个 token"),
             ("r256_vs_c256", "256 条逆序")]
    rows = []
    for key, label in names:
        v = comp["cmp"][key]
        rows.append([label, f'{v["n"]}', f'{v["bitwise_equal"]}/{v["n"]}',
                     v["first_token_bad"], f'{v["max_abs"]:.3e}'])
    table(["比较", "条数", "位级一致", "首 token 就不同", "最大 \\|Δlogprob\\|"], rows)

    T.append("### 表 T13　用引擎自己的张量回放第 42 层的 norm\n")
    rows = [[c["label"], c["rows"], c["vs_engine256"], c["vs_engine255"]]
            for c in replay["cases"]]
    table(["输入", "行数", "与 256 引擎输出的差异元素", "与 255 引擎输出的差异元素"], rows)

    T.append("### 表 T14　引擎里 fused add+norm 的启动参数\n")
    rows = []
    for tag, label in (("p255", "255 条"), ("p256", "256 条")):
        for pair in blocks[tag]["pairs"]:
            rows.append([label, pair["grid"], pair["block"], pair["count"]])
    table(["运行", "grid", "block", "启动次数"], rows)

    T.append("### 表 T15　独立调用时的 block 选择\n")
    rows = []
    for n, value in probe["rows_to_launch"].items():
        grid, block = value[0][0], value[1][0]
        rows.append([n, grid, block])
    table(["行数", "grid", "block"], sorted(rows, key=lambda r: int(r[0])))

    T.append("### 表 T16　TP=2 的批不变性\n")
    rows = []
    for key, label in (("solo_vs_batch_32", "32 条：单独 vs 批内"),
                       ("batch_repeat_32", "32 条：批请求重复"),
                       ("bs255_vs_bs256", "255 条 vs 256 条"),
                       ("bs256_repeat", "256 条重复")):
        v = tp2["cmp"][key]
        rows.append([label, f'{v["bitwise_equal"]}/{v["n"]}',
                     f'{v.get("max_abs", 0.0):.3e}'])
    table(["比较", "位级一致", "最大 \\|Δlogprob\\|"], rows)

    path = os.path.join(folder, "analysis_tables.md")
    with open(path, "w") as handle:
        handle.write("\n".join(T))
    print("wrote", path)


if __name__ == "__main__":
    main()
