"""Render the tier-2 results as Markdown tables.

Usage:
    python analyze_tier2.py results/tier2
"""
import json
import os
import sys


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def coverage(res):
    d = load(os.path.join(res, "bi_bi_coverage.json"))
    if not d:
        return []
    body = []
    for op, k in d["op_coverage"].items():
        text = ", ".join(k)
        if "elementwise_kernel" in text:
            text = "aten 逐元素内核（由几个 aten 算子拼成）"
        body.append([op, text[:96]])
    return ["### 表 T5：BI 在 SM86 上替换了哪些算子\n",
            table(["aten 算子", "实际执行的内核"], body), ""]


def matmul(res):
    d = load(os.path.join(res, "bi_bi_coverage.json"))
    s = load(os.path.join(res, "bi_matmul_shape.json"))
    if not d:
        return []
    cfg = d["matmul"]["rows"][0]["config"]
    out = ["### 表 T6：BI 线性层的 tile 配置\n"]
    out.append(
        f"K=2048，bf16。M 从 {d['matmul']['rows'][0]['M']} 走到 "
        f"{d['matmul']['rows'][9]['M']}，配置始终是 "
        f"BLOCK_M={cfg['BLOCK_SIZE_M']}、BLOCK_N={cfg['BLOCK_SIZE_N']}、"
        f"BLOCK_K={cfg['BLOCK_SIZE_K']}、GROUP_M={cfg['GROUP_SIZE_M']}、"
        f"warps={cfg['num_warps']}，不同配置数 "
        f"{d['matmul']['distinct_configs']}。\n")
    if s:
        body = []
        for shape in ("qkv（每卡）", "o_proj", "lm_head"):
            rows = [r for r in s["rows"] if r["shape"] == shape]
            bad = sum(1 for r in rows if r["row0_diff"])
            body.append([shape, rows[0]["K"], rows[0]["N"], len(rows), bad])
        out.append(table(["形状", "K", "N", "测过的 M 个数", "行 0 改位的 M 个数"], body))
        out.append("")
    return out


def moe(res):
    d = load(os.path.join(res, "bi_bi_coverage.json"))
    if not d:
        return []
    body = []
    for key, label in (("moe_bi_on", "开"), ("moe_bi_off", "关")):
        b = d[key]
        pts = ", ".join(str(c["at_M"]) for c in b["change_points"]) or "无"
        c = b["at_255"]
        body.append([label, len(b["change_points"]), pts,
                     f"{c.get('BLOCK_SIZE_M')}/{c.get('BLOCK_SIZE_N')}/"
                     f"{c.get('BLOCK_SIZE_K')}",
                     "是" if b["same_at_255_256"] else "否"])
    return ["### 表 T7：MoE 默认配置随 M 的切换点\n",
            table(["VLLM_BATCH_INVARIANT", "切换次数", "切换发生在 M",
                   "255 时的配置", "255 与 256 相同"], body), ""]


def nccl(res):
    d = load(os.path.join(res, "bi_bi_coverage.json"))
    if not d:
        return []
    body = [[f"`{k}`", v] for k, v in d["nccl_env"]["after"].items()]
    return ["### 表 T8：BI 启动时自己写入的 NCCL 环境\n",
            table(["变量", "值"], body),
            f"\n调用前用户设的是 `NCCL_ALGO={d['nccl_env']['before']['NCCL_ALGO']}`。\n"]


def pynccl(res):
    body = []
    for label, fname in (("BI 环境", "bi_pynccl_bi_env.json"),
                         ("tree，8 通道", "bi_pynccl_ch8.json"),
                         ("NCCL 自动选择", "bi_pynccl_auto.json")):
        r = load(os.path.join(res, fname))
        if not r:
            continue
        pair = next((p for p in r["pairs"] if p["a"] == 255 and p["b"] == 256), None)
        body.append([label, r["algo"] or "自动", r["min_channels"] or "默认",
                     f"{r['steps_changed']}/{r['steps_total']}",
                     f"{pair['n_diff']} / {pair['total']}" if pair else "—",
                     r["repeat_255"]["n_diff"]])
    return ["### 表 T9：引擎自带 pynccl 通信器的批大小敏感性\n",
            table(["运行", "NCCL_ALGO", "通道数", "299 步里改位的步数",
                   "255 与 256 改位元素", "同一批重复"], body), ""]


def ops(res):
    s = load(os.path.join(res, "bi_op_shape.json"))
    a = load(os.path.join(res, "bi_attn_shape.json"))
    if not s:
        return []
    body = []
    seen = []
    for row in s["rows"]:
        if "op" not in row or "M" not in row:
            continue
        if row["op"] not in seen:
            seen.append(row["op"])
    for op in seen:
        rows = [r for r in s["rows"] if r.get("op") == op and "M" in r]
        bad = sum(1 for r in rows if r["row0_diff"])
        body.append([op, len(rows), bad, rows[0]["elems"]])
    if a:
        rows = [r for r in a["rows"] if r["n_seq"] >= 255]
        bad = sum(1 for r in rows if r["row0_diff"])
        body.append(["paged decode attention", len(rows), bad, rows[0]["elems"]])
    return ["### 表 T10：每个算子单独跑，行 0 在 255 与 256 之间是否变\n",
            table(["算子", "测过的批大小个数", "行 0 变位的个数", "每行元素"], body),
            "\n批大小取 1、64、128、192、254、256、257、300，基准是 255。"
            "线性层另测了三组形状（表 T6）。\n"]


def layer(res):
    d = load(os.path.join(res, "bi_layer_probe.json"))
    if not d:
        return []
    body = []
    for name, block in d["pairs"].items():
        first = block.get("first")
        if not first:
            body.append([name.replace("_vs_", " vs "), block.get("calls"), 0,
                         "—", "—"])
        else:
            body.append([name.replace("_vs_", " vs "), block.get("calls"),
                         block.get("files_changed"),
                         first["name"],
                         f"{first['n_diff']}/{first['elems']}"
                         f"（{first['max_abs']:.2e}）"])
    return ["### 表 T11：逐层打点（行 0）\n",
            table(["比较", "捕获次数", "改位的模块数", "首个改位的模块", "幅度"], body),
            ""]


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "results/tier2"
    parts = []
    for fn in (coverage, matmul, moe, nccl, pynccl, ops, layer):
        parts.extend(fn(res))
    text = "\n".join(parts)
    path = os.path.join(res, "analysis_tables.md")
    with open(path, "w") as handle:
        handle.write(text + "\n")
    print(text)
    print("wrote", path)


if __name__ == "__main__":
    main()
