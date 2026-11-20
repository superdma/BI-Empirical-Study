"""Turn two hidden-state dumps into a step x layer difference matrix.

bi_layer_probe.py writes one .pt per captured module call; this reduces the
pair of dumps to the numbers fig_layer_probe.py draws.

Usage:
    python extract_layer_map.py --dump ~/bi_project/data/bi_layer_dump --a a256 --b a255
"""
import argparse
import json
import os

import torch


def load(tag, dump):
    folder = os.path.join(dump, tag, "rank0")
    return [torch.load(os.path.join(folder, f), weights_only=False)
            for f in sorted(os.listdir(folder)) if f.endswith(".pt")]


def bit_diff(a, b):
    if a.dtype == torch.bfloat16:
        return int((a.view(torch.int16) != b.view(torch.int16)).sum())
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.expanduser("~/bi_project/data/bi_layer_dump"))
    ap.add_argument("--a", default="a256")
    ap.add_argument("--b", default="a255")
    ap.add_argument("--out", default="bi_layer_map.json")
    args = ap.parse_args()

    rows_a = load(args.a, args.dump)
    rows_b = load(args.b, args.dump)
    n = min(len(rows_a), len(rows_b))
    grid = []
    step, layer = -1, -1
    for i in range(n):
        a, b = rows_a[i], rows_b[i]
        name = a["name"]
        if name == "QKVParallelLinear":
            layer += 1
        if name == "Qwen3MoeForCausalLM":
            step += 1
            layer = -1
        if name != "Qwen3MoeDecoderLayer" or step < 0:
            continue
        d = bit_diff(a["row0"], b["row0"])
        if "row1" in a and "row1" in b:
            d = max(d, bit_diff(a["row1"], b["row1"]))
        while len(grid) <= step:
            grid.append([0] * 48)
        grid[step][layer] = d

    with open(args.out, "w") as handle:
        json.dump({"a": args.a, "b": args.b, "grid": grid}, handle)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
