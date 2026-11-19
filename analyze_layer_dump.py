"""Read two hidden-state dumps and say where they first part ways.

bi_layer_probe.py writes one .pt per captured module call under
<dump>/<tag>/rank0/. This script walks both tags in call order, decodes each
index into (step, layer, module), and prints the first few places where the
bits differ.

Usage:
    python analyze_layer_dump.py --dump ~/bi_layer_dump --a a256 --b a255
"""
import argparse
import os

import torch


def load(tag, dump):
    folder = os.path.join(dump, tag, "rank0")
    out = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".pt"):
            out.append(torch.load(os.path.join(folder, name), weights_only=False))
    return out


def decode(rows):
    """Attach (step, layer) to every capture."""
    step = -1
    layer = -1
    decoded = []
    for row in rows:
        name = row["name"]
        if name == "QKVParallelLinear":
            layer += 1
        if name == "Qwen3MoeForCausalLM":
            step += 1
            layer = -1
        decoded.append((step, layer, name, row))
    return decoded


def bit_diff(a, b):
    if a.numel() != b.numel():
        return None, None
    if a.dtype == torch.bfloat16:
        n = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    else:
        n = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
    return n, (a.float() - b.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.expanduser("~/bi_layer_dump"))
    ap.add_argument("--a", default="a256")
    ap.add_argument("--b", default="a255")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    rows_a = load(args.a, args.dump)
    rows_b = load(args.b, args.dump)
    print(f"{args.a}: {len(rows_a)} captures, {args.b}: {len(rows_b)} captures")
    n = min(len(rows_a), len(rows_b))
    dec_a = decode(rows_a[:n])
    dec_b = decode(rows_b[:n])

    bad = []
    for i in range(n):
        sa, la, na, ra = dec_a[i]
        sb, lb, nb, rb = dec_b[i]
        if na != nb:
            print(f"call {i}: name mismatch {na} vs {nb}")
            break
        fields = [f for f in ("in0", "row0", "row1") if f in ra and f in rb]
        d = None
        m = None
        tag = na
        for f in fields:
            dn, mn = bit_diff(ra[f], rb[f])
            if dn and (d is None or dn > d):
                d, m, tag = dn, mn, f"{na}[{f}]"
        if d:
            bad.append((i, sa, la, tag, d, ra["row0"].numel(), m))

    print(f"changed captures: {len(bad)} / {n}")
    for i, s, l, name, d, tot, m in bad[: args.limit]:
        print(f"  call {i:5d}  step {s:2d}  layer {l:2d}  {name:22s}"
              f"  {d}/{tot}  max {m:.3e}")


if __name__ == "__main__":
    main()
