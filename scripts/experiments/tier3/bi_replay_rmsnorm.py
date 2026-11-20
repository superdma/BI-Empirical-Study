"""Replay the first divergent norm call with the tensors the engine used.

bi_composition.py dumps every RMSNorm call of the 256- and 255-request runs,
including the full hidden/residual matrices for a window around the first
divergence. This script takes that call and runs the same kernel outside the
engine: same tensors, same shapes, same weight, same eps.

Four arrangements are compared:

  engine256_full       the 256-row input as captured
  engine256_first255   the same input cut to 255 rows
  engine255_full       the 255-row input as captured
  engine255_pad256     the 255-row input padded back to 256 rows

If cutting or padding rows reproduces (or removes) the divergence, the kernel
depends on the batch size itself. If all four agree with each other, the
difference lives in the launch context and the profiler run has to name it.

Usage:
    python bi_replay_rmsnorm.py --out results/tier3/bi_replay_rmsnorm.json
"""
import argparse
import glob
import json
import os

import torch

DUMP = os.path.expanduser(os.environ.get("BI_DUMP", "~/bi_project/data/bi_norm_meta"))


def load_meta(tag):
    rows = []
    for path in glob.glob(os.path.join(DUMP, "meta", f"{tag}_*.jsonl")):
        with open(path) as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("rank") == 0:
                    rows.append(rec)
    return sorted(rows, key=lambda r: r["seq"])


def load_payload(tag, seq):
    path = os.path.join(DUMP, tag, "rank0", f"{seq:05d}.pt")
    return torch.load(path, weights_only=False) if os.path.exists(path) else None


def bit_diff(a, b):
    if a is None or b is None or a.numel() != b.numel():
        return None
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def find_first_divergence(meta_a, meta_b):
    by_seq = {r["seq"]: r for r in meta_b}
    for rec in meta_a:
        other = by_seq.get(rec["seq"])
        if other is None:
            continue
        pa = load_payload("c256", rec["seq"])
        pb = load_payload("c255", rec["seq"])
        if not pa or not pb or "in0" not in pa or "in0" not in pb:
            continue
        if pa["name"] != "RMSNorm" or pb["name"] != "RMSNorm":
            continue
        if bit_diff(pa["in0"], pb["in0"]) == 0 and bit_diff(pa["row0"], pb["row0"]):
            return rec, other, pa, pb
    return None


def fused(x, residual, weight, eps):
    from vllm import _custom_ops as ops
    xx, rr = x.clone(), residual.clone()
    ops.fused_add_rms_norm(xx, rr, weight, eps)
    return xx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_replay_rmsnorm.json")
    args = ap.parse_args()

    found = find_first_divergence(load_meta("c256"), load_meta("c255"))
    if found is None:
        raise SystemExit("no divergent RMSNorm call found")
    rec, other, pa, pb = found

    weight = torch.load(os.path.join(DUMP, "weights", f"{rec['weight_id']}.pt"),
                        weights_only=False).to("cuda")
    eps = rec["eps"]

    x256 = pa["full_in"].to("cuda")
    r256 = pa["full_in1"].to("cuda")
    x255 = pb["full_in"].to("cuda")
    r255 = pb["full_in1"].to("cuda")
    eng256 = pa["row0"]
    eng255 = pb["row0"]

    pad_x = torch.cat([x255, x255[-1:]], dim=0)
    pad_r = torch.cat([r255, r255[-1:]], dim=0)

    cases = [
        ("engine256_full", x256, r256),
        ("engine256_first255", x256[:255], r256[:255]),
        ("engine255_full", x255, r255),
        ("engine255_pad256", pad_x, pad_r),
    ]
    rows = []
    with torch.no_grad():
        for label, x, r in cases:
            y = fused(x, r, weight, eps)[0].to("cpu")
            rows.append({
                "label": label, "rows": int(x.shape[0]),
                "vs_engine256": bit_diff(y, eng256),
                "vs_engine255": bit_diff(y, eng255),
            })
            print(label, rows[-1], flush=True)

    result = {
        "seq": rec["seq"],
        "engine_row_diff": bit_diff(eng256, eng255),
        "meta": {k: rec[k] for k in ("x_shape", "x_stride", "x_offset",
                                     "x_ptr_mod", "eps")},
        "cases": rows,
        "case_row0_pairs": {
            "full256_vs_full255": bit_diff(
                fused(x256, r256, weight, eps)[0].to("cpu"),
                fused(x255, r255, weight, eps)[0].to("cpu")),
            "cut255_vs_full255": bit_diff(
                fused(x256[:255], r256[:255], weight, eps)[0].to("cpu"),
                fused(x255, r255, weight, eps)[0].to("cpu")),
        },
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
