"""Test each model-side op on its own: does row 0 survive a change of M?

The layer probe shows the first bit flip appearing inside a decoder layer.
This script walks the ops that layer uses, outside the engine, and asks the
same question for each: with rows [0, M) built from one generator, does row 0
come out with the same bits at M=255 and at other batch sizes?

Usage:
    python bi_op_shape.py --out results/tier2/bi_op_shape.json
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402

HIDDEN, MOE_N, EXPERTS, TOPK = 2048, 768, 128, 8
REFS = 255
SIZES = [1, 64, 128, 192, 254, 256, 257, 300]


def bit_diff(a, b):
    if a.dtype == torch.bfloat16:
        n = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    else:
        n = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
    return n, (a.float() - b.float()).abs().max().item()


def make(rows, cols):
    idx = torch.arange(rows * cols, device="cuda", dtype=torch.float32)
    return torch.sin(idx * 1e-3 + 0.37).view(rows, cols).to(torch.bfloat16)


def sweep(rows_out, name, fn, elems):
    ref = fn(REFS)[0].clone()
    torch.cuda.synchronize()
    for m in SIZES:
        out = fn(m)
        torch.cuda.synchronize()
        n, mx = bit_diff(out[0], ref)
        rows_out.append({"op": name, "M": m, "row0_diff": n,
                         "elems": elems, "max": mx})
        print(f"{name:20s} M={m:4d} row0 {n:5d}/{elems} max {mx:.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_op_shape.json")
    args = ap.parse_args()

    from vllm.model_executor.layers.batch_invariant import (
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    torch.cuda.init()
    torch.manual_seed(0)
    rows = []

    x = make(320, HIDDEN)
    w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)

    def rms(m):
        out = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.rms_norm(out, x[:m], w, 1e-6)
        return out

    def add_rms(m):
        xx = x[:m].clone()
        rr = x[:m].clone()
        torch.ops._C.fused_add_rms_norm(xx, rr, w, 1e-6)
        return xx

    sweep(rows, "rms_norm", rms, HIDDEN)
    sweep(rows, "fused_add_rms_norm", add_rms, HIDDEN)
    del x
    torch.cuda.empty_cache()

    xa = make(320, 8192)

    def silu(m):
        out = torch.empty(m, 4096, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(out, xa[:m])
        return out

    sweep(rows, "silu_and_mul", silu, 4096)
    del xa
    torch.cuda.empty_cache()

    q = make(320, 32 * 128)
    k = make(320, 4 * 128)
    ang = torch.arange(320 * 64, device="cuda", dtype=torch.float32).view(320, 64)
    cos_sin = torch.stack((torch.cos(ang * 1e-3), torch.sin(ang * 1e-3)),
                          dim=-1).to(torch.bfloat16)
    pos = torch.arange(320, device="cuda", dtype=torch.int64)

    def rope(m):
        qq = q[:m].clone()
        kk = k[:m].clone()
        torch.ops._C.rotary_embedding(pos[:m], qq, kk, 128, cos_sin[:m], True)
        return qq

    sweep(rows, "rotary_embedding", rope, 32 * 128)
    del q, k
    torch.cuda.empty_cache()

    from vllm.model_executor.layers.fused_moe import fused_experts

    hidden = make(320, HIDDEN)
    w1 = torch.randn(EXPERTS, 2 * MOE_N, HIDDEN, device="cuda",
                     dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, MOE_N, device="cuda",
                     dtype=torch.bfloat16) * 0.02
    gate = torch.randn(HIDDEN, EXPERTS, device="cuda", dtype=torch.bfloat16) * 0.02
    scores = torch.softmax((hidden.float() @ gate.float()), dim=-1)
    top_w, top_i = torch.topk(scores, TOPK, dim=-1)
    top_w = (top_w / top_w.sum(-1, keepdim=True)).to(torch.bfloat16)

    def moe(m):
        return fused_experts(hidden[:m], w1, w2, top_w[:m], top_i[:m])

    try:
        sweep(rows, "fused_experts", moe, HIDDEN)
    except Exception as exc:  # signature drift between versions
        print("fused_experts skipped:", repr(exc)[:200])
        rows.append({"op": "fused_experts", "error": repr(exc)[:200]})


    # Same data, different storage offset: does the fused norm pick a
    # different vectorised path and change the last bits?
    base = make(256, HIDDEN)
    for off in (0, 1, 2, 4, 8):
        buf_x = torch.empty(256 * HIDDEN + 16, device="cuda", dtype=torch.bfloat16)
        buf_r = torch.empty(256 * HIDDEN + 16, device="cuda", dtype=torch.bfloat16)
        xx = buf_x[off:off + 256 * HIDDEN].view(256, HIDDEN)
        rr = buf_r[off:off + 256 * HIDDEN].view(256, HIDDEN)
        xx.copy_(base)
        rr.copy_(base)
        torch.ops._C.fused_add_rms_norm(xx, rr, w, 1e-6)
        torch.cuda.synchronize()
        if off == 0:
            ref_off = xx[0].clone()
        n = (xx[0].view(torch.int16) != ref_off.view(torch.int16)).sum().item()
        rows.append({"op": "fused_add_rms_norm", "offset_elems": off,
                     "row0_diff": n, "elems": HIDDEN})
        print(f"norm offset={off:2d} elems  row0 {n:5d}/{HIDDEN}", flush=True)
        del buf_x, buf_r, xx, rr
        torch.cuda.empty_cache()

    with open(args.out, "w") as handle:
        json.dump({"rows": rows}, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
