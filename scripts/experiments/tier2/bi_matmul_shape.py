"""Is the BI linear layer itself sensitive to M?

The tile config for the persistent matmul is fixed (128/128/64 for bf16), so
the published hypothesis was that tile choice explains the 256 boundary.
This script checks the other half: even with a fixed tile config, does the
same row produce the same bits at different M?

Rows [0, M) come from one generator, so row 0 is the same tensor in every
call. Only M changes.

Usage:
    python bi_matmul_shape.py --out results/tier2/bi_matmul_shape.json
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SIZES = [1, 63, 64, 96, 127, 128, 129, 192, 255, 256, 257, 300]
SHAPES = [(2048, 1280, "qkv（每卡）"), (2048, 2048, "o_proj"),
          (2048, 151936, "lm_head")]


def bits(x):
    return x.view(torch.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_matmul_shape.json")
    args = ap.parse_args()

    from vllm.model_executor.layers.batch_invariant import (
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    torch.cuda.init()
    torch.manual_seed(0)

    result = {"device": torch.cuda.get_device_name(0), "rows": []}
    for k, n, label in SHAPES:
        a = torch.randn(320, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = None
        for m in SIZES:
            y = F.linear(a[:m], w)
            torch.cuda.synchronize()
            if ref is None:
                ref = y[0].clone()
            d = (bits(y[0]) != bits(ref)).sum().item()
            mx = (y[0].float() - ref.float()).abs().max().item()
            result["rows"].append({"shape": label, "K": k, "N": n, "M": m,
                                   "row0_diff": d, "row0_max": mx})
            print(f"{label:12s} M={m:4d} row0 diff {d:5d}/{n} max {mx:.3e}",
                  flush=True)
            del y
        del a, w
        torch.cuda.empty_cache()

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
