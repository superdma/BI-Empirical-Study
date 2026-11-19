"""Run the all-reduce shape probe through vLLM's own pynccl communicator.

Tier 1 compared all-reduce results with torch.distributed's default NCCL
communicator. The engine does not use that one: it builds a pynccl
communicator, and BI pins NCCL_ALGO=allreduce:tree, NCCL_PROTO=Simple and a
single channel before that communicator is created. This script repeats the
same comparison through PyNcclCommunicator with the same environment.

Every rank holds different data, so the sum is not a power-of-two scaling of
one rank's values. Rows [0, b) are identical for every batch size b, so their
output bits must not depend on b.

Usage:
    torchrun --nproc_per_node=4 bi_pynccl_bench.py --out results/tier2/x.json
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

SAVE_AT = {1, 63, 64, 65, 127, 128, 129, 192, 255, 256, 257, 300}


def payload(b, h, dtype, rank, device):
    idx = torch.arange(b * h, device=device, dtype=torch.float32)
    return torch.sin(idx * 1e-3 + 0.37 * (rank + 1)).view(b, h).to(dtype)


def bit_diff(a, b):
    if a.numel() == 0:
        return 0, 0.0
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    n = (a.view(view) != b.view(view)).sum().item()
    d = (a.float() - b.float()).abs().max().item()
    return n, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=2048)
    ap.add_argument("--max-bs", type=int, default=300)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--channels", type=int, default=0,
                    help="override NCCL_MIN/MAX_NCHANNELS after the BI defaults")
    ap.add_argument("--no-bi-env", action="store_true",
                    help="skip vLLM's BI NCCL environment")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="bi_pynccl.json")
    args = ap.parse_args()

    if not args.no_bi_env:
        from vllm.model_executor.layers.batch_invariant import (
            override_envs_for_invariance,
        )

        override_envs_for_invariance()
    if args.channels:
        os.environ["NCCL_MIN_NCHANNELS"] = str(args.channels)
        os.environ["NCCL_MAX_NCHANNELS"] = str(args.channels)

    dist.init_process_group("gloo", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    comm = PyNcclCommunicator(dist.group.WORLD, device=device)
    dtype = getattr(torch, args.dtype)

    steps = []
    row0 = []
    saved = {}
    prev = None
    ref0 = None
    for b in range(1, args.max_bs + 1):
        x = payload(b, args.h, dtype, rank, device)
        y = x.clone()
        comm.all_reduce(y, out_tensor=y)
        torch.cuda.synchronize()
        if prev is not None:
            n, d = bit_diff(y[: b - 1], prev[: b - 1])
            steps.append({"b": b, "n_diff": n, "max": d, "total": (b - 1) * args.h})
        if ref0 is None:
            ref0 = y[0].clone()
        n, d = bit_diff(y[0], ref0)
        row0.append({"b": b, "n_diff": n, "max": d, "total": args.h})
        if b in SAVE_AT or b == args.max_bs:
            saved[b] = y.cpu()
        prev = y

    # same batch twice
    x = payload(255, args.h, dtype, rank, device)
    y1 = x.clone()
    comm.all_reduce(y1, out_tensor=y1)
    y2 = x.clone()
    comm.all_reduce(y2, out_tensor=y2)
    torch.cuda.synchronize()
    rep_n, rep_d = bit_diff(y1, y2)

    # full-prefix comparisons between saved batch sizes
    pairs = []
    keys = sorted(saved)
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            k = min(a, c)
            n, d = bit_diff(saved[a][:k], saved[c][:k])
            pairs.append({"a": a, "b": c, "rows": k, "n_diff": n,
                          "max": d, "total": k * args.h})

    result = {
        "tag": args.tag,
        "world": world,
        "h": args.h,
        "dtype": args.dtype,
        "algo": os.environ.get("NCCL_ALGO"),
        "proto": os.environ.get("NCCL_PROTO"),
        "min_channels": os.environ.get("NCCL_MIN_NCHANNELS"),
        "max_channels": os.environ.get("NCCL_MAX_NCHANNELS"),
        "steps_changed": sum(1 for s in steps if s["n_diff"] > 0),
        "steps_total": len(steps),
        "step_changed_at": [s["b"] for s in steps if s["n_diff"] > 0],
        "row0_changed_at": [r["b"] for r in row0 if r["n_diff"] > 0],
        "row0": row0,
        "steps": steps,
        "pairs": pairs,
        "repeat_255": {"n_diff": rep_n, "max": rep_d},
    }
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        print("steps with bit changes:", result["steps_changed"], "/",
              result["steps_total"])
        print("changed at b =", result["step_changed_at"][:40])
        print("row0 differs from b=1 at b =", result["row0_changed_at"][:40])
        for p in pairs:
            if p["a"] in (255, 1) or p["b"] in (256, 257):
                print(f"  {p['a']:4d} vs {p['b']:4d}: {p['n_diff']}/{p['total']}"
                      f" max {p['max']:.3e}")
        print("repeat 255:", result["repeat_255"])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
