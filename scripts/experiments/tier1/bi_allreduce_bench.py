"""Is the cross-GPU all-reduce sensitive to tensor shape?

This runs without a model: four ranks hold the same data, all-reduce it and
compare the IEEE-754 bits of the result across (a) different reshapes of the
same payload, (b) payloads that differ only by appended zero rows, and
(c) NCCL algorithms chosen through the environment.

    torchrun --nproc_per_node=4 bi_allreduce_bench.py --out allreduce_4.json
    NCCL_ALGO=Tree torchrun --nproc_per_node=4 bi_allreduce_bench.py --tag tree --out t.json
"""
import argparse
import json
import os

import torch
import torch.distributed as dist


H = int(os.environ.get("H", "4096"))


def bit_view(t):
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    return t.view(torch.int32)


def all_reduce_sum(x):
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return y


def compare(a, b):
    av, bv = bit_view(a).flatten(), bit_view(b).flatten()
    n = int((av != bv).sum().item())
    diff = (a.flatten().float() - b.flatten().float()).abs().max().item()
    return {"n_diff": n, "total": int(av.numel()), "max_abs": diff, "bitwise": n == 0}


def payload(n, dtype, device, rank):
    """Rank-dependent values.

    Identical payloads would make every reduction order give the exact same
    answer (4 * x is a power-of-two scale), so the test has to give each rank
    different data for the reduction order to matter.
    """
    x = torch.arange(n, device=device, dtype=torch.float32) * 1e-3
    return (x + 0.37 * (rank + 1)).sin().to(dtype)


def reshape_test(dtype, device, rank, ws):
    n = 256 * H
    base = payload(n, dtype, device, rank)
    shapes = [(256, H), (128, 2 * H), (512, H // 2), (1024, H // 4), (H, 256)]
    ref = None
    rows = []
    for shape in shapes:
        y = all_reduce_sum(base.view(shape))
        if ref is None:
            ref = y.clone()
        r = compare(ref, y)
        r.update({"shape": list(shape), "dtype": str(dtype).split(".")[-1],
                  "world": ws})
        rows.append(r)
    return rows


def pad_test(dtype, device, rank, ws):
    """Same leading rows, different batch size: compare (B,H) with (B+1,H).

    The extra row is zero, so a shape-independent elementwise reduction must
    leave the first B rows untouched.
    """
    h = H
    max_b = 512
    base = payload(max_b * h, dtype, device, rank).view(max_b, h)
    results = {}
    for b in (63, 64, 65, 127, 128, 129, 191, 255, 256, 257, 511, 512):
        x = torch.zeros((b, h), dtype=dtype, device=device)
        x[:min(b, max_b)] = base[:min(b, max_b)]
        results[b] = all_reduce_sum(x)
    rows = []
    for b1, b2 in ((63, 64), (64, 65), (127, 128), (128, 129),
                   (255, 256), (256, 257), (511, 512)):
        r = compare(results[b1], results[b2][:b1])
        r.update({"dtype": str(dtype).split(".")[-1], "world": ws,
                  "pair": [b1, b2], "test": "pad"})
        rows.append(r)
    return rows


def algo_test(dtype, device, rank, ws):
    """One fixed shape, two runs in the same process: after the first reduce
    NCCL keeps its choice, so the real algorithm comparison happens across
    processes (see --tag).  This records the result for cross-run diffing."""
    n = 256 * H
    base = payload(n, dtype, device, rank).view(256, H)
    y = all_reduce_sum(base)
    return {"tag": rank, "dtype": str(dtype).split(".")[-1], "world": ws,
            "checksum": float(y.double().sum().item()),
            "first8": [float(v) for v in y.flatten()[:8].tolist()],
            "bits_first8": [int(v) for v in bit_view(y).flatten()[:8].tolist()]}


def order_test(dtype, device, rank, ws):
    """NCCL's result against hand-made summation orders.

    The same data is summed in rank order and in reverse rank order with the
    working dtype; if NCCL matches one of them, the reduction order it picked
    is visible, and a shape-dependent switch of order would show up as a
    difference between runs.
    """
    n = 256 * H
    x = payload(n, dtype, device, rank)
    y = all_reduce_sum(x)
    gathered = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gathered, x)
    exact = sum(g.double() for g in gathered).to(dtype)

    def seq(order):
        acc = gathered[order[0]].clone()
        for idx in order[1:]:
            acc = acc + gathered[idx]
        return acc

    fwd = seq(list(range(ws)))
    rev = seq(list(reversed(range(ws))))
    rows = []
    for name, other in (("exact_fp64", exact), ("rank_order", fwd),
                        ("reverse_order", rev)):
        r = compare(y, other)
        r.update({"dtype": str(dtype).split(".")[-1], "world": ws,
                  "against": name, "test": "order"})
        rows.append(r)
    return rows


def threshold_test(dtype, device, rank, ws):
    """Where does the result start to change when the message grows by one row?

    Same leading rows, batch size from 1 to --max-bs.  A non-zero n_diff at
    b -> b+1 means NCCL changed how it splits or orders the reduction.
    """
    max_b = int(os.environ.get("MAX_BS", "300"))
    base = payload(max_b * H, dtype, device, rank).view(max_b, H)
    prev = all_reduce_sum(base[:1].clone())
    rows = []
    for b in range(2, max_b + 1):
        cur = all_reduce_sum(base[:b].clone())
        r = compare(prev[:b - 1], cur[:b - 1])
        r.update({"dtype": str(dtype).split(".")[-1], "world": ws, "test": "threshold",
                  "pair": [b - 1, b], "hidden": H})
        rows.append(r)
        prev = cur
    return rows


def payload_pos(n, dtype, device, rank):
    """All-positive payload: no cancellation, so summation order matters less."""
    x = torch.arange(n, device=device, dtype=torch.float32) * 1e-3
    return (1.0 + 0.4 * (x + 0.37 * (rank + 1)).sin()).to(dtype)


def sparse_test(dtype, device, rank, ws):
    """bs=1 against much larger batches, looking only at row 0.

    This is the exact question the engine sweep answers: does the first
    request's numbers change when it shares the step with 255 others?
    """
    max_b = 300
    base = payload(max_b * H, dtype, device, rank).view(max_b, H)
    ref = all_reduce_sum(base[:1].clone())
    rows = []
    for b in (2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 240,
              255, 256, 257, 288, 300):
        cur = all_reduce_sum(base[:b].clone())
        r = compare(ref, cur[:1])
        r.update({"dtype": str(dtype).split(".")[-1], "world": ws,
                  "test": "sparse", "pair": [1, b], "hidden": H})
        rows.append(r)
    return rows


def pos_test(dtype, device, rank, ws):
    """Same sparse comparison, but with all-positive data."""
    max_b = 300
    x = torch.arange(max_b * H, device=device, dtype=torch.float32) * 1e-3
    base = (1.0 + 0.4 * (x + 0.37 * (rank + 1)).sin()).to(dtype).view(max_b, H)
    ref = all_reduce_sum(base[:1].clone())
    rows = []
    for b in (2, 8, 32, 64, 128, 192, 240, 255, 256, 257, 300):
        cur = all_reduce_sum(base[:b].clone())
        r = compare(ref, cur[:1])
        r.update({"dtype": str(dtype).split(".")[-1], "world": ws,
                  "test": "sparse_pos", "pair": [1, b], "hidden": H})
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=os.environ.get("NCCL_ALGO", "default"))
    ap.add_argument("--tests", default="reshape,pad,algo,order")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    out = {"tag": args.tag, "world": ws, "algo_env": os.environ.get("NCCL_ALGO", ""),
           "hidden": int(os.environ.get("H", "2048")),
           "proto_env": os.environ.get("NCCL_PROTO", ""),
           "device_name": torch.cuda.get_device_name(rank), "rows": []}
    wanted = args.tests.split(",")
    if "reshape" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += reshape_test(dtype, device, rank, ws)
    if "pad" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += pad_test(dtype, device, rank, ws)
    if "algo" in wanted:
        out["algo"] = [algo_test(torch.bfloat16, device, rank, ws)]
    if "order" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += order_test(dtype, device, rank, ws)
    if "pos" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += pos_test(dtype, device, rank, ws)
    if "sparse" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += sparse_test(dtype, device, rank, ws)
    if "threshold" in wanted:
        for dtype in (torch.bfloat16, torch.float32):
            out["rows"] += threshold_test(dtype, device, rank, ws)

    dist.barrier()
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps({"tag": args.tag, "world": ws,
                          "rows": out["rows"], "algo": out.get("algo")}, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
