"""Does paged decode attention depend on how many sequences are in the batch?

This mirrors the call vLLM's FlashAttention backend makes in the decode step:
one query token per sequence, paged KV cache, num_splits=1. Sequence 0 gets
the same KV content in every batch size; only the number of neighbours in the
batch changes.

Usage:
    python bi_attn_shape.py --out results/tier2/bi_attn_shape.json
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402

HEADS, KV_HEADS, DIM, BLOCK = 8, 1, 128, 16
SIZES = [1, 64, 128, 192, 255, 256, 257, 300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_attn_shape.json")
    args = ap.parse_args()

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = "cuda"

    max_blocks = 8
    n_blocks = 4096
    key_cache = torch.randn(n_blocks, BLOCK, KV_HEADS, DIM, device=device,
                            dtype=torch.bfloat16) * 0.1
    value_cache = torch.randn(n_blocks, BLOCK, KV_HEADS, DIM, device=device,
                              dtype=torch.bfloat16) * 0.1
    q_all = torch.randn(320, HEADS, DIM, device=device, dtype=torch.bfloat16)

    rows = []
    for kv_len in (11, 16, 31, 32):
        out_ref = None
        for n in SIZES:
            q = q_all[:n].contiguous()
            out = torch.empty(n, HEADS, DIM, device=device, dtype=torch.bfloat16)
            seqused = torch.full((n,), kv_len, device=device, dtype=torch.int32)
            # sequence i uses blocks [2i, 2i+1)
            block_table = torch.zeros(n, max_blocks, device=device,
                                      dtype=torch.int32)
            for i in range(n):
                block_table[i, : (kv_len + BLOCK - 1) // BLOCK] = torch.arange(
                    2 * i, 2 * i + (kv_len + BLOCK - 1) // BLOCK,
                    device=device, dtype=torch.int32)
            cu = torch.arange(0, n + 1, device=device, dtype=torch.int32)
            flash_attn_varlen_func(
                q=q, k=key_cache, v=value_cache, out=out,
                cu_seqlens_q=cu, max_seqlen_q=1,
                seqused_k=seqused, max_seqlen_k=kv_len,
                softmax_scale=DIM ** -0.5, causal=True,
                block_table=block_table, num_splits=1,
            )
            torch.cuda.synchronize()
            if out_ref is None:
                out_ref = out[0].clone()
            a = out[0]
            n_diff = (a.view(torch.int16) != out_ref.view(torch.int16)).sum().item()
            mx = (a.float() - out_ref.float()).abs().max().item()
            rows.append({"kv_len": kv_len, "n_seq": n, "row0_diff": n_diff,
                         "elems": HEADS * DIM, "max": mx})
            print(f"kv={kv_len:3d} n={n:4d} row0 {n_diff:5d}/{HEADS * DIM}"
                  f" max {mx:.3e}", flush=True)
            del out, q
            torch.cuda.empty_cache()

    with open(args.out, "w") as handle:
        json.dump({"rows": rows}, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
