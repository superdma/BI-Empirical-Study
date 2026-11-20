"""Map the decode batch-size boundary above 256.

Every earlier measurement compared exactly two batch sizes, 255 and 256. This
sweep runs the same prompts at 255, 256, 257, 288, 320, 384, 448 and 512 and
compares each run with the 255 baseline, so the boundary shows up as a curve
instead of a single step.

Usage:
    python bi_boundary_sweep.py --out results/tier4/bi_boundary_sweep.json
"""
import argparse
import json
import os
import struct
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 520)]
SIZES = [255, 256, 257, 288, 320, 384, 448, 512]


def bits(value):
    return struct.pack(">d", float(value)).hex()


def summarize(result):
    o = result.outputs[0]
    ids = list(o.token_ids)
    lps = []
    for i, tid in enumerate(ids):
        row = o.logprobs[i] if o.logprobs is not None else None
        lps.append(None if row is None or tid not in row else float(row[tid].logprob))
    return {"token_ids": ids, "logprobs": lps}


def cmp(a, b):
    la, lb = a["logprobs"], b["logprobs"]
    first = None
    for i in range(min(len(la), len(lb))):
        if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i]):
            first = i
            break
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {"ids": a["token_ids"] == b["token_ids"], "bitwise": first is None,
            "first": first, "max": max(diffs) if diffs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_boundary_sweep.json")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    prompts = [target_ids] + filler_ids
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    runs = {}
    seconds = {}
    for n in SIZES:
        t = time.time()
        outs = llm.generate(prompts[:n], sampling_params=sp)
        runs[n] = [summarize(o) for o in outs]
        seconds[n] = round(time.time() - t, 1)
        print(n, seconds[n], "s", flush=True)

    base = runs[255]
    rows = {}
    for n in SIZES:
        shared = min(255, n)
        pairs = [cmp(runs[n][i], base[i]) for i in range(shared)]
        bad = [i for i, r in enumerate(pairs) if not r["bitwise"]]
        rows[str(n)] = {
            "n": shared,
            "bitwise_equal": shared - len(bad),
            "changed": len(bad),
            "first_token_bad": sum(1 for r in pairs if r["first"] is not None and r["first"] < 2),
            "max_abs": max((r["max"] or 0.0) for r in pairs),
            "seconds": seconds[n],
        }

    result = {"env": {"model": MODEL, "tp": 4, "eager": True,
                      "max_tokens": args.max_tokens, "sizes": SIZES},
              "vs_255": rows, "total_sec": time.time() - t0}
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
