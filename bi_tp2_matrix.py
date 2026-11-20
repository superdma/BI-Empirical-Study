"""Batch invariance at TP=2.

The 255/256 boundary was measured at TP=4. Two A6000 cards are a plausible
deployment target, and the all-reduce probe says TP=2 has no message-size
sensitivity at all, so the interesting question is whether the eager boundary
moves or disappears.

Three tests, one engine load:
  * 32 prompts: solo vs one batch, and the batch repeated;
  * 256-prompt batch vs the same prompts at 255;
  * the same 256 batch repeated.

Usage:
    python bi_tp2_matrix.py --out results/tier3/bi_tp2_matrix.json
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
           for i in range(1, 258)]


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
    ap.add_argument("--out", default="bi_tp2_matrix.json")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=2, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    result = {"env": {"model": MODEL, "tp": 2, "eager": True,
                      "bi": os.environ.get("VLLM_BATCH_INVARIANT"),
                      "max_tokens": args.max_tokens}, "cmp": {}}

    small = [target_ids] + filler_ids[:31]
    solo = [summarize(llm.generate([ids], sampling_params=sp)[0]) for ids in small]
    batch = [summarize(r) for r in llm.generate(small, sampling_params=sp)]
    batch2 = [summarize(r) for r in llm.generate(small, sampling_params=sp)]
    rows = [cmp(solo[i], batch[i]) for i in range(len(small))]
    result["cmp"]["solo_vs_batch_32"] = {
        "n": len(rows), "bitwise_equal": sum(r["bitwise"] for r in rows),
        "ids_equal": sum(r["ids"] for r in rows),
        "max_abs": max(r["max"] for r in rows)}
    rows2 = [cmp(batch[i], batch2[i]) for i in range(len(small))]
    result["cmp"]["batch_repeat_32"] = {
        "n": len(rows2), "bitwise_equal": sum(r["bitwise"] for r in rows2),
        "max_abs": max(r["max"] for r in rows2)}
    print("32 prompts", result["cmp"]["solo_vs_batch_32"], flush=True)

    big = [target_ids] + filler_ids[:255]
    out256 = [summarize(r) for r in llm.generate(big, sampling_params=sp)]
    out256b = [summarize(r) for r in llm.generate(big, sampling_params=sp)]
    out255 = [summarize(r) for r in llm.generate(big[:255], sampling_params=sp)]

    def agg(rows):
        return {"n": len(rows), "bitwise_equal": sum(r["bitwise"] for r in rows),
                "ids_equal": sum(r["ids"] for r in rows),
                "first_token_bad": sum(1 for r in rows if r["first"] is not None and r["first"] < 2),
                "max_abs": max(r["max"] for r in rows)}

    result["cmp"]["bs255_vs_bs256"] = agg([cmp(out255[i], out256[i]) for i in range(255)])
    result["cmp"]["bs256_repeat"] = agg([cmp(out256[i], out256b[i]) for i in range(256)])
    result["total_sec"] = time.time() - t0

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result["cmp"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
