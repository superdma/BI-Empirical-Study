"""Vary the composition of a 256-request decode batch.

The 255/256 boundary survives every per-operator test, so the question is what
exactly changes at 256: the number of active sequences, the scheduler's view of
the batch, or the position a request occupies. Each run below changes one of
those while keeping the prompts identical.

Runs (all greedy, 8 tokens, one engine load):

  c256  256 prompts, baseline
  c255  first 255 prompts
  s255  first 255 prompts + one request that stops after its first token
  r256  the same 256 prompts in reverse order

The layer dump is written only for c256 and c255; the other runs skip it.

Usage:
    python bi_composition.py --out results/tier3/bi_composition.json
"""
import argparse
import json
import os
import shutil
import struct
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.expanduser(os.environ.get("BI_DUMP", "~/bi_norm_meta"))
os.environ["BI_DUMP"] = DUMP
os.environ["BI_NORM_META"] = "1"
os.environ["PYTHONPATH"] = os.path.join(ROOT, "bi_hook") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 258)]


def set_tag(tag):
    path = os.path.join(DUMP, "current_tag")
    if tag:
        os.makedirs(DUMP, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(tag + "\n")
    elif os.path.exists(path):
        os.remove(path)


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
    ap.add_argument("--out", default="bi_composition.json")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    shutil.rmtree(DUMP, ignore_errors=True)
    os.makedirs(os.path.join(DUMP, "meta"), exist_ok=True)
    set_tag("")

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
    prompts = [target_ids] + filler_ids[:255]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)
    short = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, logprobs=5)

    plan = [
        ("c256", prompts, [sp] * len(prompts), True),
        ("c255", prompts[:255], [sp] * 255, True),
        ("c255b", prompts[:255], [sp] * 255, False),
        ("s255", prompts[:255] + [filler_ids[255]], [sp] * 255 + [short], False),
        ("r256", list(reversed(prompts)), [sp] * len(prompts), False),
    ]
    runs = {}
    for tag, batch, sps, dump in plan:
        set_tag(tag if dump else "")
        t = time.time()
        outs = llm.generate(batch, sampling_params=sps)
        runs[tag] = [summarize(o) for o in outs]
        print(tag, len(batch), f"{time.time() - t:.1f}s", flush=True)
    set_tag("")

    result = {
        "env": {"model": MODEL, "tp": 4, "eager": True, "bi": os.environ.get("VLLM_BATCH_INVARIANT"),
                "max_tokens": args.max_tokens, "max_num_seqs": 512},
        "runs": {k: {"n": len(v)} for k, v in runs.items()},
        "cmp": {},
        "total_sec": time.time() - t0,
    }

    def pair(name, left, li, right, ri):
        rows = [cmp(left[a], right[b]) for a, b in zip(li, ri)]
        bad = [i for i, r in enumerate(rows) if not r["bitwise"]]
        result["cmp"][name] = {
            "n": len(rows),
            "bitwise_equal": len(rows) - len(bad),
            "bad": len(bad),
            "bad_index": bad[:12],
            "first_token_bad": sum(1 for r in rows if r["first"] is not None and r["first"] < 2),
            "first_token_hist": {str(k): sum(1 for r in rows if r["first"] == k)
                                 for k in sorted({r["first"] for r in rows if r["first"] is not None})},
            "max_abs": max((r["max"] or 0.0) for r in rows),
        }

    # The short request is not compared: it stops after one token.
    idx = list(range(255))
    pair("c255_vs_c256", runs["c255"], idx, runs["c256"], idx)
    pair("c255_vs_c255b", runs["c255"], idx, runs["c255b"], idx)
    pair("s255_vs_c255", runs["s255"], idx, runs["c255"], idx)
    pair("s255_vs_c256", runs["s255"], idx, runs["c256"], idx)
    # r256 holds the same prompts in reverse; map back by prompt index.
    rev = list(reversed(range(256)))
    pair("r256_vs_c256", runs["r256"], list(range(256)), runs["c256"], rev)

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({k: (v["bitwise_equal"], v["bad"], v["first_token_bad"])
                      for k, v in result["cmp"].items()}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
