"""Run the same shape matrix in whatever engine mode the environment selects.

Two shapes matter for the deployment recommendation: the 32-prompt batch from
chapter 6 and the decode sizes around 256 and 448. The script does not decide
the mode itself; `--compiled` drops `enforce_eager`, and the launcher sets
`VLLM_USE_BREAKABLE_CUDAGRAPH` when it wants that control.

Usage:
    python bi_shape_matrix.py --label eager --out results/tier4/bi_shapes_eager.json
    python bi_shape_matrix.py --label compiled --compiled \
      --out results/tier4/bi_shapes_compiled.json
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
PROMPTS32 = [
    "Explain why the sky is blue in two sentences.",
    "List three prime numbers larger than 100.",
    "What is the capital of Japan?",
    "Summarize the plot of Hamlet in one sentence.",
    "Convert 72 Fahrenheit to Celsius.",
    "Name the largest planet in the solar system.",
    "Write a haiku about rain.",
    "What does HTTP stand for?",
    "Give one reason to use a hash table.",
    "Translate 'good morning' into French.",
    "What is 17 multiplied by 23?",
    "Describe the water cycle briefly.",
    "Who wrote 'Pride and Prejudice'?",
    "Explain what a prime factor is.",
    "Name two renewable energy sources.",
    "What year did the Berlin Wall fall?",
    "Define recursion in one sentence.",
    "List the first three Fibonacci numbers.",
    "What is the boiling point of water at sea level?",
    "Explain the difference between TCP and UDP briefly.",
    "Name a famous painting by Van Gogh.",
    "What is the chemical symbol for gold?",
    "Describe how photosynthesis works in one sentence.",
    "What is the square root of 144?",
    "Give an example of a mammal that lives in water.",
    "Explain what an algorithm is.",
    "What language is spoken in Brazil?",
    "Name the three branches of the US government.",
    "What is the speed of light?",
    "Describe a black hole in one sentence.",
    "What is the difference between weather and climate?",
    "Name a common unit of electrical resistance.",
]
DECODE_SIZES = [255, 256, 448, 512]


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
    ap.add_argument("--out", default="bi_shapes.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--compiled", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=not args.compiled,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    # 32 prompts: one at a time, then together.
    ids32 = [tok.encode(p, add_special_tokens=False) for p in PROMPTS32]
    solo = []
    for i, ids in enumerate(ids32):
        solo.append(summarize(llm.generate([ids], sampling_params=sp)[0]))
        print("solo", i, flush=True)
    t = time.time()
    batch32 = [summarize(o) for o in llm.generate(ids32, sampling_params=sp)]
    print("batch32", f"{time.time() - t:.1f}s", flush=True)

    # Decode sizes: the same first 255 prompts plus fillers.
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    prompts = [target_ids] + filler_ids
    decode = {}
    for n in DECODE_SIZES:
        t = time.time()
        decode[n] = [summarize(o) for o in llm.generate(prompts[:n], sampling_params=sp)]
        print("decode", n, f"{time.time() - t:.1f}s", flush=True)

    def agg(pairs):
        bad = [i for i, r in enumerate(pairs) if not r["bitwise"]]
        return {"n": len(pairs), "bitwise_equal": len(pairs) - len(bad), "changed": len(bad),
                "first_token_bad": sum(1 for r in pairs if r["first"] is not None and r["first"] < 2),
                "max_abs": max((r["max"] or 0.0) for r in pairs)}

    rows = {"solo_vs_batch_32": agg([cmp(batch32[i], solo[i]) for i in range(32)])}
    for n in DECODE_SIZES:
        shared = min(255, n)
        rows[f"decode_{n}_vs_255"] = agg([cmp(decode[n][i], decode[255][i]) for i in range(shared)])

    result = {"env": {"model": MODEL, "tp": 4, "compiled": args.compiled,
                      "breakable_cudagraph": os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"),
                      "max_tokens": args.max_tokens, "label": args.label},
              "comp": rows, "total_sec": time.time() - t0}
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
