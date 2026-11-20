"""Separate the effect of the prompt set from the effect of generation length.

Chapter 6 reported 1/32 for eager, comparing one request at a time with the
whole batch. A later run with a different set of 32 questions and 8 tokens
gives 24/32. Before either number goes back into the report, this control
runs both prompt sets at both lengths in one process and reports the same
solo-vs-batch comparison for each.

Usage:
    python bi_prompt_control.py --out results/tier4/bi_prompt_control.json
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
FILLERS31 = [f"Question {i}: describe the number {i} in one short sentence."
             for i in range(1, 32)]
QUESTIONS32 = [
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
    return {"bitwise": first is None, "first": first, "max": max(diffs) if diffs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_prompt_control.json")
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    old_ids = ([tok.encode(TARGET, add_special_tokens=False)] +
               [tok.encode(p, add_special_tokens=False) for p in FILLERS31])
    new_ids = [tok.encode(p, add_special_tokens=False) for p in QUESTIONS32]

    plan = [
        ("old_prompts_16", old_ids, 16, False),
        ("new_prompts_16", new_ids, 16, True),
        ("old_prompts_8", old_ids, 8, True),
    ]
    out = {}
    for label, ids, mt, ignore_eos in plan:
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=mt,
                            logprobs=5, ignore_eos=ignore_eos)
        solo = [summarize(llm.generate([x], sampling_params=sp)[0]) for x in ids]
        batch = [summarize(r) for r in llm.generate(ids, sampling_params=sp)]
        rows = [cmp(batch[i], solo[i]) for i in range(len(ids))]
        bad = [i for i, r in enumerate(rows) if not r["bitwise"]]
        out[label] = {
            "n": len(ids), "max_tokens": mt, "ignore_eos": ignore_eos,
            "bitwise_equal": len(ids) - len(bad), "changed": len(bad),
            "first_token_bad": sum(1 for r in rows if r["first"] is not None and r["first"] < 2),
            "max_abs": max((r["max"] or 0.0) for r in rows),
            "bad_index": bad[:12],
        }
        print(label, out[label], flush=True)

    result = {"env": {"model": MODEL, "tp": 4, "eager": True},
              "runs": out, "total_sec": time.time() - t0}
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
