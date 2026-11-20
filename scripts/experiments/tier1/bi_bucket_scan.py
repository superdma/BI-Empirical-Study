"""Batch-size scan around CUDA-graph bucket boundaries, compiled mode.

CUDA graphs are captured per batch bucket, so 31, 32, 33 may all replay the
same graph padded to 32.  If the padded shape were what reaches the kernels,
the logprobs would be identical across a bucket and jump at its edge.  This
loads the model once and walks the sizes where buckets usually end.
"""
import json
import os
import struct
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_bucket_scan.json"))
SIZES = [int(v) for v in os.environ.get(
    "SIZES", "31,32,33,63,64,65,127,128,129,255,256,257").split(",")]
EAGER = os.environ.get("EAGER", "0") == "1"

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, max(SIZES) + 1)]


def bits(x):
    return struct.pack(">d", float(x)).hex()


def summarize(out):
    o = out.outputs[0]
    lps = []
    for i, tid in enumerate(o.token_ids):
        d = o.logprobs[i] if o.logprobs is not None else None
        lps.append(None if d is None or tid not in d else float(d[tid].logprob))
    return {"token_ids": list(o.token_ids), "logprobs": lps}


def cmp(a, b):
    la, lb = a["logprobs"], b["logprobs"]
    first = next((i for i in range(min(len(la), len(lb)))
                  if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i])), None)
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {"token_ids_equal": a["token_ids"] == b["token_ids"],
            "bitwise": first is None, "first_mismatch": first,
            "max_abs": max(diffs) if diffs else None}


def main():
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=int(os.environ.get("TP", "4")),
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              enforce_eager=EAGER, max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, logprobs=5,
                        ignore_eos=True)

    single = summarize(llm.generate([target_ids], sampling_params=sp)[0])
    res = {"env": {"model": MODEL, "eager": EAGER, "sizes": SIZES,
                   "bi": os.environ.get("VLLM_BATCH_INVARIANT", "1")},
           "single": single, "rows": []}
    for bs in SIZES:
        t = time.time()
        prompts = [target_ids] + filler_ids[:bs - 1]
        out = llm.generate(prompts, sampling_params=sp)[0]
        rec = {"bs": bs, "sec": time.time() - t}
        rec.update(cmp(single, summarize(out)))
        rec["max_abs"] = rec["max_abs"]
        res["rows"].append(rec)
        print(json.dumps({k: rec[k] for k in ("bs", "bitwise", "first_mismatch",
                                              "max_abs", "sec")}), flush=True)

    res["total_sec"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
