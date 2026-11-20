"""Sweep the cached tail length: fresh prefill vs prefix-cache hit.

For each tail length T the prompt is  a long passage (~6000 tokens) + T extra
tokens.  The first request runs with an empty cache (fresh); the second request
of the same prompt hits the prefix cache and only prefills the tail.  We
compare the first generated logprobs bit for bit.

A unique leading token per T keeps the cache from leaking between cases; both
calls of a case use the same prompt, so the comparison stays clean.
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
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_tail_scan.json"))
BASE_TOKENS = int(os.environ.get("BASE_TOKENS", "6000"))
TAILS = [int(v) for v in os.environ.get(
    "TAILS", "1,2,4,8,16,32,64,128,192,256,320,384,448,512").split(",")]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8"))

PASSAGE = ("The following is a long passage about rivers, weather and small towns. "
           "The quick brown fox jumps over the lazy dog. " * 700)
TAIL_TEXT = ("In this section the text keeps going with ordinary sentences so the "
             "tail length can be varied without changing anything else. ")


def bits(x):
    return struct.pack(">d", float(x)).hex()


def logprobs_of(out):
    o = out.outputs[0]
    vals = []
    for i, tid in enumerate(o.token_ids):
        d = o.logprobs[i] if o.logprobs is not None else None
        vals.append(None if d is None or tid not in d else float(d[tid].logprob))
    return list(o.token_ids), vals


def main():
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=int(os.environ.get("TP", "4")),
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=True,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              enforce_eager=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    base = tok.encode(PASSAGE, add_special_tokens=False)[:BASE_TOKENS]
    tail_all = tok.encode(TAIL_TEXT, add_special_tokens=False)
    while len(tail_all) < max(TAILS):
        tail_all += tail_all
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS,
                        logprobs=5, ignore_eos=True)

    # warm up the graph and the tokenizer path
    llm.generate([base[:64]], sampling_params=SamplingParams(temperature=0.0, max_tokens=2))

    salt = 20000
    res = {"env": {"model": MODEL, "base_tokens": len(base), "tails": TAILS,
                   "max_tokens": MAX_TOKENS, "pc": True, "eager": True},
           "rows": [], "errors": []}
    for i, tail in enumerate(TAILS):
        prompt = [salt + i] + base + tail_all[:tail]
        rec = {"tail": tail, "prompt_tokens": len(prompt)}
        try:
            fresh = llm.generate([prompt], sampling_params=sp)[0]
            hit = llm.generate([prompt], sampling_params=sp)[0]
        except Exception as exc:  # keep going, report the failure
            rec["error"] = repr(exc)
            res["rows"].append(rec)
            continue
        f_ids, f_lp = logprobs_of(fresh)
        h_ids, h_lp = logprobs_of(hit)
        first = next((j for j in range(min(len(f_lp), len(h_lp)))
                      if f_lp[j] is None or h_lp[j] is None
                      or bits(f_lp[j]) != bits(h_lp[j])), None)
        diffs = [abs(a - b) for a, b in zip(f_lp, h_lp) if a is not None and b is not None]
        rec.update({"token_ids_equal": f_ids == h_ids,
                    "bitwise": first is None,
                    "first_mismatch": first,
                    "max_abs": max(diffs) if diffs else None,
                    "fresh_lp": f_lp, "hit_lp": h_lp})
        res["rows"].append(rec)
        print(json.dumps({k: rec[k] for k in ("tail", "bitwise", "first_mismatch",
                                              "max_abs", "token_ids_equal")}))

    res["total_sec"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
