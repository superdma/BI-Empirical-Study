import os, sys, json, time, struct

os.environ["VLLM_BATCH_INVARIANT"] = os.environ.get("BI", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TP = int(os.environ.get("TP", "4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.88"))
CHUNK = int(os.environ.get("CHUNK", "8192"))
PC = os.environ.get("PC", "1") == "1"
EAGER = os.environ.get("EAGER", "1") == "1"
ATTN = os.environ.get("ATTN", "FLASH_ATTN")
MOE = os.environ.get("MOE", "auto")
BS = int(os.environ.get("BS", "32"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
TEMP = float(os.environ.get("TEMP", "0.0"))
TOPP = float(os.environ.get("TOPP", "1.0"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_extra.json"))
TAG = os.environ.get("TAG", "extra")

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, max(BS, 4) + 1)]
LONG_PROMPT = ("The following is a long passage. " +
               "The quick brown fox jumps over the lazy dog. " * 1600)


def bits(x):
    return struct.pack(">d", float(x)).hex()


def summarize(reqout):
    o = reqout.outputs[0]
    tids = list(o.token_ids)
    lps = []
    for i, tid in enumerate(tids):
        d = o.logprobs[i] if o.logprobs is not None else None
        lps.append(None if d is None or tid not in d else float(d[tid].logprob))
    return {"token_ids": tids, "logprobs": lps, "text": o.text}


def cmp(a, b):
    ta, tb = a["token_ids"], b["token_ids"]
    la, lb = a["logprobs"] or [], b["logprobs"] or []
    first = None
    for i in range(min(len(la), len(lb))):
        if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i]):
            first = i
            break
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {
        "token_ids_equal": ta == tb,
        "logprobs_bitwise_equal": len(la) == len(lb) and first is None,
        "first_logprob_mismatch": first,
        "max_abs_logprob_diff": max(diffs) if diffs else None,
        "len_a": len(ta), "len_b": len(tb),
    }


def main():
    t0 = time.time()
    llm = LLM(
        model=MODEL, tensor_parallel_size=TP, gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_LEN, max_num_batched_tokens=CHUNK,
        enable_prefix_caching=PC, enable_chunked_prefill=True,
        attention_config={"backend": ATTN}, trust_remote_code=False,
        disable_log_stats=True, enforce_eager=EAGER, moe_backend=MOE,
    )
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS[:BS - 1]]
    long_ids = tok.encode(LONG_PROMPT, add_special_tokens=False)[: int(os.environ.get("LONG_TOKENS", "7600"))]

    greedy = SamplingParams(temperature=0.0, top_p=1.0, seed=SEED,
                            max_tokens=MAX_TOKENS, logprobs=5)
    sampled = SamplingParams(temperature=TEMP, top_p=TOPP, seed=SEED,
                             max_tokens=MAX_TOKENS, logprobs=5)

    res = {"env": {"tag": TAG, "model": MODEL, "tp": TP, "chunk": CHUNK, "pc": PC,
                   "eager": EAGER, "attn": ATTN, "moe": MOE, "bs": BS,
                   "max_tokens": MAX_TOKENS, "temp": TEMP, "top_p": TOPP,
                   "seed": SEED, "bi": os.environ.get("BI", "1"),
                   "long_tokens": len(long_ids),
                   "target_ids": target_ids},
           "cases": {}, "comparisons": {}, "timing_sec": {}, "errors": []}

    def run(name, prompts, sp, idx=0):
        try:
            t = time.time()
            out = llm.generate(prompts, sampling_params=sp)
            res["timing_sec"][name] = time.time() - t
            return summarize(out[idx])
        except Exception as e:
            res["errors"].append(f"{name}: {e!r}")
            return None

    single = run("single", [target_ids], greedy)
    res["cases"]["single"] = single
    batch_prompts = [target_ids] + filler_ids
    batch = run(f"batch{BS}", batch_prompts, greedy)
    res["cases"]["batch"] = batch
    rev_prompts = list(reversed(batch_prompts))
    rev = run("reverse", rev_prompts, greedy, idx=len(rev_prompts) - 1)
    res["cases"]["reverse"] = rev
    # generate returns outputs in input order for list prompts; find target by order
    if rev is not None:
        # target is last in rev_prompts, and outputs preserve prompt order
        pass
    repeat = run("repeat", [target_ids], greedy)
    res["cases"]["repeat"] = repeat

    sampled_single = sampled_batch = None
    if TEMP > 0:
        sampled_single = run("sampled_single", [target_ids], sampled)
        res["cases"]["sampled_single"] = sampled_single
        sampled_batch = run("sampled_batch", batch_prompts, sampled)
        res["cases"]["sampled_batch"] = sampled_batch

    long_single = long_batch = None
    if len(long_ids) > 0:
        # batch first, single second: with PC=0 both are fresh; with PC=1 the
        # single request hits the prefix cached by the batch request.
        long_batch = run("long_batch", [long_ids] + filler_ids[:3], greedy)
        res["cases"]["long_batch"] = long_batch
        long_single = run("long_single", [long_ids], greedy)
        res["cases"]["long_single"] = long_single

    refeed = None
    if single is not None:
        full = target_ids + single["token_ids"]
        sp = SamplingParams(temperature=0.0, top_p=1.0, seed=SEED,
                            max_tokens=1, prompt_logprobs=1)
        try:
            t = time.time()
            out = llm.generate([full], sampling_params=sp)[0]
            res["timing_sec"]["refeed"] = time.time() - t
            pls = out.prompt_logprobs
            dec = single["logprobs"] or []
            pairs = []
            for i, tid in enumerate(single["token_ids"]):
                pos = len(target_ids) + i
                d = pls[pos] if pos < len(pls) else None
                ent = None if d is None else (d.get(tid) or d.get(str(tid)))
                pairs.append(None if ent is None else (float(dec[i]), float(ent.logprob)))
            valid = [p for p in pairs if p is not None]
            refeed = {"available": True, "n_compared": len(valid),
                      "all_bitwise_equal": bool(valid) and all(bits(x) == bits(y) for x, y in valid),
                      "max_abs_diff": max((abs(x - y) for x, y in valid), default=None),
                      "first_mismatch": next((i for i, p in enumerate(pairs)
                                              if p is not None and bits(p[0]) != bits(p[1])), None)}
        except Exception as e:
            res["errors"].append(f"refeed: {e!r}")
            refeed = {"available": False}
    res["cases"]["refeed"] = refeed

    if single and batch: res["comparisons"]["single_vs_batch"] = cmp(single, batch)
    if single and rev: res["comparisons"]["single_vs_reverse"] = cmp(single, rev)
    if single and repeat: res["comparisons"]["single_vs_repeat"] = cmp(single, repeat)
    if sampled_single and sampled_batch:
        res["comparisons"]["sampled_single_vs_batch"] = cmp(sampled_single, sampled_batch)
    if long_single and long_batch:
        res["comparisons"]["long_batch_vs_single"] = cmp(long_batch, long_single)
    if refeed is not None:
        res["comparisons"]["decode_vs_refeed"] = refeed

    res["env"]["total_sec"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    brief = {"tag": TAG, "env": {k: v for k, v in res["env"].items()
                                 if k not in ("target_ids",)},
             "comparisons": res["comparisons"], "errors": res["errors"],
             "timing": res["timing_sec"]}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
