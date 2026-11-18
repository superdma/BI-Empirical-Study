import os, sys, json, struct, time, gc, traceback

os.environ["VLLM_BATCH_INVARIANT"] = os.environ.get("BI", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Avoid FlashInfer JIT for the top-p/top-k sampler (ninja is not installed).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
import vllm.envs as vllm_envs

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TP = int(os.environ.get("TP", "4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.88"))
CHUNK = int(os.environ.get("CHUNK", "8192"))
PC = os.environ.get("PC", "1") == "1"
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_results.json"))
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "7600"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
EAGER = os.environ.get("EAGER", "0") == "1"

TARGET = "The capital of France is"
LONG_PROMPT = ("The following is a long passage. " + "The quick brown fox jumps over the lazy dog. " * 900)
FILLERS = [
    f"Question {i}: describe the number {i} in one short sentence."
    for i in range(1, 32)
]


def bits(x):
    return struct.pack(">d", float(x)).hex()


def extract(out):
    o = out.outputs[0]
    tids = list(o.token_ids)
    lps = []
    for i, tid in enumerate(tids):
        d = o.logprobs[i] if o.logprobs is not None else None
        if d is None or tid not in d:
            lps.append(None)
        else:
            lps.append(float(d[tid].logprob))
    return tids, lps


def compare(a, b):
    ta, la = extract(a)
    tb, lb = extract(b)
    tok_eq = ta == tb
    lp_eq = (
        len(la) == len(lb)
        and all(x is not None and y is not None and bits(x) == bits(y)
                for x, y in zip(la, lb))
    )
    diffs = [
        abs(x - y) for x, y in zip(la, lb)
        if x is not None and y is not None
    ]
    first = None
    for i, (x, y) in enumerate(zip(la, lb)):
        if x is None or y is None or bits(x) != bits(y):
            first = i
            break
    return {
        "token_ids_equal": tok_eq,
        "logprobs_bitwise_equal": lp_eq,
        "len_a": len(ta),
        "len_b": len(tb),
        "max_abs_logprob_diff": max(diffs) if diffs else None,
        "first_logprob_mismatch": first,
        "token_ids_a": ta,
        "token_ids_b": tb,
        "logprobs_a": la,
        "logprobs_b": lb,
    }


def main():
    t_start = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_LEN,
        max_num_batched_tokens=CHUNK,
        enable_prefix_caching=PC,
        enable_chunked_prefill=True,
        attention_config={"backend": "FLASH_ATTN"},
        trust_remote_code=False,
        disable_log_stats=True,
        enforce_eager=EAGER,
    )
    try:
        from vllm.model_executor.layers.batch_invariant import _batch_invariant_MODE
        bi_mode = bool(_batch_invariant_MODE)
    except Exception:
        bi_mode = None
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        logprobs=5,
        seed=0,
    )

    target_ids = tokenizer.encode(TARGET, add_special_tokens=True)
    filler_ids = [tokenizer.encode(p, add_special_tokens=True) for p in FILLERS]

    # 1) single request (bs=1), cold-ish
    t0 = time.time()
    single = llm.generate([target_ids], sampling_params=sp)[0]
    t_single = time.time() - t0

    # 2) target + 31 fillers (bs=32), target first
    t0 = time.time()
    batch = llm.generate(
        [target_ids] + filler_ids,
        sampling_params=sp,
    )
    t_batch = time.time() - t0
    batch_target = batch[0]

    # 3) reverse order, target last
    t0 = time.time()
    rev = llm.generate(
        list(reversed([target_ids] + filler_ids)),
        sampling_params=sp,
    )
    t_rev = time.time() - t0
    rev_target = rev[-1]

    # 4) repeat single (prefix-cache hit if PC=True)
    t0 = time.time()
    repeat = llm.generate([target_ids], sampling_params=sp)[0]
    t_repeat = time.time() - t0

    # 5) long prompt: ~7.6k tokens, exercises chunked prefill
    long_ids = tokenizer.encode(LONG_PROMPT, add_special_tokens=True)[:min(LONG_TOKENS, MAX_LEN - 128)]
    t0 = time.time()
    errors = {}
    try:
        long_single = llm.generate([long_ids], sampling_params=sp)[0]
    except Exception as e:
        traceback.print_exc()
        long_single = None
        errors["long_prompt"] = repr(e)
    t_long = time.time() - t0

    # 6) decode vs re-feed
    gen_ids = list(single.outputs[0].token_ids)
    full_ids = list(target_ids) + gen_ids
    sp_pl = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        seed=0,
    )
    try:
        refeed = llm.generate(
            [full_ids],
            sampling_params=sp_pl,
        )[0]
    except Exception as e:
        traceback.print_exc()
        refeed = None
        errors["refeed"] = repr(e)

    refeed_cmp = {"available": refeed is not None and refeed.prompt_logprobs is not None}
    if refeed is not None and refeed.prompt_logprobs is not None:
        dec = single.outputs[0].logprobs
        pls = refeed.prompt_logprobs
        pairs = []
        for i, tid in enumerate(gen_ids):
            pos = len(target_ids) + i
            d_dec = dec[i] if dec is not None else None
            d_ref = pls[pos] if pos < len(pls) else None
            if d_dec is None or d_ref is None or tid not in d_dec or tid not in d_ref:
                pairs.append(None)
            else:
                pairs.append((float(d_dec[tid].logprob),
                              float(d_ref[tid].logprob)))
        valid = [p for p in pairs if p is not None]
        refeed_cmp.update({
            "n_generated": len(gen_ids),
            "n_compared": len(valid),
            "all_bitwise_equal": all(bits(a) == bits(b) for a, b in valid)
                                 if valid else False,
            "max_abs_diff": max((abs(a - b) for a, b in valid), default=None),
            "first_mismatch": next(
                (i for i, p in enumerate(pairs)
                 if p is not None and bits(p[0]) != bits(p[1])),
                None,
            ),
            "pairs": pairs,
        })

    result = {
        "env": {
            "model": MODEL,
            "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
            "bi_requested": os.environ.get("BI", "1"),
            "vllm_env_batch_invariant": bool(vllm_envs.VLLM_BATCH_INVARIANT),
            "batch_invariant_mode_enabled_in_driver": bi_mode,
            "note": "init_batch_invariance() runs in the EngineCore worker; the driver flag stays False by design. See run log for aten::mm override at batch_invariant.py.",
            "tensor_parallel_size": TP,
            "max_model_len": MAX_LEN,
            "max_num_batched_tokens": CHUNK,
            "enable_prefix_caching": PC,
            "gpu_memory_utilization": GPU_UTIL,
            "attention_backend": "FLASH_ATTN",
            "enforce_eager": EAGER,
        },
        "target_prompt": TARGET,
        "single": {
            "token_ids": extract(single)[0],
            "logprobs": extract(single)[1],
        },
        "long_prompt": {
            "n_tokens": len(long_ids),
            "token_ids": extract(long_single)[0] if long_single is not None else None,
            "logprobs": extract(long_single)[1] if long_single is not None else None,
        },
        "comparisons": {
            "bs1_vs_bs32": compare(single, batch_target),
            "bs1_vs_reverse_order": compare(single, rev_target),
            "bs1_vs_repeat_prefix_cache": compare(single, repeat),
            "decode_vs_refeed": refeed_cmp,
        },
        "errors": errors,
        "timing_sec": {
            "single": t_single,
            "batch32": t_batch,
            "reverse32": t_rev,
            "repeat": t_repeat,
            "long_prompt": t_long,
            "total": time.time() - t_start,
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "out": OUT,
        "bs1_vs_bs32": {
            k: result["comparisons"]["bs1_vs_bs32"][k]
            for k in ("token_ids_equal", "logprobs_bitwise_equal",
                      "max_abs_logprob_diff", "first_logprob_mismatch")
        },
        "bs1_vs_reverse_order": {
            k: result["comparisons"]["bs1_vs_reverse_order"][k]
            for k in ("token_ids_equal", "logprobs_bitwise_equal",
                      "max_abs_logprob_diff", "first_logprob_mismatch")
        },
        "bs1_vs_repeat_prefix_cache": {
            k: result["comparisons"]["bs1_vs_repeat_prefix_cache"][k]
            for k in ("token_ids_equal", "logprobs_bitwise_equal",
                      "max_abs_logprob_diff", "first_logprob_mismatch")
        },
        "decode_vs_refeed": {
            k: refeed_cmp.get(k) for k in
            ("available", "n_generated", "n_compared",
             "all_bitwise_equal", "max_abs_diff", "first_mismatch")
        },
        "timing_sec": result["timing_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
