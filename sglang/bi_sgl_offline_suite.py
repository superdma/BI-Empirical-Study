"""Offline batch-invariance suite for SGLang.

One engine launch runs five tests, mirroring the vLLM scripts:
  1. 32 prompts: solo vs batch vs batch-repeat
  2. decode batch-size sweep 128..257 (target prompt only)
  3. prefill length boundary: N=32 prompts, L=9 tokens
  4. decode logprob vs re-feed prompt logprob
  5. long prompt (~7584 tokens): fresh vs radix-cache hit

Environment:
  MODEL, TP, DET (0/1), CUDAGRAPH (0/1), RADIX (0/1), OUTDIR, TAG
"""
import json
import os
import struct
import time

import sglang as sgl

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TP = int(os.environ.get("TP", "4"))
DET = os.environ.get("DET", "1") == "1"
CUDAGRAPH = os.environ.get("CUDAGRAPH", "1") == "1"
RADIX = os.environ.get("RADIX", "1") == "1"
OUTDIR = os.environ.get("OUTDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
TAG = os.environ.get("TAG", "sgl")

TARGET = "The capital of France is"
FILLER = "Question {i}: describe the number {i} in one short sentence."
BODY = "The quick brown fox jumps over the lazy dog and keeps running. "


def bits(x):
    return struct.pack(">d", float(x)).hex()


def meta(res):
    """Return (token_ids, logprobs) for one generation result."""
    m = res["meta_info"]
    pairs = m.get("output_token_logprobs") or []
    ids = [int(p[1]) for p in pairs if p and p[1] is not None]
    lps = [float(p[0]) for p in pairs if p and p[0] is not None]
    return ids, lps


def cmp(a, b):
    ia, la = a
    ib, lb = b
    n = min(len(la), len(lb))
    first = next((i for i in range(n) if bits(la[i]) != bits(lb[i])), None)
    return {
        "ids": ia == ib,
        "bitwise": first is None and len(la) == len(lb),
        "first": first,
        "max": max((abs(x - y) for x, y in zip(la, lb)), default=None),
        "n": n,
    }


def summarize(cmp_map):
    bad = [k for k, v in cmp_map.items() if not (v["ids"] and v["bitwise"])]
    return {"n": len(cmp_map), "bad": len(bad), "bad_names": bad[:10]}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    kwargs = dict(
        model_path=MODEL,
        tp_size=TP,
        mem_fraction_static=0.80,
        context_length=8192,
        chunked_prefill_size=8192,
        attention_backend=os.environ.get("ATTN", "flashinfer"),
        enable_deterministic_inference=DET,
        disable_cuda_graph=not CUDAGRAPH,
        disable_radix_cache=not RADIX,
        log_level="warning",
    )
    if os.environ.get("DCAR", "0") == "1":
        kwargs["disable_custom_all_reduce"] = True
    if os.environ.get("MOE"):
        kwargs["moe_runner_backend"] = os.environ["MOE"]
    if os.environ.get("PP", "1") != "1":
        kwargs["pp_size"] = int(os.environ["PP"])
    t_load = time.time()
    engine = sgl.Engine(**kwargs)
    load_sec = time.time() - t_load
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = {"max_new_tokens": 16, "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}

    def gen(inputs):
        return engine.generate(input_ids=inputs, sampling_params=sp, return_logprob=True)

    env = {
        "engine": "sglang",
        "version": sgl.__version__,
        "det": DET,
        "cuda_graph": CUDAGRAPH,
        "radix": RADIX,
        "tp": TP,
        "model": MODEL,
        "load_sec": load_sec,
    }
    print(json.dumps(env, ensure_ascii=False), flush=True)

    # ---- test 1: 32 prompts, solo vs batch ----
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    fillers = [tok.encode(FILLER.format(i=i), add_special_tokens=False) for i in range(1, 32)]
    prompts = [target_ids] + fillers

    solo = [meta(r) for r in [engine.generate(input_ids=p, sampling_params=sp,
                                              return_logprob=True) for p in prompts]]
    b1 = [meta(r) for r in gen(prompts)]
    b2 = [meta(r) for r in gen(prompts)]
    res1 = {
        "env": dict(env, test="batchall", n=len(prompts)),
        "cmp": {f"prompt{i}": cmp(solo[i], b1[i]) for i in range(len(prompts))},
        "repeat": {f"prompt{i}": cmp(b1[i], b2[i]) for i in range(len(prompts))},
    }
    res1["summary"] = {"solo_vs_batch": summarize(res1["cmp"]),
                       "batch_repeat": summarize(res1["repeat"])}
    json.dump(res1, open(os.path.join(OUTDIR, f"{TAG}_batchall.json"), "w"), indent=2)
    print(json.dumps({"test": "batchall", **res1["summary"]}), flush=True)

    # ---- test 2: decode batch-size sweep ----
    sweep = [128, 192, 224, 240, 255, 256, 257]
    base0 = solo[0]
    rows = []
    for bs in sweep:
        batch = [target_ids] * bs
        t = time.time()
        out = gen(batch)
        first = meta(out[0])
        c = cmp(base0, first)
        rows.append({"bs": bs, **c, "sec": round(time.time() - t, 3)})
        print(json.dumps({"test": "bssweep", "bs": bs, "ids": c["ids"],
                          "bitwise": c["bitwise"], "first": c["first"],
                          "max": c["max"]}), flush=True)
        del out
    json.dump({"env": dict(env, test="bssweep"), "rows": rows},
              open(os.path.join(OUTDIR, f"{TAG}_bssweep.json"), "w"), indent=2)

    # ---- test 3: prefill length boundary (N=32, L=9) ----
    body = tok.encode(BODY, add_special_tokens=False)
    L = 9
    len_prompts = [target_ids]
    for i in range(1, 32):
        pre = tok.encode(f"Question {i}:", add_special_tokens=False)
        ids = (pre + body * 20)[:L]
        len_prompts.append(ids)
    total_tokens = sum(len(p) for p in len_prompts)
    lsolo = [meta(engine.generate(input_ids=p, sampling_params=sp, return_logprob=True))
             for p in len_prompts]
    lb1 = [meta(r) for r in gen(len_prompts)]
    lb2 = [meta(r) for r in gen(len_prompts)]
    res3 = {
        "env": dict(env, test="len", n=len(len_prompts), L=L, total_prompt_tokens=total_tokens),
        "cmp": {f"p{i}": cmp(lsolo[i], lb1[i]) for i in range(len(len_prompts))},
        "repeat": {f"p{i}": cmp(lb1[i], lb2[i]) for i in range(len(len_prompts))},
    }
    res3["summary"] = {"solo_vs_batch": summarize(res3["cmp"]),
                       "batch_repeat": summarize(res3["repeat"])}
    json.dump(res3, open(os.path.join(OUTDIR, f"{TAG}_len.json"), "w"), indent=2)
    print(json.dumps({"test": "len", "total_tokens": total_tokens, **res3["summary"]}), flush=True)

    # ---- test 4: decode vs re-feed ----
    sp64 = {"max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}
    out = engine.generate(input_ids=target_ids, sampling_params=sp64, return_logprob=True)
    gids, glp = meta(out)
    full = target_ids + gids
    ref = engine.generate(input_ids=full,
                          sampling_params={"max_new_tokens": 1, "temperature": 0.0,
                                           "ignore_eos": True},
                          return_logprob=True, logprob_start_len=0)
    inp = ref["meta_info"].get("input_token_logprobs") or []
    pairs = []
    for i, gid in enumerate(gids):
        pos = len(target_ids) + i
        item = inp[pos] if pos < len(inp) else None
        if not item or item[0] is None:
            pairs.append(None)
            continue
        lp = float(item[0])
        pairs.append({"pos": pos, "token_id": gid, "decode": glp[i], "refeed": lp,
                      "bitwise_equal": bits(glp[i]) == bits(lp)})
    valid = [p for p in pairs if p]
    mism = [p for p in valid if not p["bitwise_equal"]]
    res4 = {
        "env": dict(env, test="refeed", n_generated=len(gids), prompt_ids=target_ids),
        "token_ids": gids,
        "decode_logprobs": glp,
        "pairs": pairs,
        "n_compared": len(valid),
        "n_mismatch": len(mism),
        "first_mismatch": next((i for i, p in enumerate(pairs)
                                if p and not p["bitwise_equal"]), None),
        "max_abs_diff": max((abs(p["decode"] - p["refeed"]) for p in mism), default=None),
    }
    json.dump(res4, open(os.path.join(OUTDIR, f"{TAG}_refeed.json"), "w"), indent=2)
    print(json.dumps({"test": "refeed", "n_compared": res4["n_compared"],
                      "n_mismatch": res4["n_mismatch"],
                      "first_mismatch": res4["first_mismatch"],
                      "max_abs_diff": res4["max_abs_diff"]}), flush=True)

    # ---- test 5: long prompt, fresh vs cache hit ----
    long_ids = tok.encode("Passage. " + BODY * 700, add_special_tokens=False)[:7584]
    sp8 = {"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}
    fresh = meta(engine.generate(input_ids=long_ids, sampling_params=sp8, return_logprob=True))
    hit = meta(engine.generate(input_ids=long_ids, sampling_params=sp8, return_logprob=True))
    again = meta(engine.generate(input_ids=long_ids, sampling_params=sp8, return_logprob=True))
    res5 = {
        "env": dict(env, test="long", n_prompt_tokens=len(long_ids)),
        "fresh_vs_hit": cmp(fresh, hit),
        "hit_vs_hit2": cmp(hit, again),
    }
    json.dump(res5, open(os.path.join(OUTDIR, f"{TAG}_long.json"), "w"), indent=2)
    print(json.dumps({"test": "long", "tokens": len(long_ids),
                      "fresh_vs_hit": res5["fresh_vs_hit"],
                      "hit_vs_hit2": res5["hit_vs_hit2"]}), flush=True)

    engine.shutdown()
    print(json.dumps({"tag": TAG, "total_sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
