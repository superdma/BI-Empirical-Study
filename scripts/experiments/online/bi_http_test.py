import os, sys, json, time, struct, urllib.request, urllib.error

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_results.json"))
TAG = os.environ.get("TAG", "server")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 32)]
LONG_PROMPT = ("The following is a long passage. " +
               "The quick brown fox jumps over the lazy dog. " * 900)


def post(path, payload, timeout=900):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def tokenize(text):
    return list(post("/tokenize", {"model": MODEL, "prompt": text})["tokens"])


def complete(prompts, max_tokens=MAX_TOKENS, prompt_logprobs=None, salt=None):
    payload = {
        "model": MODEL, "prompt": prompts, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "logprobs": 5, "seed": 0,
        "return_token_ids": True,
    }
    if prompt_logprobs is not None:
        payload["prompt_logprobs"] = prompt_logprobs
    if salt is not None:
        payload["cache_salt"] = salt
    t0 = time.time()
    r = post("/v1/completions", payload)
    return r, time.time() - t0


def pick(resp, idx):
    for c in resp["choices"]:
        if c.get("index") == idx:
            return c
    raise RuntimeError(f"choice {idx} missing")


def bits(x):
    return struct.pack(">d", float(x)).hex()


def cmp_choice(a, b):
    ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
    la = (a.get("logprobs") or {}).get("token_logprobs") or []
    lb = (b.get("logprobs") or {}).get("token_logprobs") or []
    toa = (a.get("logprobs") or {}).get("top_logprobs") or []
    tob = (b.get("logprobs") or {}).get("top_logprobs") or []
    first = None
    for i in range(min(len(la), len(lb))):
        if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i]):
            first = i
            break
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {
        "token_ids_equal": ta == tb,
        "token_logprobs_bitwise_equal": len(la) == len(lb) and first is None,
        "first_logprob_mismatch": first,
        "max_abs_logprob_diff": max(diffs) if diffs else None,
        "top_logprobs_equal": toa == tob,
        "len_a": len(ta), "len_b": len(tb),
        "token_ids_a": ta, "token_ids_b": tb,
        "token_logprobs_a": la, "token_logprobs_b": lb,
    }


def main():
    t_start = time.time()
    errors = {}
    target_ids = tokenize(TARGET)
    filler_ids = [tokenize(p) for p in FILLERS]
    long_ids = tokenize(LONG_PROMPT)[:7600]

    # short prompt, fresh (unique salt per case)
    single_resp, t_single = complete([target_ids], salt="case_single")
    single = pick(single_resp, 0)

    batch_resp, t_batch = complete([target_ids] + filler_ids, salt="case_batch32")
    batch_target = pick(batch_resp, 0)

    rev_prompts = list(reversed([target_ids] + filler_ids))
    rev_resp, t_rev = complete(rev_prompts, salt="case_reverse")
    rev_target = pick(rev_resp, len(rev_prompts) - 1)

    # same salt as the single case -> prefix-cache hit
    repeat_resp, t_repeat = complete([target_ids], salt="case_single")
    repeat = pick(repeat_resp, 0)

    # long prompt: one fresh bs=1 and one fresh bs=4 (different salts -> no cache reuse)
    long_resp, t_long = complete([long_ids], salt="case_long_single")
    long_single = pick(long_resp, 0)

    long_batch_resp, t_long_batch = complete([long_ids] + filler_ids[:3],
                                             salt="case_long_batch_fresh")
    long_batch_fresh = pick(long_batch_resp, 0)

    # same salt as long_single -> prefix-cache hit, batch of 4
    long_hit_resp, t_long_hit = complete([long_ids] + filler_ids[:3],
                                         salt="case_long_single")
    long_batch_hit = pick(long_hit_resp, 0)

    # decode vs re-feed
    gen_ids = single.get("token_ids") or []
    full_ids = target_ids + gen_ids
    refeed_resp, t_refeed = complete([full_ids], max_tokens=1,
                                     prompt_logprobs=1, salt="case_refeed")
    refeed = pick(refeed_resp, 0)

    refeed_cmp = {"available": False}
    pls = refeed.get("prompt_logprobs")
    if pls is not None:
        dec = (single.get("logprobs") or {}).get("token_logprobs") or []
        pairs = []
        for i, tid in enumerate(gen_ids):
            pos = len(target_ids) + i
            d_ref = pls[pos] if pos < len(pls) else None
            if d_ref is None:
                pairs.append(None); continue
            ent = d_ref.get(str(tid)) or d_ref.get(tid)
            pairs.append(None if ent is None else (float(dec[i]), float(ent["logprob"])))
        valid = [p for p in pairs if p is not None]
        refeed_cmp = {
            "available": True, "n_generated": len(gen_ids), "n_compared": len(valid),
            "all_bitwise_equal": bool(valid) and all(bits(x) == bits(y) for x, y in valid),
            "max_abs_diff": max((abs(x - y) for x, y in valid), default=None),
            "first_mismatch": next((i for i, p in enumerate(pairs)
                                    if p is not None and bits(p[0]) != bits(p[1])), None),
            "pairs": pairs,
        }

    result = {
        "env": {"base": BASE, "tag": TAG, "model": MODEL, "max_tokens": MAX_TOKENS,
                "target_prompt": TARGET, "n_fillers": len(FILLERS),
                "long_prompt_tokens": len(long_ids)},
        "prompt_token_ids": {"target": target_ids, "long_first_8": long_ids[:8]},
        "single": {"text": single.get("text"), "token_ids": single.get("token_ids"),
                   "logprobs": (single.get("logprobs") or {}).get("token_logprobs"),
                   "top_logprobs": (single.get("logprobs") or {}).get("top_logprobs")},
        "long_single": {"text": long_single.get("text"),
                        "token_ids": long_single.get("token_ids"),
                        "logprobs": (long_single.get("logprobs") or {}).get("token_logprobs"),
                        "top_logprobs": (long_single.get("logprobs") or {}).get("top_logprobs")},
        "comparisons": {
            "server_bs1_vs_bs32": cmp_choice(single, batch_target),
            "server_bs1_vs_reverse_order": cmp_choice(single, rev_target),
            "server_short_fresh_vs_cache_hit": cmp_choice(single, repeat),
            "server_long_bs1_vs_bs4_fresh": cmp_choice(long_single, long_batch_fresh),
            "server_long_fresh_vs_cache_hit": cmp_choice(long_single, long_batch_hit),
            "server_decode_vs_refeed": refeed_cmp,
        },
        "errors": errors,
        "timing_sec": {"single": t_single, "batch32": t_batch, "reverse32": t_rev,
                       "repeat": t_repeat, "long_single": t_long,
                       "long_batch_fresh": t_long_batch, "long_batch_cache_hit": t_long_hit,
                       "refeed": t_refeed, "total": time.time() - t_start},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary = {}
    for k, c in result["comparisons"].items():
        if k == "server_decode_vs_refeed":
            summary[k] = {kk: c.get(kk) for kk in ("available", "n_generated", "n_compared",
                                                   "all_bitwise_equal", "max_abs_diff",
                                                   "first_mismatch")}
        else:
            summary[k] = {kk: c.get(kk) for kk in ("token_ids_equal",
                                                   "token_logprobs_bitwise_equal",
                                                   "top_logprobs_equal",
                                                   "first_logprob_mismatch",
                                                   "max_abs_logprob_diff")}
    print(json.dumps({"out": OUT, "env": result["env"], "summary": summary,
                      "errors": errors, "timing": result["timing_sec"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
