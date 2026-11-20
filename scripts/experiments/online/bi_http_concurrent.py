import os, json, time, struct, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BASE", "http://127.0.0.1:8030")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_concurrent.json"))
TAG = os.environ.get("TAG", "concurrent")
N = int(os.environ.get("N", "32"))
MT = int(os.environ.get("MAX_TOKENS", "16"))
FILLER_TOKENS = int(os.environ.get("FILLER_TOKENS", "0"))

def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def tok(s):
    return list(post("/tokenize", {"model": MODEL, "prompt": s})["tokens"])

def gen(ids, salt):
    p = {"model": MODEL, "prompt": [ids], "max_tokens": MT, "temperature": 0.0,
         "top_p": 1.0, "seed": 0, "logprobs": 5, "return_token_ids": True,
         "cache_salt": salt}
    t = time.time()
    r = post("/v1/completions", p)
    c = r["choices"][0]
    return {"token_ids": c.get("token_ids"),
            "logprobs": c["logprobs"]["token_logprobs"],
            "sec": time.time() - t}

def bits(x): return struct.pack(">d", float(x)).hex()

def cmp(a, b):
    ta, tb = a["token_ids"], b["token_ids"]
    la, lb = a["logprobs"], b["logprobs"]
    return {"ids": ta == tb,
            "bitwise": all(bits(x) == bits(y) for x, y in zip(la, lb)),
            "first": next((i for i, (x, y) in enumerate(zip(la, lb)) if bits(x) != bits(y)), None),
            "max": max((abs(x - y) for x, y in zip(la, lb)), default=None)}

def main():
    t0 = time.time()
    target = tok("The capital of France is")
    if FILLER_TOKENS:
        filler_text = ("The quick brown fox jumps over the lazy dog. " * ((FILLER_TOKENS // 10) + 1))
        fillers = [tok(f"Passage {i}. " + filler_text)[:FILLER_TOKENS] for i in range(1, N)]
    else:
        fillers = [tok(f"Question {i}: describe the number {i} in one short sentence.")
                   for i in range(1, N)]
    prompts = [target] + fillers

    res = {"env": {"base": BASE, "tag": TAG, "n": N, "max_tokens": MT,
                   "filler_tokens": FILLER_TOKENS, "target_tokens": len(target)},
           "baseline": {}, "concurrent": {}, "cmp": {}, "errors": []}

    # sequential baselines, each alone in the engine
    for i, p in enumerate(prompts):
        try:
            res["baseline"][str(i)] = gen(p, f"base{i}")
        except Exception as e:
            res["errors"].append(f"baseline {i}: {e!r}")

    # all prompts at once
    t_conc = time.time()
    def work(i):
        try:
            return i, gen(prompts[i], f"conc{i}")
        except Exception as e:
            return i, {"error": repr(e)}
    with ThreadPoolExecutor(max_workers=N) as ex:
        for i, r in ex.map(work, range(N)):
            res["concurrent"][str(i)] = r
    res["env"]["concurrent_wall_sec"] = time.time() - t_conc

    for i in range(N):
        a = res["baseline"].get(str(i)); b = res["concurrent"].get(str(i))
        if a and b and "error" not in b:
            res["cmp"][f"prompt{i}_seq_vs_concurrent"] = cmp(a, b)

    # N identical target requests at once
    t_same = time.time()
    def same(i):
        try:
            return i, gen(target, f"same{i}")
        except Exception as e:
            return i, {"error": repr(e)}
    with ThreadPoolExecutor(max_workers=N) as ex:
        same_res = dict(ex.map(same, range(N)))
    res["env"]["identical_wall_sec"] = time.time() - t_same
    base0 = res["baseline"].get("0")
    for i in range(N):
        b = same_res.get(i)
        if base0 and b and "error" not in b:
            res["cmp"][f"identical{i}_vs_baseline"] = cmp(base0, b)

    res["env"]["total_sec"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=2)

    bad = [k for k, v in res["cmp"].items() if not (v["ids"] and v["bitwise"])]
    print(json.dumps({"tag": TAG, "n": N, "filler_tokens": FILLER_TOKENS,
                      "n_cmp": len(res["cmp"]), "n_bad": len(bad),
                      "bad": bad[:10], "errors": res["errors"][:5],
                      "timing": {k: res["env"][k] for k in
                                 ("concurrent_wall_sec", "identical_wall_sec", "total_sec")}},
                     indent=2))

if __name__ == "__main__":
    main()
