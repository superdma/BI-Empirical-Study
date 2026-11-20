import os, json, time, struct, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8030")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_batchall.json"))
TAG = os.environ.get("TAG", "batchall")
N = int(os.environ.get("N", "32"))
MT = int(os.environ.get("MAX_TOKENS", "16"))

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, N)]


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tok(s):
    return list(post("/tokenize", {"model": MODEL, "prompt": s})["tokens"])


def gen(prompts, salt):
    p = {"model": MODEL, "prompt": prompts, "max_tokens": MT, "temperature": 0.0,
         "top_p": 1.0, "seed": 0, "logprobs": 5, "return_token_ids": True,
         "cache_salt": salt}
    t = time.time()
    r = post("/v1/completions", p)
    return r, time.time() - t


def bits(x):
    return struct.pack(">d", float(x)).hex()


def choice(r, idx):
    for c in r["choices"]:
        if c.get("index") == idx:
            return c
    raise RuntimeError(idx)


def cmp(a, b):
    ta, tb = a.get("token_ids") or [], b.get("token_ids") or []
    la = a["logprobs"]["token_logprobs"]
    lb = b["logprobs"]["token_logprobs"]
    first = next((i for i in range(min(len(la), len(lb)))
                  if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i])), None)
    return {"ids": ta == tb, "bitwise": first is None, "first": first,
            "max": max((abs(x - y) for x, y in zip(la, lb)
                        if x is not None and y is not None), default=None)}


def main():
    t0 = time.time()
    prompts = [tok(TARGET)] + [tok(p) for p in FILLERS]
    res = {"env": {"base": BASE, "tag": TAG, "n": N, "max_tokens": MT},
           "baseline": {}, "batch": [], "cmp": {}, "repeat": {}, "errors": []}

    for i, ids in enumerate(prompts):
        try:
            r, sec = gen([ids], f"solo{i}")
            res["baseline"][str(i)] = choice(r, 0)
        except Exception as e:
            res["errors"].append(f"baseline {i}: {e!r}")

    r, sec = gen(prompts, "batchall")
    res["env"]["batch_sec"] = sec
    for i in range(N):
        c = choice(r, i)
        res["batch"].append(c)
        if str(i) in res["baseline"]:
            res["cmp"][f"prompt{i}"] = cmp(res["baseline"][str(i)], c)

    # repeat the batch request to check its own determinism
    r2, sec2 = gen(prompts, "batchall2")
    res["env"]["batch2_sec"] = sec2
    for i in range(N):
        res["repeat"][f"prompt{i}"] = cmp(choice(r, i), choice(r2, i))

    res["env"]["total_sec"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=2)
    bad = [k for k, v in res["cmp"].items() if not (v["ids"] and v["bitwise"])]
    bad_rep = [k for k, v in res["repeat"].items() if not (v["ids"] and v["bitwise"])]
    print(json.dumps({"tag": TAG, "n": N,
                      "solo_vs_batch_bad": len(bad), "bad": bad[:8],
                      "batch_repeat_bad": len(bad_rep),
                      "batch_sec": sec, "batch2_sec": sec2}, indent=2))


if __name__ == "__main__":
    main()
