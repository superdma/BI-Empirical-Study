import os, json, time, struct, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8053")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_len.json"))
TAG = os.environ.get("TAG", "len")
N = int(os.environ.get("N", "32"))
L = int(os.environ.get("L", "9"))
MT = int(os.environ.get("MAX_TOKENS", "16"))

TARGET = "The capital of France is"
BODY = "The quick brown fox jumps over the lazy dog and keeps running. "


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
    r = post("/v1/completions", p)
    return r


def bits(x): return struct.pack(">d", float(x)).hex()


def choice(r, i):
    for c in r["choices"]:
        if c.get("index") == i:
            return c
    raise RuntimeError(i)


def cmp(a, b):
    la, lb = a["logprobs"]["token_logprobs"], b["logprobs"]["token_logprobs"]
    first = next((i for i in range(min(len(la), len(lb)))
                  if bits(la[i]) != bits(lb[i])), None)
    return {"ids": a["token_ids"] == b["token_ids"], "bitwise": first is None,
            "first": first,
            "max": max((abs(x - y) for x, y in zip(la, lb)), default=None)}


def main():
    t0 = time.time()
    tgt = tok(TARGET)
    body = tok(BODY)
    prompts = [tgt]
    for i in range(1, N):
        pre = tok(f"Question {i}:")
        ids = (pre + body * 20)[:L]
        prompts.append(ids)
    total = sum(len(p) for p in prompts)
    res = {"env": {"base": BASE, "tag": TAG, "n": N, "L": L,
                   "total_prompt_tokens": total, "max_tokens": MT},
           "baseline": {}, "batch": [], "cmp": {}, "repeat": {}, "errors": []}
    for i, ids in enumerate(prompts):
        try:
            r = gen([ids], f"solo{i}")
            res["baseline"][str(i)] = choice(r, 0)
        except Exception as e:
            res["errors"].append(f"baseline {i}: {e!r}")
    r1 = gen(prompts, "batchA")
    r2 = gen(prompts, "batchB")
    for i in range(N):
        a = choice(r1, i); b = choice(r2, i)
        res["batch"].append(a)
        if str(i) in res["baseline"]:
            res["cmp"][f"p{i}"] = cmp(res["baseline"][str(i)], a)
        res["repeat"][f"p{i}"] = cmp(a, b)
    res["env"]["total_sec"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=2)
    bad = [k for k, v in res["cmp"].items() if not (v["ids"] and v["bitwise"])]
    rep = [k for k, v in res["repeat"].items() if not (v["ids"] and v["bitwise"])]
    print(json.dumps({"tag": TAG, "N": N, "L": L, "total_tokens": total,
                      "bad": len(bad), "bad_names": bad[:6],
                      "repeat_bad": len(rep), "total_sec": res["env"]["total_sec"]}, indent=2))


if __name__ == "__main__":
    main()
