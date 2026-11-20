import os, json, time, struct, threading, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8030")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_mixed.json"))
TAG = os.environ.get("TAG", "mixed")
DELAYS = [float(x) for x in os.environ.get("DELAYS", "0.2,0.6,1.5,3.0").split(",")]
A_TOKENS = int(os.environ.get("A_TOKENS", "16"))
B_TOKENS = int(os.environ.get("B_TOKENS", "300"))

A = "Question 1: describe the number 1 in one short sentence."
B = ("Write a long, detailed essay about the history of computing. " * 8)


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    return obj, time.time() - t


def tok(s):
    return list(post("/tokenize", {"model": MODEL, "prompt": s})[0]["tokens"])


def gen(ids, salt, mt, ignore_eos=False):
    p = {"model": MODEL, "prompt": [ids], "max_tokens": mt, "temperature": 0.0,
         "top_p": 1.0, "seed": 0, "logprobs": 5, "return_token_ids": True,
         "cache_salt": salt}
    if ignore_eos:
        p["ignore_eos"] = True
    r, sec = post("/v1/completions", p)
    c = r["choices"][0]
    return {"token_ids": c.get("token_ids"),
            "logprobs": c["logprobs"]["token_logprobs"],
            "text": c.get("text"), "sec": sec}


def bits(x):
    return struct.pack(">d", float(x)).hex()


def cmp(a, b):
    la, lb = a["logprobs"], b["logprobs"]
    first = next((i for i in range(min(len(la), len(lb)))
                  if bits(la[i]) != bits(lb[i])), None)
    return {"ids": a["token_ids"] == b["token_ids"], "bitwise": first is None,
            "first": first,
            "max": max((abs(x - y) for x, y in zip(la, lb)), default=None)}


def main():
    aid, bid = tok(A), tok(B)
    res = {"env": {"base": BASE, "tag": TAG, "a_tokens": len(aid),
                   "b_tokens": len(bid), "b_max_tokens": B_TOKENS,
                   "delays": DELAYS},
           "baselines": [], "cases": [], "errors": []}

    base = gen(aid, "a_base", A_TOKENS)
    res["baselines"].append(base)
    # confirm the baseline repeats bitwise
    base2 = gen(aid, "a_base2", A_TOKENS)
    res["baselines"].append(base2)
    res["baseline_stable"] = cmp(base, base2)

    for delay in DELAYS:
        holder = {}

        def run_b():
            try:
                holder["b"] = gen(bid, f"b_{delay}", B_TOKENS, ignore_eos=True)
            except Exception as e:
                holder["error"] = repr(e)

        th = threading.Thread(target=run_b)
        t0 = time.time()
        th.start()
        time.sleep(delay)
        a = gen(aid, f"a_{delay}", A_TOKENS)
        t_a = time.time() - t0
        th.join()
        res["cases"].append({"delay": delay, "a_sec": t_a,
                             "b_sec": holder.get("b", {}).get("sec"),
                             "b_error": holder.get("error"),
                             "a": a, "cmp": cmp(base, a)})

    # after B is done, A must reproduce the baseline
    a_after = gen(aid, "a_after", A_TOKENS)
    res["after"] = {"a": a_after, "cmp": cmp(base, a_after)}
    res["total_sec"] = None

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    brief = {"tag": TAG, "baseline_stable": res["baseline_stable"],
             "cases": [{"delay": c["delay"], "a_sec": round(c["a_sec"], 3),
                        "b_sec": c["b_sec"], "cmp": c["cmp"]} for c in res["cases"]],
             "after": res["after"]["cmp"]}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
