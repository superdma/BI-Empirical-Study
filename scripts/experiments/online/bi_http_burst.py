import os, json, time, struct, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BASE", "http://127.0.0.1:8030")
MODEL = os.environ.get("MODEL_NAME", os.path.expanduser("~/models/Qwen3-30B-A3B"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_http_burst.json"))
TAG = os.environ.get("TAG", "burst")
N = int(os.environ.get("N", "32"))
MT = int(os.environ.get("MAX_TOKENS", "16"))
DELAY = float(os.environ.get("DELAY", "1.0"))
B_TOKENS = int(os.environ.get("B_TOKENS", "400"))

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, N)]
B = ("Write a long, detailed essay about the history of computing and its "
     "impact on society, covering hardware, software, and networks. " * 4)


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tok(s):
    return list(post("/tokenize", {"model": MODEL, "prompt": s})["tokens"])


def gen(ids, salt, mt, ignore_eos=False):
    p = {"model": MODEL, "prompt": [ids], "max_tokens": mt, "temperature": 0.0,
         "top_p": 1.0, "seed": 0, "logprobs": 5, "return_token_ids": True,
         "cache_salt": salt}
    if ignore_eos:
        p["ignore_eos"] = True
    t = time.time()
    r = post("/v1/completions", p)
    c = r["choices"][0]
    return {"token_ids": c.get("token_ids"),
            "logprobs": c["logprobs"]["token_logprobs"], "sec": time.time() - t}


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
    t0 = time.time()
    prompts = [tok(TARGET)] + [tok(p) for p in FILLERS]
    res = {"env": {"base": BASE, "tag": TAG, "n": N, "max_tokens": MT,
                   "delay": DELAY, "b_max_tokens": B_TOKENS},
           "baseline": {}, "burst_during_decode": {}, "burst_alone": {},
           "cmp_during": {}, "cmp_alone": {}, "errors": []}

    for i, ids in enumerate(prompts):
        try:
            res["baseline"][str(i)] = gen(ids, f"base{i}", MT)
        except Exception as e:
            res["errors"].append(f"baseline {i}: {e!r}")

    bid = tok(B)
    holder = {}

    def run_b():
        try:
            holder["b"] = gen(bid, "b_bg", B_TOKENS, ignore_eos=True)
        except Exception as e:
            holder["error"] = repr(e)

    th = threading.Thread(target=run_b)
    th.start()
    time.sleep(DELAY)

    def work(i):
        try:
            return i, gen(prompts[i], f"burst{i}", MT)
        except Exception as e:
            return i, {"error": repr(e)}

    tw = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        for i, r in ex.map(work, range(N)):
            res["burst_during_decode"][str(i)] = r
    res["env"]["burst_wall_sec"] = time.time() - tw
    th.join()

    for i in range(N):
        a, b = res["baseline"].get(str(i)), res["burst_during_decode"].get(str(i))
        if a and b and "error" not in b:
            res["cmp_during"][f"prompt{i}"] = cmp(a, b)

    tw2 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        def work2(i):
            try:
                return i, gen(prompts[i], f"alone{i}", MT)
            except Exception as e:
                return i, {"error": repr(e)}
        for i, r in ex.map(work2, range(N)):
            res["burst_alone"][str(i)] = r
    res["env"]["alone_wall_sec"] = time.time() - tw2

    for i in range(N):
        a, b = res["baseline"].get(str(i)), res["burst_alone"].get(str(i))
        if a and b and "error" not in b:
            res["cmp_alone"][f"prompt{i}"] = cmp(a, b)

    res["env"]["total_sec"] = time.time() - t0
    res["env"]["b_sec"] = holder.get("b", {}).get("sec")
    res["env"]["b_error"] = holder.get("error")
    json.dump(res, open(OUT, "w"), indent=2)

    for key in ("cmp_during", "cmp_alone"):
        bad = [k for k, v in res[key].items() if not (v["ids"] and v["bitwise"])]
        print(key, "n", len(res[key]), "bad", len(bad), bad[:6])
    print("b_sec", res["env"]["b_sec"], "burst_sec", res["env"]["burst_wall_sec"])


if __name__ == "__main__":
    main()
