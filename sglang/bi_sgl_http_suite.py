"""Online batch-invariance suite against a running SGLang server.

Uses the native /generate endpoint, which returns token-level logprobs as
numbers (the OpenAI-compatible endpoint returns the same values through
/v1/completions but drops some fields we compare).

Tests:
  A. sequential baselines, 32 prompts
  B. 32 concurrent requests vs baseline
  C. 32 identical requests at once vs baseline
  D. burst: 32 concurrent while a long request is decoding

Environment: BASE, N, MT, TAG, OUTDIR, DELAY, B_TOKENS
"""
import json
import os
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BASE", "http://127.0.0.1:30000")
N = int(os.environ.get("N", "32"))
MT = int(os.environ.get("MT", "16"))
TAG = os.environ.get("TAG", "sgl_http")
OUTDIR = os.environ.get("OUTDIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results")
DELAY = float(os.environ.get("DELAY", "1.0"))
B_TOKENS = int(os.environ.get("B_TOKENS", "400"))

TARGET = "The capital of France is"
FILLER = "Question {i}: describe the number {i} in one short sentence."
LONG = ("Write a long, detailed essay about the history of computing and its "
        "impact on society, covering hardware, software, and networks. ") * 4


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def gen(text, mt):
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": mt, "temperature": 0.0,
                            "top_p": 1.0, "ignore_eos": True},
        "return_logprob": True,
    }
    t = time.time()
    r = post("/generate", payload)
    m = r.get("meta_info", {})
    pairs = m.get("output_token_logprobs") or []
    return {
        "token_ids": [int(p[1]) for p in pairs if p and p[1] is not None],
        "logprobs": [float(p[0]) for p in pairs if p and p[0] is not None],
        "sec": time.time() - t,
    }


def bits(x):
    return struct.pack(">d", float(x)).hex()


def cmp(a, b):
    la, lb = a["logprobs"], b["logprobs"]
    n = min(len(la), len(lb))
    first = next((i for i in range(n) if bits(la[i]) != bits(lb[i])), None)
    return {"ids": a["token_ids"] == b["token_ids"],
            "bitwise": first is None and len(la) == len(lb),
            "first": first,
            "max": max((abs(x - y) for x, y in zip(la, lb)), default=None)}


def summarize(m):
    bad = [k for k, v in m.items() if not (v["ids"] and v["bitwise"])]
    return {"n": len(m), "bad": len(bad), "bad_names": bad[:8]}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    prompts = [TARGET] + [FILLER.format(i=i) for i in range(1, N)]
    info = get("/get_server_info")
    env = {"base": BASE, "n": N, "max_tokens": MT, "delay": DELAY, "b_max_tokens": B_TOKENS,
           "model": info.get("model_path"), "version": info.get("version"),
           "det": info.get("enable_deterministic_inference")}
    res = {"env": env, "baseline": {}, "concurrent": {}, "identical": {},
           "burst": {}, "cmp": {}, "errors": []}

    for i, text in enumerate(prompts):
        try:
            res["baseline"][str(i)] = gen(text, MT)
        except Exception as exc:
            res["errors"].append(f"baseline {i}: {exc!r}")
    print(json.dumps({"step": "baseline", "n": len(res["baseline"]),
                      "errors": res["errors"][:3]}), flush=True)

    def work(i, text, mt):
        try:
            return i, gen(text, mt)
        except Exception as exc:
            return i, {"error": repr(exc)}

    t_conc = time.time()
    with ThreadPoolExecutor(max_workers=N) as pool:
        for i, r in pool.map(lambda i: work(i, prompts[i], MT), range(N)):
            res["concurrent"][str(i)] = r
    res["env"]["concurrent_sec"] = round(time.time() - t_conc, 2)
    for i in range(N):
        a, b = res["baseline"].get(str(i)), res["concurrent"].get(str(i))
        if a and b and "error" not in b:
            res["cmp"][f"p{i}_seq_vs_concurrent"] = cmp(a, b)
    print(json.dumps({"step": "concurrent", **summarize(
        {k: v for k, v in res["cmp"].items() if "concurrent" in k}),
        "wall": res["env"]["concurrent_sec"]}), flush=True)

    t_same = time.time()
    with ThreadPoolExecutor(max_workers=N) as pool:
        for i, r in pool.map(lambda i: work(i, TARGET, MT), range(N)):
            res["identical"][str(i)] = r
    res["env"]["identical_sec"] = round(time.time() - t_same, 2)
    base0 = res["baseline"].get("0")
    for i in range(N):
        b = res["identical"].get(str(i))
        if base0 and b and "error" not in b:
            res["cmp"][f"identical{i}_vs_baseline"] = cmp(base0, b)
    print(json.dumps({"step": "identical", **summarize(
        {k: v for k, v in res["cmp"].items() if "identical" in k})}), flush=True)

    holder = {}

    def long_job():
        try:
            holder["r"] = gen(LONG, B_TOKENS)
        except Exception as exc:
            holder["error"] = repr(exc)

    th = threading.Thread(target=long_job)
    th.start()
    time.sleep(DELAY)
    t_burst = time.time()
    with ThreadPoolExecutor(max_workers=N) as pool:
        for i, r in pool.map(lambda i: work(i, prompts[i], MT), range(N)):
            res["burst"][str(i)] = r
    res["env"]["burst_sec"] = round(time.time() - t_burst, 2)
    th.join()
    for i in range(N):
        a, b = res["baseline"].get(str(i)), res["burst"].get(str(i))
        if a and b and "error" not in b:
            res["cmp"][f"p{i}_seq_vs_burst"] = cmp(a, b)
    res["env"]["long_sec"] = round(holder.get("r", {}).get("sec", 0), 2)
    res["env"]["long_error"] = holder.get("error")
    print(json.dumps({"step": "burst", **summarize(
        {k: v for k, v in res["cmp"].items() if "burst" in k})}), flush=True)

    res["env"]["total_sec"] = round(time.time() - t0, 1)
    res["summary"] = summarize(res["cmp"])
    path = os.path.join(OUTDIR, f"{TAG}_http.json")
    json.dump(res, open(path, "w"), indent=2)
    print(json.dumps({"saved": path, "summary": res["summary"],
                      "errors": res["errors"][:5], "env": res["env"]}), flush=True)


if __name__ == "__main__":
    main()
