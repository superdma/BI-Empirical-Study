import os, json, struct, time

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
EAGER = os.environ.get("EAGER", "1") == "1"
PC = os.environ.get("PC", "0") == "1"
MT = int(os.environ.get("MT", "16"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_batchall_offline.json"))

TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 32)]


def bits(x):
    return struct.pack(">d", float(x)).hex()


def summarize(r):
    o = r.outputs[0]
    ids = list(o.token_ids)
    lp = [None if o.logprobs is None or i >= len(o.logprobs) or o.logprobs[i] is None
          or ids[i] not in o.logprobs[i]
          else float(o.logprobs[i][ids[i]].logprob) for i in range(len(ids))]
    return {"token_ids": ids, "logprobs": lp, "text": o.text}


def cmp(a, b):
    la, lb = a["logprobs"], b["logprobs"]
    first = next((i for i in range(min(len(la), len(lb)))
                  if la[i] is None or lb[i] is None or bits(la[i]) != bits(lb[i])), None)
    diffs = [abs(x - y) for x, y in zip(la, lb) if x is not None and y is not None]
    return {"ids": a["token_ids"] == b["token_ids"], "bitwise": first is None,
            "first": first, "max": max(diffs) if diffs else None}


def main():
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, gpu_memory_utilization=0.88,
              max_model_len=8192, max_num_batched_tokens=8192,
              enable_prefix_caching=PC, enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              trust_remote_code=False, disable_log_stats=True, enforce_eager=EAGER)
    tok = llm.get_tokenizer()
    prompts = [tok.encode(TARGET, add_special_tokens=False)] + \
              [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    sp = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=MT, logprobs=5)

    solo = []
    for ids in prompts:
        solo.append(summarize(llm.generate([ids], sampling_params=sp)[0]))
    batch = [summarize(r) for r in llm.generate(prompts, sampling_params=sp)]
    batch2 = [summarize(r) for r in llm.generate(prompts, sampling_params=sp)]

    res = {"env": {"eager": EAGER, "pc": PC, "max_tokens": MT, "n": len(prompts)},
           "solo": solo, "batch": batch, "batch2": batch2,
           "cmp": {f"prompt{i}": cmp(solo[i], batch[i]) for i in range(len(prompts))},
           "repeat": {f"prompt{i}": cmp(batch[i], batch2[i]) for i in range(len(prompts))},
           "total_sec": time.time() - t0}
    json.dump(res, open(OUT, "w"), indent=2)
    bad = [k for k, v in res["cmp"].items() if not (v["ids"] and v["bitwise"])]
    brep = [k for k, v in res["repeat"].items() if not (v["ids"] and v["bitwise"])]
    print(json.dumps({"out": OUT, "n": len(prompts), "bad": len(bad), "bad_names": bad[:8],
                      "batch_repeat_bad": len(brep), "total_sec": res["total_sec"]}, indent=2))


if __name__ == "__main__":
    main()
