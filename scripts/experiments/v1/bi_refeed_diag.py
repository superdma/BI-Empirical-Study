import os, json, struct, time

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
EAGER = os.environ.get("EAGER", "1") == "1"
PC = os.environ.get("PC", "0") == "1"
MT = int(os.environ.get("MT", "64"))
OUT = os.environ.get("OUT", os.path.expanduser("~/bi_refeed_diag.json"))

TARGET = "The capital of France is"


def bits(x):
    return struct.pack(">d", float(x)).hex()


def main():
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, gpu_memory_utilization=0.88,
              max_model_len=8192, max_num_batched_tokens=8192,
              enable_prefix_caching=PC, enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              trust_remote_code=False, disable_log_stats=True, enforce_eager=EAGER)
    tok = llm.get_tokenizer()
    tid = tok.encode(TARGET, add_special_tokens=False)
    sp = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=MT,
                        logprobs=5, ignore_eos=True)
    out = llm.generate([tid], sampling_params=sp)[0]
    o = out.outputs[0]
    gids = list(o.token_ids)
    dec = [float(o.logprobs[i][gids[i]].logprob) for i in range(len(gids))]

    result = {"env": {"eager": EAGER, "pc": PC, "max_tokens": MT,
                      "prompt_ids": tid, "n_generated": len(gids)},
              "token_ids": gids, "decode_logprobs": dec, "refeeds": {}}

    for k in (1, 5, 20):
        full = tid + gids
        r = llm.generate([full], sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=k))[0]
        pls = r.prompt_logprobs
        pairs = []
        for i, gid in enumerate(gids):
            pos = len(tid) + i
            d = pls[pos] if pos < len(pls) else None
            ent = None if d is None else d.get(gid)
            if ent is None and d is not None:
                ent = d.get(str(gid))
            pairs.append(None if ent is None else {
                "pos": pos, "token_id": gid,
                "decode": dec[i], "refeed": float(ent.logprob),
                "rank": getattr(ent, "rank", None),
                "bitwise_equal": bits(dec[i]) == bits(float(ent.logprob)),
            })
        valid = [p for p in pairs if p is not None]
        first = next((i for i, p in enumerate(pairs)
                      if p is not None and not p["bitwise_equal"]), None)
        mism = [p for p in valid if not p["bitwise_equal"]]
        result["refeeds"][str(k)] = {
            "pairs": pairs,
            "n_compared": len(valid),
            "first_mismatch": first,
            "n_mismatch": len(mism),
            "max_abs_diff": max((abs(p["decode"] - p["refeed"]) for p in mism), default=None),
        }
    result["total_sec"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    brief = {k: {kk: v[kk] for kk in ("n_compared", "first_mismatch", "n_mismatch", "max_abs_diff")}
             for k, v in result["refeeds"].items()}
    print(json.dumps({"out": OUT, "env": result["env"], "brief": brief}, indent=2))


if __name__ == "__main__":
    main()
