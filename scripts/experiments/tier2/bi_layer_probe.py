"""Find the first layer whose output changes when the same batch runs twice.

The 32-prompt comparison says eager loses batch invariance at batch size 256.
This probe asks two narrower questions in one engine load:

  * is the same 256-request batch repeatable at all (twice, same process)?
  * if not, which decoder layer first differs, and by how much?

A forward hook (see bi_hook/sitecustomize.py) records row 0 of every
Qwen3MoeDecoderLayer / Qwen3MoeAttention / Qwen3MoeSparseMoeBlock output
inside the workers. Row 0 belongs to the same prompt in every run, so its
bits should not depend on the batch it travels in.

Usage:
    python bi_layer_probe.py --out results/tier2/bi_layer_probe.json
"""
import argparse
import json
import os
import shutil

def _repo_root():
    """Repository root, or the deployed layout root when there is no .git."""
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        if (os.path.isdir(os.path.join(path, "scripts"))
                and os.path.isdir(os.path.join(path, "results"))):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.dirname(os.path.abspath(__file__))
        path = parent


ROOT = _repo_root()
DUMP = os.path.expanduser(os.environ.get("BI_DUMP", "~/bi_project/data/bi_layer_dump"))
os.environ["BI_DUMP"] = DUMP
os.environ["PYTHONPATH"] = os.path.join(ROOT, "scripts", "bi_hook") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 258)]


def set_tag(tag):
    os.makedirs(DUMP, exist_ok=True)
    with open(os.path.join(DUMP, "current_tag"), "w") as handle:
        handle.write(tag + "\n")


def load(tag):
    folder = os.path.join(DUMP, tag, "rank0")
    if not os.path.isdir(folder):
        return []
    rows = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".pt"):
            rows.append(torch.load(os.path.join(folder, name), weights_only=False))
    return rows


def summarize(out):
    o = out.outputs[0]
    lps = []
    for i, tid in enumerate(o.token_ids):
        d = o.logprobs[i] if o.logprobs is not None else None
        lps.append(None if d is None or tid not in d else float(d[tid].logprob))
    return {"token_ids": list(o.token_ids), "logprobs": lps}


def bit_diff(a, b):
    if a.numel() != b.numel():
        return None, None
    if a.dtype == torch.bfloat16:
        n = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    else:
        n = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
    return n, (a.float() - b.float()).abs().max().item()


def compare(rows_a, rows_b):
    first = None
    n_diff_files = 0
    details = []
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        if a["name"] != b["name"]:
            return {"error": f"call {i}: {a['name']} vs {b['name']}"}
        n, d = bit_diff(a["row0"], b["row0"])
        if n is None:
            continue
        if n:
            n_diff_files += 1
            if first is None:
                first = {"index": i, "name": a["name"],
                         "shape": a["shape"], "n_diff": n,
                         "elems": a["row0"].numel(), "max_abs": d}
            details.append({"index": i, "name": a["name"], "n_diff": n})
    return {"calls": len(rows_a), "files_changed": n_diff_files,
            "first": first, "changed": details[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_layer_probe.json")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    shutil.rmtree(DUMP, ignore_errors=True)
    os.makedirs(DUMP, exist_ok=True)
    set_tag("init")

    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)
    p256 = [target_ids] + filler_ids[:255]
    p255 = p256[:255]

    result = {"env": {"model": MODEL, "bi": os.environ.get("VLLM_BATCH_INVARIANT"),
                      "eager": True, "tp": 4}, "runs": {}}
    plan = [("a256", p256), ("b256", p256), ("a255", p255)]
    for tag, prompts in plan:
        set_tag(tag)
        out = llm.generate(prompts, sampling_params=sp)[0]
        result["runs"][tag] = summarize(out)
        print(tag, "done", flush=True)

    set_tag("done")
    rows = {tag: load(tag) for tag, _ in plan}
    pairs = [("a256", "b256"), ("a256", "a255")]
    result["pairs"] = {}
    for a, b in pairs:
        if rows[a] and rows[b]:
            result["pairs"][f"{a}_vs_{b}"] = compare(rows[a], rows[b])
            print(f"{a} vs {b}:", json.dumps(result["pairs"][f"{a}_vs_{b}"])[:300],
                  flush=True)

    for tag, _ in plan:
        other = result["runs"][tag]
        assert other["token_ids"], tag

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
