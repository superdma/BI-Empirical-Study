"""Repeat the divergent RMSNorm call inside the engine's own worker.

`bi_replay_rmsnorm.py` runs the norm in a fresh process and gets the 255
answer from the 256 run's own tensors. That leaves two possibilities: the
engine's call is fine and something outside our view changes the result, or
the engine process itself computes the same inputs differently.

This script runs the 256 and the 255 batch with the layer hook active and
sets `BI_INPROC_SEQ`, so the worker repeats that one norm call with
`vllm._custom_ops` right after the module returns. Per call it records how
the repeated result compares with what the engine returned, and saves both
row 0s for an offline comparison with the other run.

Usage:
    BI_DUMP=~/bi_project/data/bi_inproc_dump BI_INPROC_SEQ=2548 BI_INPROC_OUT=~/bi_inproc/rec.jsonl \
      python bi_inproc_replay.py --out results/tier3/bi_inproc_replay.json
"""
import argparse
import json
import os
import shutil
import time

def _repo_root():
    path = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(path, ".git")):
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.dirname(os.path.abspath(__file__))
        path = parent
    return path


ROOT = _repo_root()
DUMP = os.path.expanduser(os.environ.get("BI_DUMP", "~/bi_project/data/bi_inproc_dump"))
os.environ["BI_DUMP"] = DUMP
os.environ["PYTHONPATH"] = os.path.join(ROOT, "scripts", "bi_hook") + ":" + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 258)]


def set_tag(tag):
    path = os.path.join(DUMP, "current_tag")
    if tag:
        os.makedirs(DUMP, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(tag + "\n")
    elif os.path.exists(path):
        os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_inproc_replay.json")
    ap.add_argument("--max-tokens", type=int, default=6)
    args = ap.parse_args()

    rec_path = os.environ.get("BI_INPROC_OUT", "")
    os.makedirs(os.path.dirname(rec_path) or ".", exist_ok=True)
    if rec_path and os.path.exists(rec_path):
        os.remove(rec_path)
    shutil.rmtree(DUMP, ignore_errors=True)

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    prompts = [target_ids] + filler_ids[:255]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    for tag, batch in (("i256", prompts), ("i255", prompts[:255])):
        set_tag(tag)
        t = time.time()
        llm.generate(batch, sampling_params=sp)
        print(tag, len(batch), f"{time.time() - t:.1f}s", flush=True)
    set_tag("")

    records = []
    if rec_path and os.path.exists(rec_path):
        with open(rec_path) as handle:
            records = [json.loads(line) for line in handle if line.strip()]

    result = {
        "env": {"model": MODEL, "tp": 4, "eager": True, "max_tokens": args.max_tokens,
                "inproc_seq": os.environ.get("BI_INPROC_SEQ")},
        "records": records,
        "total_sec": time.time() - t0,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(records, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
