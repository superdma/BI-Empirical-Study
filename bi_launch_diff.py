"""Compare kernel launch parameters between the 255- and 256-request runs.

The replay experiment can tell whether a different memory layout reproduces the
divergence. This one asks the complementary question: does the engine launch
the same kernels with the same grid, block and shared-memory settings?

torch.profiler (kineto) records those numbers for every kernel, so no nsys is
needed. Each run is profiled around one generate() call; the traces are parsed
here and the decode-phase RMSNorm launches are compared side by side.

Usage:
    VLLM_TORCH_PROFILER_DIR=$HOME/bi_traces python bi_launch_diff.py \
        --out results/tier3/bi_launch_diff.json
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import ProfilerConfig  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 258)]


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    events = [e for e in data.get("traceEvents", []) if e.get("cat") == "kernel"]
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def params(event):
    args = event.get("args") or {}
    return {
        "name": event.get("name"),
        "grid": args.get("grid"),
        "block": args.get("block"),
        "shared": args.get("shared memory"),
        "regs": args.get("registers per thread"),
    }


def is_rms(name):
    return bool(name) and ("rms_norm" in name.lower() or "rmsnorm" in name.lower())


def pick_trace(paths):
    """Rank 0 runs every layer, so prefer a file that says rank 0."""
    for path in paths:
        if "rank0" in os.path.basename(path) or "rank_0" in os.path.basename(path):
            return path
    return paths[0] if paths else None


def decode_rms(events):
    """RMSNorm launches during decode: the grid equals the active batch size."""
    rows = []
    for event in events:
        if not is_rms(event.get("name")):
            continue
        p = params(event)
        grid = p["grid"] or []
        if grid and grid[0] in (255, 256):
            rows.append(p)
    return rows


def first_param_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if (x["grid"], x["block"], x["shared"]) != (y["grid"], y["block"], y["shared"]):
            return {"index": i, "a": x, "b": y}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_launch_diff.json")
    ap.add_argument("--trace-dir", default=os.environ.get("VLLM_TORCH_PROFILER_DIR",
                                                         os.path.expanduser("~/bi_traces")))
    ap.add_argument("--max-tokens", type=int, default=6)
    args = ap.parse_args()

    prof_dir = args.trace_dir
    shutil.rmtree(prof_dir, ignore_errors=True)
    os.makedirs(prof_dir, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = prof_dir

    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True,
              profiler_config=ProfilerConfig(profiler="torch",
                                             torch_profiler_dir=prof_dir,
                                             torch_profiler_with_stack=False))
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    prompts = [target_ids] + filler_ids[:255]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    traces = {}
    for tag, batch in (("p256", prompts), ("p255", prompts[:255])):
        before = set(glob.glob(os.path.join(prof_dir, "*")))
        llm.start_profile()
        llm.generate(batch, sampling_params=sp)
        llm.stop_profile()
        time.sleep(2)
        new = sorted(set(glob.glob(os.path.join(prof_dir, "*"))) - before)
        traces[tag] = new
        print(tag, len(new), "trace files", flush=True)

    result = {"env": {"tp": 4, "eager": True, "max_tokens": args.max_tokens},
              "traces": {}, "decode_rms": {}, "total_sec": time.time() - t0}

    seqs = {}
    for tag, paths in traces.items():
        path = pick_trace(paths)
        events = load_events(path) if path else []
        rows = decode_rms(events)
        seqs[tag] = rows
        result["traces"][tag] = {"file": os.path.basename(path) if path else None,
                                 "kernels": len(events), "decode_rms": len(rows),
                                 "distinct": len({(r["grid"][0], r["block"][0], r["shared"])
                                                  for r in rows})}
        print(tag, result["traces"][tag], flush=True)

    a, b = seqs.get("p256", []), seqs.get("p255", [])
    result["decode_rms"] = {
        "n_256": len(a), "n_255": len(b),
        "first_param_diff": first_param_diff(a, b),
        "sample_256": a[:24],
        "sample_255": b[:24],
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result["decode_rms"]["first_param_diff"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
