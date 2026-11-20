"""Replay the divergent norm call and report the kernel launch it used.

The in-process replay reproduces the 256 result exactly, while the fresh
process replay lands on the 255 result with the same tensors. The two calls
differ in at least the process; this probe pins down what each launch looks
like by profiling it, and runs the same comparison with and without
`VLLM_BATCH_INVARIANT` set before vLLM is imported.

Usage:
    python bi_replay_probe.py --bi 0 --out probe_bi0.json
    python bi_replay_probe.py --bi 1 --out probe_bi1.json
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--bi", default="0")
ap.add_argument("--out", default="probe.json")
ap.add_argument("--seq", type=int, default=2548)
args = ap.parse_args()
if args.bi == "1":
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
else:
    os.environ.pop("VLLM_BATCH_INVARIANT", None)

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

DUMP = os.path.expanduser(os.environ.get("BI_DUMP", "~/bi_norm_meta"))


def load(tag, seq):
    return torch.load(os.path.join(DUMP, tag, "rank0", f"{seq:05d}.pt"),
                      weights_only=False)


def meta(tag, seq):
    for path in glob.glob(os.path.join(DUMP, "meta", f"{tag}_*.jsonl")):
        with open(path) as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("seq") == seq and rec.get("rank") == 0:
                    return rec
    return {}


def bit_diff(a, b):
    if a is None or b is None or a.numel() != b.numel():
        return None
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def main():
    pa = load("c256", args.seq)
    pb = load("c255", args.seq)
    rec = meta("c256", args.seq)
    weight = torch.load(os.path.join(DUMP, "weights", f"{rec['weight_id']}.pt"),
                        weights_only=False).to("cuda")
    eps = rec["eps"]
    from vllm import _custom_ops as ops

    x256 = pa["full_in"].to("cuda").contiguous()
    r256 = pa["full_in1"].to("cuda").contiguous()
    x255 = pb["full_in"].to("cuda").contiguous()
    r255 = pb["full_in1"].to("cuda").contiguous()

    rows = []
    with torch.no_grad():
        xfill = x256[0:1].expand(256, x256.shape[1]).contiguous()
        rfill = r256[0:1].expand(256, r256.shape[1]).contiguous()
        for label, x, r in (("x256", x256, r256), ("x255", x255, r255),
                            ("xfill256", xfill, rfill)):
            xx, rr = x.clone(), r.clone()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                ops.fused_add_rms_norm(xx, rr, weight, eps)
                torch.cuda.synchronize()
            trace_path = os.path.join(os.path.dirname(args.out),
                                      f"trace_bi{args.bi}_{label}.json")
            prof.export_chrome_trace(trace_path)
            blocks = []
            with open(trace_path) as handle:
                trace = json.load(handle)
            for ev in trace.get("traceEvents", []):
                if ev.get("cat") != "kernel" or "rms" not in ev.get("name", "").lower():
                    continue
                a = ev.get("args", {})
                blocks.append({"name": ev["name"],
                               "grid": a.get("grid"), "block": a.get("block"),
                               "shared": a.get("shared memory"),
                               "regs": a.get("registers per thread")})
            row0 = xx[0].to("cpu")
            rows.append({
                "label": label, "rows": int(x.shape[0]),
                "vs_engine256": bit_diff(row0, pa["row0"]),
                "vs_engine255": bit_diff(row0, pb["row0"]),
                "vs_other_input": bit_diff(row0, (x255 if label == "x256" else x256)[0].to("cpu")),
                "launches": blocks[:4],
            })
            print(label, rows[-1], flush=True)

    result = {"bi": args.bi, "torch": torch.__version__,
              "vllm_bi_env": os.environ.get("VLLM_BATCH_INVARIANT"),
              "seq": args.seq, "cases": rows}
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
