"""Summarise (grid, block) pairs of the fused add+norm kernel from traces.

Usage:
    python extract_launch_blocks.py --traces p256=/path/a.json.gz p255=/path/b.json.gz \
        --out results/tier3/bi_launch_blocks.json
"""
import argparse
import collections
import gzip
import json


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    events = [e for e in data.get("traceEvents", []) if e.get("cat") == "kernel"]
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--out", default="bi_launch_blocks.json")
    args = ap.parse_args()

    out = {}
    for spec in args.traces:
        tag, path = spec.split("=", 1)
        rows = collections.Counter()
        steps = []
        for event in load(path):
            if "fused_add_rms_norm_kernel" not in (event.get("name") or ""):
                continue
            a = event.get("args") or {}
            grid = (a.get("grid") or [0])[0]
            block = (a.get("block") or [0])[0]
            rows[(grid, block)] += 1
            if grid in (255, 256):
                steps.append([grid, block])
        out[tag] = {
            "pairs": [{"grid": g, "block": b, "count": c}
                      for (g, b), c in sorted(rows.items())],
            "decode_sequence": steps[:288],
            "decode_blocks": {str(b): steps.count([g, b])
                              for g, b in set(map(tuple, steps))},
        }
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
