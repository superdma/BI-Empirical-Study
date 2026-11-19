"""What does vLLM 0.23.0's batch-invariant mode actually pin?

Three questions, answered without loading a model:

1. which aten ops get replaced when VLLM_BATCH_INVARIANT=1 on SM86, and
   which kernel each one lands on;
2. whether the linear-layer tile config changes with M (the hypothesis that
   256 is a matmul tile boundary);
3. whether the MoE default config changes with M, with and without BI.

Usage:
    python bi_bi_coverage.py --out results/tier2/bi_bi_coverage.json
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import vllm.envs as venvs  # noqa: E402
from vllm.model_executor.layers import batch_invariant as bi  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    get_default_config,
)

M_SWEEP = [1, 8, 64, 128, 192, 255, 256, 257, 300, 512]
M_SWEEP_MOE = list(range(1, 601))
HIDDEN, MOE_N, MOE_K, EXPERTS, TOPK = 2048, 768, 2048, 128, 8


def gpu_kernels(fn, repeat=2):
    """Names of the CUDA kernels a callable launches."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    names = set()
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            names.add(ev.key.split("(")[0].strip())
    return sorted(names)


def op_coverage():
    torch.manual_seed(0)
    a = torch.randn(256, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, HIDDEN, device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    xs = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    outs = {}
    cases = {
        "aten::mm": lambda: torch.mm(a, w.t()),
        "aten::addmm": lambda: torch.addmm(b1, a, w.t()),
        "aten::matmul": lambda: torch.matmul(a, w.t()),
        "aten::linear": lambda: F.linear(a, w),
        "aten::softmax": lambda: torch.softmax(xs, dim=-1),
        "aten::_log_softmax": lambda: torch.log_softmax(xs, dim=-1),
        "aten::mean.dim": lambda: torch.mean(xs, dim=-1),
        "aten::bmm": lambda: torch.bmm(xs.view(8, 32, 1024), xs.view(8, 1024, 32)),
    }
    for name, fn in cases.items():
        outs[name] = gpu_kernels(fn)
    return outs


def matmul_config_trace():
    """Record the tile config the BI matmul picks for each M."""
    kernel = bi.matmul_kernel_persistent
    original = kernel.run
    seen = []

    def run(*args, **kwargs):
        if len(args) >= 7:
            seen.append(
                {
                    "M": int(args[4]),
                    "N": int(args[5]),
                    "K": int(args[6]),
                    **{
                        k: int(kwargs[k])
                        for k in (
                            "BLOCK_SIZE_M",
                            "BLOCK_SIZE_N",
                            "BLOCK_SIZE_K",
                            "GROUP_SIZE_M",
                            "NUM_SMS",
                            "num_warps",
                            "num_stages",
                        )
                        if k in kwargs
                    },
                }
            )
        return original(*args, **kwargs)

    kernel.run = run
    rows = []
    try:
        for m in M_SWEEP:
            a = torch.randn(m, MOE_K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(MOE_K, HIDDEN, device="cuda", dtype=torch.bfloat16)
            seen.clear()
            bi.matmul_persistent(a, w)
            torch.cuda.synchronize()
            cfg = seen[-1] if seen else {}
            rows.append(
                {
                    "M": m,
                    "config": {k: v for k, v in cfg.items() if k not in ("M", "N", "K")},
                    "grid": min(
                        cfg.get("NUM_SMS", 0),
                        -(-m // cfg.get("BLOCK_SIZE_M", 1))
                        * -(-HIDDEN // cfg.get("BLOCK_SIZE_N", 1)),
                    ),
                }
            )
        # lm_head shape at the two sizes that matter
        for m in (255, 256):
            a = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(HIDDEN, 151936, device="cuda", dtype=torch.bfloat16)
            seen.clear()
            bi.matmul_persistent(a, w)
            torch.cuda.synchronize()
            cfg = seen[-1] if seen else {}
            rows.append(
                {
                    "M": m,
                    "shape": "lm_head",
                    "config": {k: v for k, v in cfg.items() if k not in ("M", "N", "K")},
                }
            )
            del a, w
            torch.cuda.empty_cache()
    finally:
        kernel.run = original
    distinct = {json.dumps(r["config"], sort_keys=True) for r in rows}
    return {"rows": rows, "distinct_configs": len(distinct)}


def moe_config_trace(batch_invariant):
    old = venvs.VLLM_BATCH_INVARIANT
    venvs.VLLM_BATCH_INVARIANT = batch_invariant
    try:
        rows = []
        for m in M_SWEEP_MOE:
            cfg = get_default_config(
                m, EXPERTS, MOE_N, MOE_K, TOPK, "bfloat16", None
            )
            rows.append({"M": m, "config": dict(cfg)})
    finally:
        venvs.VLLM_BATCH_INVARIANT = old
    changes = []
    for prev, cur in zip(rows, rows[1:]):
        if prev["config"] != cur["config"]:
            changes.append(
                {
                    "at_M": cur["M"],
                    "from": prev["config"],
                    "to": cur["config"],
                }
            )
    return {
        "batch_invariant": batch_invariant,
        "rows": rows,
        "change_points": changes,
        "at_255": rows[254]["config"],
        "at_256": rows[255]["config"],
        "same_at_255_256": rows[254]["config"] == rows[255]["config"],
    }


def nccl_env_override():
    before = {
        k: os.environ.get(k)
        for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
    }
    os.environ["NCCL_ALGO"] = "allreduce:ring"
    bi.override_envs_for_invariance()
    after = {
        k: os.environ.get(k)
        for k in (
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_NVLS_ENABLE",
            "VLLM_ALLREDUCE_USE_SYMM_MEM",
        )
    }
    return {"before": before, "after": after}


def attention_rule():
    import vllm.v1.attention.backends.flash_attn as fa

    path = fa.__file__
    hits = []
    with open(path) as handle:
        for i, line in enumerate(handle, 1):
            if "num_splits=1 if" in line or "batch_invariant_enabled" in line:
                hits.append({"line": i, "text": line.strip()})
    return {"file": "vllm/v1/attention/backends/flash_attn.py", "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bi_bi_coverage.json")
    args = ap.parse_args()

    bi.enable_batch_invariant_mode()
    torch.cuda.init()

    data = {
        "vllm_version": __import__("vllm").__version__,
        "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "device": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "capability": list(torch.cuda.get_device_capability(0)),
        "op_coverage": op_coverage(),
        "matmul": matmul_config_trace(),
        "moe_bi_on": moe_config_trace(True),
        "moe_bi_off": moe_config_trace(False),
        "nccl_env": nccl_env_override(),
        "attention": attention_rule(),
        "hidden_size": HIDDEN,
        "moe_dims": {"E": EXPERTS, "N": MOE_N, "K": MOE_K, "topk": TOPK},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(data, handle, indent=2)

    print("=== ops")
    for name, ks in data["op_coverage"].items():
        print(f"  {name:22s} -> {', '.join(ks) or '(no kernel recorded)'}")
    print("=== matmul configs (distinct):",
          data["matmul"]["distinct_configs"])
    for row in data["matmul"]["rows"]:
        print(f"  M={row['M']:4d} {row['config']}")
    print("=== MoE change points, BI on:",
          [c["at_M"] for c in data["moe_bi_on"]["change_points"]])
    print("=== MoE change points, BI off:",
          [c["at_M"] for c in data["moe_bi_off"]["change_points"]])
    print("    same at 255/256:",
          data["moe_bi_on"]["same_at_255_256"],
          data["moe_bi_off"]["same_at_255_256"])
    print("=== NCCL env after init_batch_invariance:")
    for k, v in data["nccl_env"]["after"].items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
