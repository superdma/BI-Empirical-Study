"""Run one decode batch under Nsight Systems, tracing only the generate window.

torch.profiler already showed that block changes at 256 rows; what it cannot
show is the stream a launch goes to, whether the launch sits inside a graph,
and what else is running at the same moment. Nsight Systems records all three.

vLLM runs the model in spawned worker processes, so a `cudaProfilerApi`
capture range armed in this process would miss every kernel. Nsight follows
the workers by injection instead, and the trace therefore covers model
loading as well; `analyze_nsys.py` filters on the norm kernels, which only
run during the forward pass.

Usage:
    nsys profile -t cuda,nvtx --force-overwrite=true -o ~/bi_nsys/p256 \
      python bi_nsys_capture.py --which p256
"""
import argparse
import os
import time

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402
import torch  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3-30B-A3B"))
TARGET = "The capital of France is"
FILLERS = [f"Question {i}: describe the number {i} in one short sentence."
           for i in range(1, 258)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["p255", "p256"], required=True)
    ap.add_argument("--max-tokens", type=int, default=6)
    args = ap.parse_args()

    llm = LLM(model=MODEL, tensor_parallel_size=4, enforce_eager=True,
              gpu_memory_utilization=0.88, max_model_len=8192,
              max_num_batched_tokens=8192, enable_prefix_caching=False,
              enable_chunked_prefill=True,
              attention_config={"backend": "FLASH_ATTN"},
              max_num_seqs=512, disable_log_stats=True)
    tok = llm.get_tokenizer()
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    filler_ids = [tok.encode(p, add_special_tokens=False) for p in FILLERS]
    prompts = ([target_ids] + filler_ids[:255])
    if args.which == "p255":
        prompts = prompts[:255]
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=5, ignore_eos=True)

    # Warm up first, then the measured batch.
    llm.generate(prompts[:8], sampling_params=sp)
    torch.cuda.synchronize()
    t = time.time()
    llm.generate(prompts, sampling_params=sp)
    torch.cuda.synchronize()
    print(args.which, len(prompts), f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
