#!/usr/bin/env bash
# Eager BI fallback for 4x RTX A6000. vLLM 0.23.0+cu129, Qwen3-30B-A3B, BF16, TP=4.
#
# WARNING (v2 result): eager mode is NOT bitwise for general prompt sets on this
# stack. In a 32-prompt solo-vs-batch test only the 5-token target matched
# (1/32); torch.compile passed 32/32. Use serve_bi_compiled.sh unless you have
# verified your exact small, synchronized workload. See docs/08_deployment.md.
set -euo pipefail

export VLLM_BATCH_INVARIANT=1
# This host has no InfiniBand; NCCL 2.28.9 segfaults in ncclNetPluginInit without it.
export NCCL_IB_DISABLE=1
# ninja is not installed in the vllm_update env; disable FlashInfer sampler JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-${HOME}/models/Qwen3-30B-A3B}"
PORT="${PORT:-8000}"

# Prefix caching is OFF by default: in eager mode a long prompt that hits the
# prefix cache can return different token ids than the same prompt freshly
# prefilled (see docs/06). Some short prompts happened to match either way in
# v1, but that was target-only and is not a general guarantee. Set PC=1 only if
# you have verified your exact workload or treat cache state as an input.
if [ "${PC:-0}" = "1" ]; then
  PREFIX_CACHE_FLAG="--enable-prefix-caching"
else
  PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
fi

exec ${HOME}/miniconda3/envs/vllm_update/bin/vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  "$PREFIX_CACHE_FLAG" \
  --enable-chunked-prefill \
  --attention-config.backend FLASH_ATTN \
  --enforce-eager \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --port "$PORT"
