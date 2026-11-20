#!/usr/bin/env bash
# BI serving config that passed every in-process test on 4x RTX A6000:
# 32 prompts solo-vs-batch, bs=256, 256-token decode/refeed, 32-way concurrent
# and bursty HTTP requests. vLLM 0.23.0+cu129, Qwen3-30B-A3B, BF16, TP=4.
#
# Important: compiled mode is only reproducible within a fixed compilation
# artifact. Do not clear the compile cache between runs you intend to compare;
# changes to vLLM/PyTorch/CUDA, chunk size, TP, or attention backend require a
# fresh validation (see docs/08_deployment.md).
set -euo pipefail

export VLLM_BATCH_INVARIANT=1
# This host has no InfiniBand; NCCL 2.28.9 segfaults in ncclNetPluginInit without it.
export NCCL_IB_DISABLE=1
# ninja is not installed in the vllm_update env; disable FlashInfer sampler JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-${HOME}/models/Qwen3-30B-A3B}"
PORT="${PORT:-8000}"

# Prefix caching is OFF by default. In this configuration the long-prompt
# fresh-vs-hit diagnosis was exact, but turning PC off removes one variable.
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
  --port "$PORT"
