#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_refeed() {
  TAG=$1; EAGER=$2; MT=$3
  OUT=${HOME}/bi_refeed_${TAG}.json
  LOG=${HOME}/bi_refeed_${TAG}.log
  echo "=== REFEEDB START $TAG $(date -Is) eager=$EAGER mt=$MT"
  env VLLM_BATCH_INVARIANT=1 NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      MODEL=${HOME}/models/Qwen3-30B-A3B EAGER=$EAGER PC=0 MT=$MT OUT=$OUT \
      $PY ${ROOT}/bi_refeed_diag.py > "$LOG" 2>&1
  echo "=== REFEEDB RC=$? $TAG"
  tail -n 25 "$LOG"
  sleep 5
}

run_refeed eager_pc0_mt128 1 128
run_refeed eager_pc0_mt192 1 192
run_refeed eager_pc0_mt256 1 256
run_refeed compiled_pc0_mt256 0 256
echo "=== REFEEDB DONE $(date -Is)"
