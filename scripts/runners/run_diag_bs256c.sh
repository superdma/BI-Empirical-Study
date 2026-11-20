#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

run_one() {
  TAG=$1; BS=$2; MT=$3; TEMP=$4; TOPP=$5; MNS=$6
  OUT=${HOME}/bi_diag_${TAG}.json
  LOG=${HOME}/bi_diag_${TAG}.log
  echo "=== START $TAG $(date -Is) bs=$BS mt=$MT temp=$TEMP top_p=$TOPP mns=$MNS"
  env VLLM_BATCH_INVARIANT=1 NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      MODEL=${HOME}/models/Qwen3-30B-A3B TP=4 CHUNK=8192 PC=0 EAGER=1 \
      ATTN=FLASH_ATTN MOE=auto BS=$BS MAX_TOKENS=$MT TEMP=$TEMP TOPP=$TOPP SEED=42 \
      IGNORE_EOS=1 ALL_TARGET=0 MAX_NUM_SEQS=$MNS LONG_TOKENS=7600 \
      OUT=$OUT TAG=$TAG \
      $PY ${EXP}/v1/bi_extra_diag.py > "$LOG" 2>&1
  rc=$?
  echo "=== RC=$rc $TAG"
  tail -n 18 "$LOG"
  [ -f "$OUT" ] && echo "OUT_OK $OUT"
  sleep 5
}

# exact boundary and >256 behaviour
run_one bs255_pc0        255 32 0.0 1.0 0
run_one bs257_mns512_pc0 257 32 0.0 1.0 512
# sampling: do differing logits change the drawn token?
run_one bs256_temp07_pc0 256 32 0.7 0.95 0
run_one bs256_temp07_mns128_pc0 256 32 0.7 0.95 128
echo "=== DIAG BS256C DONE $(date -Is)"
