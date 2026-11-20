#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
RESULTS="${RESULTS:-${ROOT}/results/extra}"
LOGS="${LOGS:-${ROOT}/logs}"
mkdir -p "${RESULTS}" "${LOGS}"
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

run_one() {
  TAG=$1; PC=$2; EAGER=$3; BS=$4; MT=$5; IE=$6; MNS=$7
  OUT=${RESULTS}/bi_diag_${TAG}.json
  LOG=${LOGS}/bi_diag_${TAG}.log
  echo "=== START $TAG $(date -Is) pc=$PC eager=$EAGER bs=$BS mt=$MT ignore_eos=$IE max_num_seqs=$MNS"
  env VLLM_BATCH_INVARIANT=1 NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      MODEL=${HOME}/models/Qwen3-30B-A3B TP=4 CHUNK=8192 PC=$PC EAGER=$EAGER \
      ATTN=FLASH_ATTN MOE=auto BS=$BS MAX_TOKENS=$MT TEMP=0.0 TOPP=1.0 SEED=42 \
      IGNORE_EOS=$IE ALL_TARGET=0 MAX_NUM_SEQS=$MNS LONG_TOKENS=7600 \
      OUT=$OUT TAG=$TAG \
      $PY ${EXP}/v1/bi_extra_diag.py > "$LOG" 2>&1
  rc=$?
  echo "=== RC=$rc $TAG"
  tail -n 18 "$LOG"
  [ -f "$OUT" ] && echo "OUT_OK $OUT"
  sleep 5
}

# bracket the unsafe batch-size threshold
run_one bs224_pc0          0 1 224 32 0 0
run_one bs240_pc0          0 1 240 32 0 0
# scheduler-side mitigation: cap concurrent sequences
run_one bs256_mns128_pc0   0 1 256 32 1 128
# does the logprob divergence eventually change token ids over 256 steps?
run_one bs256_gen256_pc0   0 1 256 256 1 0
run_one bs256_gen256_mns128_pc0 0 1 256 256 1 128
# compiled path at bs=256
run_one bs256_comp_gen256_pc0 0 0 256 256 1 0
echo "=== DIAG BS256B DONE $(date -Is)"
