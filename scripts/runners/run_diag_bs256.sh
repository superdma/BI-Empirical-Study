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
  TAG=$1; PC=$2; EAGER=$3; BS=$4; MT=$5; IE=$6; AT=$7; MNS=$8
  OUT=${RESULTS}/bi_diag_${TAG}.json
  LOG=${LOGS}/bi_diag_${TAG}.log
  echo "=== START $TAG $(date -Is) pc=$PC eager=$EAGER bs=$BS mt=$MT ignore_eos=$IE all_target=$AT max_num_seqs=$MNS"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print "gpu used before:",s,"MiB"}'
  env VLLM_BATCH_INVARIANT=1 NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      MODEL=${HOME}/models/Qwen3-30B-A3B TP=4 CHUNK=8192 PC=$PC EAGER=$EAGER \
      ATTN=FLASH_ATTN MOE=auto BS=$BS MAX_TOKENS=$MT TEMP=0.0 TOPP=1.0 SEED=42 \
      IGNORE_EOS=$IE ALL_TARGET=$AT MAX_NUM_SEQS=$MNS LONG_TOKENS=7600 \
      OUT=$OUT TAG=$TAG \
      $PY ${EXP}/v1/bi_extra_diag.py > "$LOG" 2>&1
  rc=$?
  echo "=== RC=$rc $TAG"
  tail -n 30 "$LOG"
  [ -f "$OUT" ] && echo "OUT_OK $OUT"
  sleep 5
}

# isolate the bs=256 divergence: cache state, EOS-driven batch shrink, batch shape
run_one bs256_pc0        0 1 256 32 0 0 0
run_one bs256_pc0_ie     0 1 256 32 1 0 0
run_one bs256_pc0_allt   0 1 256 32 1 1 0
run_one bs192_pc0        0 1 192 32 0 0 0
run_one bs128_pc0        0 1 128 32 0 0 0
run_one bs256_pc1        1 1 256 32 0 0 0
run_one bs256_comp_pc0   0 0 256 32 0 0 0
run_one bs256_mns512_pc0 0 1 256 32 0 0 512
# 256-token decode-vs-refeed divergence: cache + compilation
run_one gen256_pc0       0 1 32 256 1 0 0
run_one gen256_comp_pc0  0 0 32 256 1 0 0
echo "=== DIAG BS256 DONE $(date -Is)"
