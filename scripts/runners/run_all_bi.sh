#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
RESULTS="${RESULTS:-${ROOT}/results}"
LOGS="${LOGS:-${ROOT}/logs}"
mkdir -p "${RESULTS}" "${LOGS}"
# This host has no InfiniBand; NCCL 2.28.9 segfaults in its IB plugin init without this.
export NCCL_IB_DISABLE=1
V=${HOME}/miniconda3/envs/vllm_update/bin/python
cd "${ROOT}"

run_one() {
  local chunk="$1" pc="$2" out="$3"
  echo "============================================================"
  echo "=== CHUNK=${chunk} PC=${pc} OUT=${out} ==="
  echo "============================================================"
  VLLM_BATCH_INVARIANT=1 \
  MODEL=${HOME}/models/Qwen3-30B-A3B \
  TP=4 MAX_LEN=8192 GPU_UTIL=0.88 \
  CHUNK="${chunk}" PC="${pc}" MAX_TOKENS=32 OUT="${out}" \
  "$V" ${EXP}/v1/bi_test_qwen3_30b.py || echo "RUN_FAILED ${out}"
}

run_one 8192 1 ${RESULTS}/bi_results_chunk8192_pc1.json
run_one 2048 1 ${RESULTS}/bi_results_chunk2048_pc1.json
run_one 8192 0 ${RESULTS}/bi_results_chunk8192_pc0.json

echo "============================================================"
echo "=== CONTROL: batch-invariant OFF (CHUNK=8192, PC=1) ==="
echo "============================================================"
VLLM_BATCH_INVARIANT=0 \
MODEL=${HOME}/models/Qwen3-30B-A3B \
TP=4 MAX_LEN=8192 GPU_UTIL=0.88 \
CHUNK=8192 PC=1 MAX_TOKENS=32 BI=0 OUT=${RESULTS}/bi_off_chunk8192_pc1.json \
"$V" ${EXP}/v1/bi_test_qwen3_30b.py || echo "RUN_FAILED control"

echo "============================================================"
echo "=== COMPARE chunk 8192 vs 2048 (PC=1) ==="
echo "============================================================"
if [ -f ${RESULTS}/bi_results_chunk8192_pc1.json ] && [ -f ${RESULTS}/bi_results_chunk2048_pc1.json ]; then
  "$V" ${EXP}/v1/compare_bi.py \
    ${RESULTS}/bi_results_chunk8192_pc1.json \
    ${RESULTS}/bi_results_chunk2048_pc1.json || echo "COMPARE_FAILED chunk"
fi

echo "============================================================"
echo "=== COMPARE prefix cache ON vs OFF (CHUNK=8192) ==="
echo "============================================================"
if [ -f ${RESULTS}/bi_results_chunk8192_pc1.json ] && [ -f ${RESULTS}/bi_results_chunk8192_pc0.json ]; then
  "$V" ${EXP}/v1/compare_bi.py \
    ${RESULTS}/bi_results_chunk8192_pc1.json \
    ${RESULTS}/bi_results_chunk8192_pc0.json || echo "COMPARE_FAILED pc"
fi

echo "ALL_DONE"
