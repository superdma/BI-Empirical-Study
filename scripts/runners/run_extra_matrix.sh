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
  TAG=$1; TP=$2; CHUNK=$3; PC=$4; EAGER=$5; ATTN=$6; MOE=$7; BS=$8; MT=$9; TEMP=${10}; TOPP=${11}; LONG=${12}
  OUT=${RESULTS}/bi_extra_${TAG}.json
  LOG=${LOGS}/bi_extra_${TAG}.log
  echo "=== START $TAG $(date -Is) tp=$TP chunk=$CHUNK pc=$PC eager=$EAGER attn=$ATTN moe=$MOE bs=$BS mt=$MT temp=$TEMP long=$LONG"
  free_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
  echo "gpu used before: $free_mem MiB"
  env VLLM_BATCH_INVARIANT=1 NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      MODEL=${HOME}/models/Qwen3-30B-A3B TP=$TP CHUNK=$CHUNK PC=$PC EAGER=$EAGER \
      ATTN=$ATTN MOE=$MOE BS=$BS MAX_TOKENS=$MT TEMP=$TEMP TOPP=$TOPP SEED=42 \
      LONG_TOKENS=$LONG OUT=$OUT TAG=$TAG \
      $PY ${EXP}/v1/bi_extra_qwen3_30b.py > "$LOG" 2>&1
  rc=$?
  echo "=== RC=$rc $TAG"
  tail -n 35 "$LOG"
  [ -f "$OUT" ] && echo "OUT_OK $OUT"
  sleep 5
}

run_one tp4_eager_bs128      4 8192 1 1 FLASH_ATTN auto 128 32 0.0 1.0 7600
run_one tp4_eager_bs256      4 8192 1 1 FLASH_ATTN auto 256 32 0.0 1.0 7600
run_one tp4_eager_temp07     4 8192 1 1 FLASH_ATTN auto 32 32 0.7 0.95 7600
run_one tp4_eager_gen256     4 8192 1 1 FLASH_ATTN auto 32 256 0.0 1.0 7600
run_one tp2_eager_bs32       2 8192 1 1 FLASH_ATTN auto 32 32 0.0 1.0 7600
run_one tp2_eager_bs32_repeat 2 8192 1 1 FLASH_ATTN auto 32 32 0.0 1.0 7600
run_one tp4_tritonattn       4 8192 1 1 TRITON_ATTN auto 32 32 0.0 1.0 7600
run_one tp4_moe_triton_unfused 4 8192 1 1 FLASH_ATTN triton_unfused 32 32 0.0 1.0 7600
run_one tp4_moe_cutlass      4 8192 1 1 FLASH_ATTN cutlass 32 32 0.0 1.0 7600
run_one tp4_pc0_long8180     4 8192 0 1 FLASH_ATTN auto 4 8 0.0 1.0 8180
run_one tp4_pc1_long8180     4 8192 1 1 FLASH_ATTN auto 4 8 0.0 1.0 8180
run_one tp4_eager_chunk2048_bs128 4 2048 1 1 FLASH_ATTN auto 128 32 0.0 1.0 7600
echo "=== EXTRA MATRIX DONE $(date -Is)"
