#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
RESULTS="${RESULTS:-${ROOT}/results/http}"
LOGS="${LOGS:-${ROOT}/logs}"
mkdir -p "${RESULTS}" "${LOGS}"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
LOG=${LOGS}/http_diag.log
setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --enable-chunked-prefill --attention-config.backend FLASH_ATTN --enforce-eager --enable-prefix-caching --port 8020" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 140); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8020/health 2>/dev/null)
  [ "$code" = "200" ] && break
  kill -0 $PID 2>/dev/null || { echo DIED; tail -n 30 "$LOG"; exit 1; }
  sleep 5
done
echo "health=$code after $((i*5))s"
BASE=http://127.0.0.1:8020 OUT=${RESULTS}/bi_long_diag.json $PY ${EXP}/v1/bi_long_diag.py 2>&1 | tail -n 40
kill -TERM -$PID 2>/dev/null; for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done; kill -KILL -$PID 2>/dev/null
