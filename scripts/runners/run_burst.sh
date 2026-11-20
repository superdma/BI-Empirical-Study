#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_cfg() {
  TAG=$1; EAGER=$2; PC=$3; PORT=$4; TIMEOUT=${5:-1200}
  LOG=${HOME}/burst_${TAG}.server.log
  TLOG=${HOME}/burst_${TAG}.test.log
  OUT=${HOME}/bi_http_burst_${TAG}.json
  EXTRA=""
  [ "$EAGER" = "1" ] && EXTRA="--enforce-eager"
  if [ "$PC" = "1" ]; then EXTRA="$EXTRA --enable-prefix-caching"; else EXTRA="$EXTRA --no-enable-prefix-caching"; fi
  echo "=== BURST START $TAG eager=$EAGER pc=$PC port=$PORT $(date -Is)"
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --enable-chunked-prefill --attention-config.backend FLASH_ATTN $EXTRA --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || { echo "died"; tail -n 30 "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== BURST HEALTH FAIL $TAG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1
  fi
  echo "=== BURST HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT TAG=$TAG OUT=$OUT N=32 DELAY=1.0 \
    $PY ${EXP}/online/bi_http_burst.py > "$TLOG" 2>&1
  echo "=== BURST TEST RC=$? $TAG"; tail -n 15 "$TLOG"
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
}

run_cfg eager_pc0    1 0 8043 900
run_cfg compiled_pc1 0 1 8044 1200
echo "=== BURST ALL DONE $(date -Is)"
