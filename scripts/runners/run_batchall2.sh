#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_cfg() {
  TAG=$1; EAGER=$2; ATTN=$3; CHUNKED=$4; PORT=$5; TIMEOUT=${6:-1200}
  LOG=${HOME}/batchall_${TAG}.server.log
  TLOG=${HOME}/batchall_${TAG}.test.log
  OUT=${HOME}/bi_http_batchall_${TAG}.json
  EXTRA=""
  [ "$EAGER" = "1" ] && EXTRA="--enforce-eager"
  if [ "$CHUNKED" = "1" ]; then EXTRA="$EXTRA --enable-chunked-prefill"; else EXTRA="$EXTRA --no-enable-chunked-prefill"; fi
  echo "=== BATCHALL2 START $TAG eager=$EAGER attn=$ATTN chunked=$CHUNKED port=$PORT $(date -Is)"
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --attention-config.backend $ATTN --no-enable-prefix-caching $EXTRA --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || { echo "died"; tail -n 30 "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== BATCHALL2 HEALTH FAIL $TAG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1
  fi
  echo "=== BATCHALL2 HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT TAG=$TAG OUT=$OUT N=32 \
    $PY ${EXP}/online/bi_http_batchall.py > "$TLOG" 2>&1
  echo "=== BATCHALL2 TEST RC=$? $TAG"; cat "$TLOG"
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
}

run_cfg eager_triton_pc0   1 TRITON_ATTN 1 8047 900
run_cfg eager_flash_nochunk_pc0 1 FLASH_ATTN 0 8048 900
echo "=== BATCHALL2 ALL DONE $(date -Is)"
