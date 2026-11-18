#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_cfg() {
  TAG=$1; EAGER=$2; PORT=$3; TIMEOUT=${4:-1200}
  LOG=${HOME}/batchall_${TAG}.server.log
  TLOG=${HOME}/batchall_${TAG}.test.log
  OUT=${HOME}/bi_http_batchall_${TAG}.json
  EXTRA=""
  [ "$EAGER" = "1" ] && EXTRA="--enforce-eager"
  echo "=== BATCHALL START $TAG eager=$EAGER port=$PORT $(date -Is)"
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --enable-chunked-prefill --attention-config.backend FLASH_ATTN --no-enable-prefix-caching $EXTRA --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || { echo "died"; tail -n 30 "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== BATCHALL HEALTH FAIL $TAG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1
  fi
  echo "=== BATCHALL HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT TAG=$TAG OUT=$OUT N=32 \
    $PY ${ROOT}/bi_http_batchall.py > "$TLOG" 2>&1
  echo "=== BATCHALL TEST RC=$? $TAG"; cat "$TLOG"
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
}

run_cfg eager_pc0 1 8045 900
run_cfg compiled_pc1 0 8046 1200
echo "=== BATCHALL ALL DONE $(date -Is)"
