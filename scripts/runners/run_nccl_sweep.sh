#!/bin/bash
# Rerun the 32-prompt offline comparison with a pinned NCCL algorithm.
# NCCL_ALGO only takes effect at communicator setup, so each config needs its
# own server start.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
RESULTS="${RESULTS:-${ROOT}/results/tier1}"
LOGS="${LOGS:-${ROOT}/logs}"
mkdir -p "${RESULTS}" "${LOGS}"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_cfg() {
  TAG=$1; PORT=$2; ALGO=$3; PROTO=${4:-}
  LOG=${LOGS}/bi_nccl_${TAG}.server.log
  TLOG=${LOGS}/bi_nccl_${TAG}.test.log
  OUT=${RESULTS}/bi_nccl_${TAG}.json
  ENVX=""
  [ -n "$ALGO" ] && ENVX="$ENVX NCCL_ALGO=$ALGO"
  [ -n "$PROTO" ] && ENVX="$ENVX NCCL_PROTO=$PROTO"
  echo "=== NCCL START $TAG algo=$ALGO proto=$PROTO port=$PORT $(date -Is)"
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 $ENVX \
      $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 \
      --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 \
      --enable-chunked-prefill --attention-config.backend FLASH_ATTN \
      --no-enable-prefix-caching --enforce-eager --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || { echo "died"; tail -n 30 "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== NCCL HEALTH FAIL $TAG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1
  fi
  echo "=== NCCL HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT TAG=$TAG OUT=$OUT N=32 \
    $PY ${EXP}/online/bi_http_batchall.py > "$TLOG" 2>&1
  echo "=== NCCL TEST RC=$? $TAG"; tail -n 25 "$TLOG"
  grep -iE "nccl.*(algo|ring|tree|fallback|warn)" "$LOG" | head -5
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
}

run_cfg default 8061 "" ""
run_cfg ring 8062 "allreduce:ring" ""
run_cfg tree 8063 "allreduce:tree" ""
echo "=== NCCL ALL DONE $(date -Is)"
