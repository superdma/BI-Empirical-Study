#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
RESULTS="${RESULTS:-${ROOT}/results/http/concurrency}"
LOGS="${LOGS:-${ROOT}/logs}"
mkdir -p "${RESULTS}" "${LOGS}"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

run_cfg() {
  TAG=$1; EAGER=$2; PC=$3; PORT=$4; N=$5; FILLER=$6; TIMEOUT=${7:-700}
  LOG=${LOGS}/http_${TAG}.log
  TLOG=${LOGS}/bi_http_${TAG}.test.log
  OUT=${RESULTS}/bi_http_${TAG}.json
  echo "=== START $TAG eager=$EAGER pc=$PC port=$PORT n=$N filler=$FILLER $(date -Is)"
  EXTRA=""
  [ "$EAGER" = "1" ] && EXTRA="--enforce-eager"
  if [ "$PC" = "1" ]; then EXTRA="$EXTRA --enable-prefix-caching"; else EXTRA="$EXTRA --no-enable-prefix-caching"; fi
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --enable-chunked-prefill --attention-config.backend FLASH_ATTN $EXTRA --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && { ok=1; break; }
    kill -0 $PID 2>/dev/null || { echo "died"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== HEALTH FAIL $TAG"; tail -n 40 "$LOG"
    kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1
  fi
  echo "=== HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT N=$N FILLER_TOKENS=$FILLER OUT=$OUT TAG=$TAG \
    $PY ${EXP}/online/bi_http_concurrent.py > "$TLOG" 2>&1
  echo "=== TEST RC=$? $TAG"; tail -n 30 "$TLOG"
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
  echo "=== END $TAG $(date -Is)"
}

run_cfg conc_eager_pc0      1 0 8030 32 0   700
run_cfg conc_eager_pc0_long 1 0 8031 16 512 700
run_cfg conc_eager_pc1      1 1 8033 32 0   700
run_cfg conc_compiled_pc1   0 1 8032 32 0   1200
echo "=== CONCURRENCY DONE $(date -Is)"
