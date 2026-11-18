#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

free_gpus() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    if [ "${used:-99999}" -lt 2000 ]; then return 0; fi
    sleep 5
  done
  echo "WARN GPU memory still used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

run_cfg() {
  TAG=$1; BI=$2; EAGER=$3; CHUNK=$4; PC=$5; PORT=$6; TIMEOUT=${7:-700}
  LOG=${HOME}/http_${TAG}.log
  TLOG=${HOME}/bi_http_${TAG}.test.log
  OUT=${HOME}/bi_http_${TAG}.json
  echo "=== START $TAG bi=$BI eager=$EAGER chunk=$CHUNK pc=$PC port=$PORT $(date -Is)"
  free_gpus
  EXTRA=""
  if [ "$EAGER" = "1" ]; then EXTRA="--enforce-eager"; fi
  if [ "$PC" = "1" ]; then EXTRA="$EXTRA --enable-prefix-caching"; else EXTRA="$EXTRA --no-enable-prefix-caching"; fi
  setsid bash -c "exec env VLLM_BATCH_INVARIANT=$BI VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens $CHUNK --enable-chunked-prefill --attention-config.backend FLASH_ATTN $EXTRA --port $PORT" > "$LOG" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 $((TIMEOUT/5))); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    if [ "$code" = "200" ]; then ok=1; break; fi
    if ! kill -0 $PID 2>/dev/null; then echo "server died"; break; fi
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "=== HEALTH FAIL $TAG"
    tail -n 40 "$LOG"
    kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null
    return 1
  fi
  echo "=== HEALTH OK $TAG after $((i*5))s"
  BASE=http://127.0.0.1:$PORT TAG=$TAG OUT=$OUT $PY ${ROOT}/bi_http_test.py > "$TLOG" 2>&1
  echo "=== TEST EXIT $? $TAG"
  tail -n 45 "$TLOG"
  kill -TERM -$PID 2>/dev/null
  for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
  kill -KILL -$PID 2>/dev/null
  sleep 5
  echo "=== END $TAG $(date -Is)"
}

run_cfg eager_8192_pc1_run1 1 1 8192 1 8011 700
run_cfg eager_8192_pc1_run2 1 1 8192 1 8012 700
run_cfg eager_2048_pc1     1 1 2048 1 8013 700
run_cfg eager_8192_pc0     1 1 8192 0 8014 700
run_cfg bi0_eager_8192_pc1 0 1 8192 1 8015 700
run_cfg compiled_8192_pc1  1 0 8192 1 8016 1800
echo "=== MATRIX DONE $(date -Is)"
