#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${HOME}/models/Qwen3-30B-A3B
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
VLLM=${HOME}/miniconda3/envs/vllm_update/bin/vllm
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
PORT=8054
LOG=${HOME}/len_eager.server.log
setsid bash -c "exec env VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_IB_DISABLE=1 $VLLM serve $MODEL --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --enable-chunked-prefill --attention-config.backend FLASH_ATTN --no-enable-prefix-caching --enforce-eager --port $PORT" > "$LOG" 2>&1 &
PID=$!
ok=0
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; break; }
  kill -0 $PID 2>/dev/null || { echo died; break; }
  sleep 5
done
if [ "$ok" != "1" ]; then echo "=== LEN HEALTH FAIL"; tail -n 30 "$LOG"; kill -TERM -$PID; exit 1; fi
echo "=== LEN HEALTH OK after $((i*5))s"
for spec in "32 5" "32 8" "32 9" "16 30" "8 40"; do
  set -- $spec; N=$1; L=$2
  echo "=== LEN N=$N L=$L $(date -Is)"
  BASE=http://127.0.0.1:$PORT TAG=eager_n${N}_l${L} N=$N L=$L \
    OUT=${HOME}/bi_http_len_eager_n${N}_l${L}.json \
    $PY ${ROOT}/bi_http_len.py 2>&1 | tail -n 12
done
kill -TERM -$PID 2>/dev/null
for i in $(seq 1 20); do kill -0 $PID 2>/dev/null || break; sleep 2; done
kill -KILL -$PID 2>/dev/null
echo "=== LEN DONE $(date -Is)"
