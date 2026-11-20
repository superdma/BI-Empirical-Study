#!/bin/bash
# Unattended driver: three offline configs + two online configs on 4x A6000.
# Launch detached:
#   setsid nohup bash sglang/run_sglang_30b.sh > sglang/results/run_30b.log 2>&1 &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH
PY="${PY:-$HOME/miniconda3/envs/sglang054/bin/python}"
OFF="$HERE/bi_sgl_offline_suite.py"
HTTP="$HERE/bi_sgl_http_suite.py"
export MODEL=$HOME/models/Qwen3-30B-A3B
export TP=4
export OUTDIR="${OUTDIR:-$HERE/results}"
export RADIX=1

run_offline() {
  local tag=$1 det=$2 graph=$3
  echo "=== offline $tag start $(date '+%F %T')"
  TAG=$tag DET=$det CUDAGRAPH=$graph timeout 5400 "$PY" "$OFF"
  echo "=== offline $tag exit=$? $(date '+%F %T')"
}

wait_health() {
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || echo 000)
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

run_online() {
  local tag=$1 det=$2
  local extra=""
  [ "$det" = "1" ] && extra="--enable-deterministic-inference"
  echo "=== online $tag start $(date '+%F %T')"
  "$PY" -m sglang.launch_server --model-path "$MODEL" --tp-size 4 \
    --mem-fraction-static 0.80 --context-length 8192 --attention-backend flashinfer \
    $extra --log-level warning > "$OUTDIR/server_$tag.log" 2>&1 &
  local pid=$!
  if wait_health; then
    TAG=$tag OUTDIR=$OUTDIR timeout 3600 "$PY" "$HTTP"
    echo "=== online $tag suite exit=$?"
  else
    echo "=== online $tag server never became healthy"
  fi
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 10
  echo "=== online $tag done $(date '+%F %T')"
}

run_offline sgl_det 1 1
run_offline sgl_nodet 0 1
run_offline sgl_det_nograph 1 0
run_online sgl_det_http 1
run_online sgl_nodet_http 0

echo "=== all done $(date '+%F %T')"
