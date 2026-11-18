#!/bin/bash
# PP follow-ups: CUDA-graph variant offline, then the online suite.
# Launch detached:
#   setsid nohup bash ~/sgl_suite/run_sglang_pp2.sh > ~/sgl_results/run_pp2.log 2>&1 &
set -u

export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH
PY=$HOME/venvs/sglang054/bin/python
OFF=$HOME/sgl_suite/bi_sgl_offline_suite.py
HTTP=$HOME/sgl_suite/bi_sgl_http_suite.py
MODEL=$HOME/models/Qwen3-30B-A3B
export OUTDIR=$HOME/sgl_results
export TP=1
export RADIX=1

wait_health() {
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || echo 000)
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

echo "=== offline sgl_det_pp4_graph start $(date '+%F %T')"
MODEL=$MODEL PP=4 DET=1 CUDAGRAPH=1 TAG=sgl_det_pp4_graph timeout 3600 "$PY" "$OFF"
echo "=== offline sgl_det_pp4_graph exit=$? $(date '+%F %T')"

echo "=== pp online start $(date '+%F %T')"
"$PY" -m sglang.launch_server --model-path "$MODEL" --pp-size 4 \
  --mem-fraction-static 0.80 --context-length 8192 --attention-backend flashinfer \
  --enable-deterministic-inference --log-level warning > "$OUTDIR/server_sgl_det_pp4.log" 2>&1 &
pid=$!
if wait_health; then
  TAG=sgl_det_pp4_http timeout 3600 "$PY" "$HTTP"
  echo "=== pp online suite exit=$?"
else
  echo "=== pp server never became healthy"
fi
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
echo "=== pp online done $(date '+%F %T')"
