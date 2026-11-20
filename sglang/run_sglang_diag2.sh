#!/bin/bash
# Second diagnostic round: narrow down the tensor-parallel effect on a dense model.
# Launch detached:
#   setsid nohup bash sglang/run_sglang_diag2.sh > sglang/results/run_diag2.log 2>&1 &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH
PY="${PY:-$HOME/miniconda3/envs/sglang054/bin/python}"
OFF="$HERE/bi_sgl_offline_suite.py"
export OUTDIR="${OUTDIR:-$HERE/results}"
export DET=1
export CUDAGRAPH=0
export MODEL=$HOME/models/Qwen3-0.6B

run() {
  local tag=$1 tp=$2; shift 2
  echo "=== $tag start $(date '+%F %T') tp=$tp extra=$*"
  env TP=$tp TAG=$tag "$@" timeout 3600 "$PY" "$OFF"
  echo "=== $tag exit=$? $(date '+%F %T')"
}

run smoke_det_tp2 2
run smoke_det_tp4_triton 4 ATTN=triton

echo "=== diag2 done $(date '+%F %T')"
