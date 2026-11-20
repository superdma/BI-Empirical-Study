#!/bin/bash
# Third round: pipeline parallelism instead of tensor parallelism.
# PP shards layers across GPUs and needs no all-reduce, so it separates
# "all-reduce breaks invariance" from "the model does".
#   A. dense 0.6B, PP=4
#   B. MoE 30B,   PP=4  (12 layers per stage, fits in 48 GB)
# Launch detached:
#   setsid nohup bash sglang/run_sglang_diag3.sh > sglang/results/run_diag3.log 2>&1 &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH
PY="${PY:-$HOME/miniconda3/envs/sglang054/bin/python}"
OFF="$HERE/bi_sgl_offline_suite.py"
export OUTDIR="${OUTDIR:-$HERE/results}"
export DET=1
export CUDAGRAPH=0
export TP=1

run() {
  local tag=$1 model=$2 pp=$3; shift 3
  echo "=== $tag start $(date '+%F %T') model=$model pp=$pp extra=$*"
  env MODEL=$model PP=$pp TAG=$tag "$@" timeout 3600 "$PY" "$OFF"
  echo "=== $tag exit=$? $(date '+%F %T')"
}

run smoke_det_pp4 "$HOME/models/Qwen3-0.6B" 4
run sgl_det_pp4 "$HOME/models/Qwen3-30B-A3B" 4

echo "=== diag3 done $(date '+%F %T')"
