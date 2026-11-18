#!/bin/bash
# Follow-up diagnostics: which component breaks batch invariance?
#   A. dense model, TP=4            -> isolates tensor parallelism
#   B. MoE model, custom all-reduce off -> isolates the all-reduce kernel choice
#   C. MoE model, triton attention  -> isolates the attention backend
# Launch detached:
#   setsid nohup bash ~/sgl_suite/run_sglang_diag.sh > ~/sgl_results/run_diag.log 2>&1 &
set -u

export CUDA_HOME=$HOME/miniconda3/envs/sglang
export PATH=$CUDA_HOME/bin:$PATH
PY=$HOME/venvs/sglang054/bin/python
OFF=$HOME/sgl_suite/bi_sgl_offline_suite.py
export OUTDIR=$HOME/sgl_results
export DET=1
export CUDAGRAPH=0

run() {
  local tag=$1 model=$2 tp=$3; shift 3
  echo "=== $tag start $(date '+%F %T') model=$model tp=$tp extra=$*"
  env MODEL=$model TP=$tp TAG=$tag "$@" timeout 3600 "$PY" "$OFF"
  echo "=== $tag exit=$? $(date '+%F %T')"
}

run smoke_det_tp4 "$HOME/models/Qwen3-0.6B" 4
run sgl_det_dcar "$HOME/models/Qwen3-30B-A3B" 4 DCAR=1
run sgl_det_triton "$HOME/models/Qwen3-30B-A3B" 4 ATTN=triton

echo "=== diag done $(date '+%F %T')"
