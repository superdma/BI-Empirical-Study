#!/usr/bin/env bash
# Rerun A4 with the corrected hook and add the eager baseline for the
# breakable-CUDA-graph control.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/vllm_update/bin/python}"
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
export VLLM_BATCH_INVARIANT=1
mkdir -p logs out "$HOME/bi_inproc"
step() { local name="$1"; shift; echo "[$(date +%H:%M:%S)] start $name" >&2; "$@" > "logs/$name.log" 2>&1; echo $? > "logs/$name.rc"; echo "[$(date +%H:%M:%S)] end $name rc=$(cat "logs/$name.rc")" >&2; }
step a4b env PYTHONPATH="$PWD/bi_hook" BI_DUMP="$HOME/bi_inproc_dump" BI_INPROC_SEQ=2548 \
  BI_INPROC_OUT="$HOME/bi_inproc/rec.jsonl" \
  "$PY" -u bi_inproc_replay.py --out out/bi_inproc_replay.json
step b5 "$PY" -u bi_shape_matrix.py --label eager_default --out out/bi_shapes_eager.json
touch logs/FIX_DONE
