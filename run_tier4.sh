#!/usr/bin/env bash
# Fourth round: in-process replay, Nsight Systems, boundary sweep, compiled
# coverage and the breakable-CUDA-graph control. Run on the GPU host from the
# directory that holds the scripts (default ~/bi_tier4):
#
#   nohup bash run_tier4.sh > logs/driver.log 2>&1 &
#
# Each step writes its own log and a .rc file, and ALL_DONE marks the end.
set -uo pipefail

cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/vllm_update/bin/python}"
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_BATCH_INVARIANT=1

mkdir -p logs out "$HOME/bi_inproc" "$HOME/bi_nsys"

step() {  # step <name> <command...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] start $name" >&2
  "$@" > "logs/$name.log" 2>&1
  echo $? > "logs/$name.rc"
  echo "[$(date +%H:%M:%S)] end $name rc=$(cat "logs/$name.rc")" >&2
}

# A4: repeat the divergent norm call inside the worker process.
step a4 env PYTHONPATH="$PWD/bi_hook" \
  BI_DUMP="$HOME/bi_inproc_dump" BI_INPROC_SEQ=2548 \
  BI_INPROC_OUT="$HOME/bi_inproc/rec.jsonl" \
  "$PY" -u bi_inproc_replay.py --out out/bi_inproc_replay.json

# B2: decode batch sizes 255..512 against the 255 baseline.
step b2 "$PY" -u bi_boundary_sweep.py --out out/bi_boundary_sweep.json

# B4: same eager matrix with the breakable CUDA graph path turned off.
step b4 env VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  "$PY" -u bi_shape_matrix.py --label eager_no_bcg \
  --out out/bi_shapes_no_bcg.json

# B3: compiled mode on the same matrix.
step b3 "$PY" -u bi_shape_matrix.py --label compiled --compiled \
  --out out/bi_shapes_compiled.json

# A2: two Nsight Systems traces, then the comparison.
step a2_256 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o "$HOME/bi_nsys/p256" "$PY" -u bi_nsys_capture.py --which p256
step a2_255 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o "$HOME/bi_nsys/p255" "$PY" -u bi_nsys_capture.py --which p255
step a2_export nsys export --type sqlite --force-overwrite=true \
  -o "$HOME/bi_nsys/p256.sqlite" "$HOME/bi_nsys/p256.nsys-rep"
step a2_export2 nsys export --type sqlite --force-overwrite=true \
  -o "$HOME/bi_nsys/p255.sqlite" "$HOME/bi_nsys/p255.nsys-rep"
step a2_an "$PY" -u analyze_nsys.py --a "$HOME/bi_nsys/p255.sqlite" \
  --b "$HOME/bi_nsys/p256.sqlite" --out out/bi_nsys_diff.json

touch logs/ALL_DONE
echo "all done" >&2
