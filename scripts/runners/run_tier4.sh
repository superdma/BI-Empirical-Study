#!/usr/bin/env bash
# Fourth round: in-process replay, Nsight Systems, boundary sweep, compiled
# coverage, the breakable-CUDA-graph control and the prompt-set control.
#
# Run from a checkout of this repository on the GPU host:
#   bash scripts/runners/run_tier4.sh
# Results go to results/tier4, logs and return codes to logs/tier4, and the
# captured tensors to ${BI_DATA:-$HOME/bi_project/data}.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
EM="${ROOT}/scripts/experiments/tier4"
PY="${PY:-$HOME/miniconda3/envs/vllm_update/bin/python}"
DATA="${BI_DATA:-$HOME/bi_project/data}"
mkdir -p results/tier4 logs/tier4 "${DATA}/bi_inproc_dump" "${DATA}/bi_nsys"

export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_BATCH_INVARIANT=1

step() {  # step <name> <command...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] start $name" >&2
  "$@" > "logs/tier4/$name.log" 2>&1
  echo $? > "logs/tier4/$name.rc"
  echo "[$(date +%H:%M:%S)] end $name rc=$(cat "logs/tier4/$name.rc")" >&2
}

# A4: repeat the divergent norm call inside the worker process.
step a4 env PYTHONPATH="${ROOT}/scripts/bi_hook" \
  BI_DUMP="${DATA}/bi_inproc_dump" BI_INPROC_SEQ=2548 \
  BI_INPROC_OUT="${DATA}/bi_inproc_dump/rec.jsonl" \
  "$PY" -u "${EM}/bi_inproc_replay.py" --out results/tier4/bi_inproc_replay.json

# B2: decode batch sizes 255..512 against the 255 baseline.
step b2 "$PY" -u "${EM}/bi_boundary_sweep.py" --out results/tier4/bi_boundary_sweep.json

# B4/B5/B3: the same shape matrix in three engine modes.
step b4 env VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  "$PY" -u "${EM}/bi_shape_matrix.py" --label eager_no_bcg \
  --out results/tier4/bi_shapes_no_bcg.json
step b5 "$PY" -u "${EM}/bi_shape_matrix.py" --label eager_default \
  --out results/tier4/bi_shapes_eager.json
step b3 "$PY" -u "${EM}/bi_shape_matrix.py" --label compiled --compiled \
  --out results/tier4/bi_shapes_compiled.json

# A2: two Nsight Systems traces, then the comparison.
step a2_256 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o "${DATA}/bi_nsys/p256" "$PY" -u "${EM}/bi_nsys_capture.py" --which p256
step a2_255 nsys profile -t cuda,nvtx --force-overwrite=true \
  -o "${DATA}/bi_nsys/p255" "$PY" -u "${EM}/bi_nsys_capture.py" --which p255
step a2_export nsys export --type sqlite --force-overwrite=true \
  -o "${DATA}/bi_nsys/p256.sqlite" "${DATA}/bi_nsys/p256.nsys-rep"
step a2_export2 nsys export --type sqlite --force-overwrite=true \
  -o "${DATA}/bi_nsys/p255.sqlite" "${DATA}/bi_nsys/p255.nsys-rep"
step a2_an "$PY" -u "${EM}/analyze_nsys.py" --a "${DATA}/bi_nsys/p255.sqlite" \
  --b "${DATA}/bi_nsys/p256.sqlite" --out results/tier4/bi_nsys_diff.json

# C1: prompt set versus generation length.
step c1 "$PY" -u "${EM}/bi_prompt_control.py" --out results/tier4/bi_prompt_control.json

touch logs/tier4/ALL_DONE
echo "all done" >&2
