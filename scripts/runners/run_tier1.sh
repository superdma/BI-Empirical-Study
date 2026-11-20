#!/bin/bash
# Tier-1 mechanism checks, run in order on the 4xA6000 box.
#   1. all-reduce microbenchmark (no model)
#   2. NCCL algorithm sweep on the 32-prompt comparison
#   3. prefix-cache tail scan
#   4. CUDA-graph bucket scan
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${ROOT}/scripts/experiments"
PY=${HOME}/miniconda3/envs/vllm_update/bin/python
export NCCL_IB_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

echo "=== TIER1 START $(date -Is)"

echo "--- 1a allreduce world=4 default"
$PY -m torch.distributed.run --nproc_per_node=4 ${EXP}/tier1/bi_allreduce_bench.py \
  --out ${HOME}/bi_allreduce_w4_default.json --tag default 2>&1 | tail -n 40

echo "--- 1b allreduce world=4 allreduce:tree"
NCCL_ALGO=allreduce:tree $PY -m torch.distributed.run --nproc_per_node=4 ${EXP}/tier1/bi_allreduce_bench.py \
  --out ${HOME}/bi_allreduce_w4_tree.json --tag tree 2>&1 | tail -n 40

echo "--- 1c allreduce world=2 allreduce:ring"
NCCL_ALGO=allreduce:ring $PY -m torch.distributed.run --nproc_per_node=2 ${EXP}/tier1/bi_allreduce_bench.py \
  --out ${HOME}/bi_allreduce_w2_ring.json --tag ring 2>&1 | tail -n 40

echo "--- 2 NCCL sweep"
bash ${ROOT}/run_nccl_sweep.sh

echo "--- 3 tail scan"
OUT=${HOME}/bi_tail_scan.json $PY ${EXP}/tier1/bi_tail_scan.py 2>&1 | tail -n 40

echo "--- 4 bucket scan"
OUT=${HOME}/bi_bucket_scan.json $PY ${EXP}/tier1/bi_bucket_scan.py 2>&1 | tail -n 40

echo "=== TIER1 DONE $(date -Is)"
