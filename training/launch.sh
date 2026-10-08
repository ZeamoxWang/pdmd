#!/bin/bash
# Launch PDMD training on one or more 8-GPU nodes.
#
#   OUTPUT=runs/pdmd4 \
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0> MASTER_PORT=29500 bash training/launch.sh
#
# Defaults reproduce the paper's 4-NFE H3 run (training/configs/pdmd_4nfe_544p.json):
# global batch 16 = one sample per GPU on 2 nodes; with fewer GPUs the same
# global batch is reached by gradient accumulation. Re-running the same command
# resumes from <OUTPUT>/checkpoints/latest.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-training/configs/pdmd_4nfe_544p.json}
NNODES=${NNODES:-1}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
if [ "$NNODES" -gt 1 ]; then
  LAUNCH="--nnodes=$NNODES --node_rank=${NODE_RANK:?} --rdzv_endpoint=${MASTER_ADDR:?}:${MASTER_PORT:-29500}"
else
  LAUNCH="--standalone"
fi
# MODEL_ROOT / CACHE_ROOT are optional: unset, the pinned Hugging Face releases are
# downloaded once per node into the Hugging Face cache.
exec python -m torch.distributed.run $LAUNCH --nproc_per_node=$NPROC training/train.py \
  ${MODEL_ROOT:+--base "$MODEL_ROOT"} ${CACHE_ROOT:+--cache "$CACHE_ROOT"} \
  --config "$CONFIG" --output "${OUTPUT:?}" --resume auto ${PROFILE:+--profile $PROFILE} "$@"
