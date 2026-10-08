#!/bin/bash
# Render the 387 VideoGen-Eval prompts with a trained student, one process per GPU.
#
#   MODEL_ROOT=/data/MiniMax-H3 LORA=runs/pdmd4/milestones/iter_002500/student_lora.safetensors \
#   PROMPTS=<pdmd repo>/eval/prompt/prompts_vgeneval_h3pe_v2.jsonl OUT=renders/pdmd4_2500 \
#   bash scripts/sample_vgeneval.sh
#
# Then score OUT/videos with eval/video/score.sh of https://github.com/ZeamoxWang/pdmd
# (VBench quality + Qwen semantic judge; Total = (4*Quality + Semantic) / 5).
# Requires `python scripts/download_assets.py --eval`.
set -euo pipefail
cd "$(dirname "$0")/.."
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
EMB=${EMB:-$OUT/prompt_features}
if [ ! -f "$EMB/COMPLETE-rank-0000.json" ]; then
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python scripts/cache_prompts.py --base "${MODEL_ROOT:?}" --prompts "${PROMPTS:?}" --output "$EMB"
fi
for phase in denoise decode; do
  pids=()
  for ((i=0; i<NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i python scripts/infer.py --base "$MODEL_ROOT" --prompts "$PROMPTS" --embeddings "$EMB" \
      --output "${OUT:?}" ${LORA:+--lora "$LORA"} --nfe 4 --video-shift 12 --audio-shift 3 --seed 42 --frames 124 --size-map configs/vgeneval_544p_sizes.json \
      --shard $i/$NGPU --phase $phase &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "videos: $(ls "$OUT"/videos/*.mp4 | wc -l) in $OUT/videos"
