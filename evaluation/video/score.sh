#!/usr/bin/env bash
# Score existing videos with VBench quality metrics and the fixed v4 semantic protocol.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash evaluation/video/score.sh VIDEO_DIR JUDGE_MODEL_DIR OUTPUT_DIR [NUM_GPUS]"
  echo "Optional: QUALITY_PYTHON, JUDGE_PYTHON, QUALITY_GPU, JUDGE_GPUS (comma-separated)."
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VIDEOS=$1
MODEL=$2
OUT=$3
NUM_GPUS=${4:-1}
QUALITY_PYTHON=${QUALITY_PYTHON:-python3}
JUDGE_PYTHON=${JUDGE_PYTHON:-python3}
QUALITY_GPU=${QUALITY_GPU:-0}
IFS=',' read -r -a GPU_IDS <<< "${JUDGE_GPUS:-}"
PROMPTS="$SCRIPT_DIR/../data/vgeneval/prompts_vgeneval_t2v.jsonl"
VOCAB="$SCRIPT_DIR/../data/vgeneval/vocab_v4.json"

mkdir -p "$OUT/judge" "$OUT/quality" "$OUT/summary"
"$QUALITY_PYTHON" "$SCRIPT_DIR/vgeneval_eval_info.py" \
  --videos "$VIDEOS" --prompts "$PROMPTS" --out "$OUT/input"

# Run each dimension in a separate process to release its model before the next.
for dim in subject_consistency background_consistency temporal_flickering \
           motion_smoothness dynamic_degree aesthetic_quality imaging_quality; do
  CUDA_VISIBLE_DEVICES="$QUALITY_GPU" "$QUALITY_PYTHON" "$SCRIPT_DIR/vgeneval_quality.py" \
    --info "$OUT/input/vbench_eval_info.json" --dimension "$dim" --out "$OUT/quality"
done

"$JUDGE_PYTHON" "$SCRIPT_DIR/vgeneval_semantic_judge.py" \
  --mode aux-from-vocab --model "$MODEL" --prompts "$PROMPTS" \
  --prompt-field original --vocab "$VOCAB" --aux "$OUT/aux_v4.json"

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
  gpu=${GPU_IDS[$i]:-$i}
  CUDA_VISIBLE_DEVICES="$gpu" "$JUDGE_PYTHON" "$SCRIPT_DIR/vgeneval_semantic_judge.py" \
    --mode judge --model "$MODEL" --prompts "$PROMPTS" --prompt-field original \
    --aux "$OUT/aux_v4.json" --videos "$VIDEOS" --shard "$i/$NUM_GPUS" \
    --out "$OUT/judge/shard$i.jsonl" > "$OUT/judge/shard$i.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$JUDGE_PYTHON" "$SCRIPT_DIR/vgeneval_vbench_aggregate.py" \
  --eval-dir "$OUT/quality" --judge-dir "$OUT/judge" \
  --col "$(basename "$VIDEOS")" --out "$OUT/summary" \
  --meta '{"prompt_field":"original","aux_version":"v4"}'
