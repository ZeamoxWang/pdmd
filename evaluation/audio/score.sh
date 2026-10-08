#!/usr/bin/env bash
# Local single-GPU orchestration of the supplied audio harness.
set -euo pipefail
ROOT=${1:?Usage: score.sh VIDEO_ROOT MODEL_BUNDLE OUTPUT_DIR [DATASET_JSON]}
MODELS=${2:?}
OUT=${3:?}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATASET=${4:-$SCRIPT_DIR/../data/vgeneval/audio_dataset.json}
PYTHON=${AUDIO_PYTHON:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
"$PYTHON" "$SCRIPT_DIR/run_aesthetics.py" --dataset "$DATASET" --root "$ROOT" \
  --out "$OUT/parts/aesthetics" --ckpt "$MODELS/audiobox/checkpoint.pt"
"$PYTHON" "$SCRIPT_DIR/run_avbench.py" --dataset "$DATASET" --root "$ROOT" \
  --out "$OUT/parts/avbench" --models "$MODELS" --work "$OUT/work"
"$PYTHON" "$SCRIPT_DIR/report.py" --dataset "$DATASET" --parts "$OUT/parts" --out "$OUT/summary"
