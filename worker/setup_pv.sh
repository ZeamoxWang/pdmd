#!/usr/bin/env bash
# One-off preparation of /pv/h3 (safe to re-run; finished steps are skipped).
# Runs on CPU in k8s/setup_job.yaml:
#   1. helper code from the MiniMax-H3-Turbo repo (job parsing, video/audio muxing)
#   2. the MiniMax-H3 base model (text encoder, VAEs, schedulers, base transformer)
#   3. the PDMD 4-NFE full transformer and the PDMD 2-NFE LoRA
#   4. the 2-NFE LoRA fused into the base transformer in fp32
set -euo pipefail

ROOT=/pv/h3
TURBO_COMMIT=02e26d591f7a04d5d1a074c9566d5dd4f22f6225
export HF_HOME=$ROOT/hf_cache
mkdir -p "$ROOT/ckpt" "$ROOT/queue/done" "$ROOT/queue/failed" "$ROOT/outputs"

echo "=== [1/4] MiniMax-H3-Turbo helper code @ $TURBO_COMMIT"
if [ ! -d "$ROOT/Minimax-H3-Turbo/.git" ]; then
  rm -rf "$ROOT/Minimax-H3-Turbo"
  git clone https://github.com/ModelTC/Minimax-H3-Turbo.git "$ROOT/Minimax-H3-Turbo"
fi
git -C "$ROOT/Minimax-H3-Turbo" fetch --quiet origin "$TURBO_COMMIT" || true
git -C "$ROOT/Minimax-H3-Turbo" checkout --quiet "$TURBO_COMMIT"

echo "=== [2/4] MiniMax-H3 base model (~135GB)"
# transformer_ref/ (Ref2VA) and the original-format FL2VA/ and Ref2VA/ trees are not used
hf download MiniMaxAI/MiniMax-H3 --exclude "transformer_ref/*" "FL2VA/*" "Ref2VA/*"

echo "=== [3/4] PDMD checkpoints"
hf download pdmd2026/pdmd_4NFE_full --local-dir "$ROOT/ckpt/pdmd_4NFE_full"
hf download pdmd2026/pdmd_2NFE_lora --local-dir "$ROOT/ckpt/pdmd_2NFE_lora"

echo "=== [4/4] Fuse the 2-NFE LoRA into the base transformer (fp32, ~15 min)"
FUSED=$ROOT/ckpt/pdmd_2NFE_fused
if [ -f "$FUSED/diffusion_pytorch_model.safetensors.index.json" ]; then
  echo "already fused: $FUSED"
else
  python /opt/h3/fuse_lora_fp32.py \
    --lora "$ROOT/ckpt/pdmd_2NFE_lora/lora_model_0.safetensors" \
    --output "$FUSED"
fi

if [ ! -f "$ROOT/serve_args" ]; then
  echo "--transformer-path $ROOT/ckpt/pdmd_4NFE_full --inference-steps 4" > "$ROOT/serve_args"
fi
echo "=== setup done; serve_args: $(cat "$ROOT/serve_args")"
