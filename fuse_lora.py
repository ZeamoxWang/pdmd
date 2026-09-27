"""把 Turbo LoRA 融合进 bf16 transformer 并存盘（只需跑一次，CPU 上完成）。

int8 权重无法直接 fuse LoRA，所以先在 bf16 上融合，推理时再量化成 int8。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel

sys.path.insert(0, "/pv/h3/Minimax-H3-Turbo")
from inference_minimax_h3 import load_lora_adapter  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--lora-path", type=Path, required=True)
parser.add_argument("--lora-alpha", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

t0 = time.time()
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16
)
print(f"loaded base transformer in {time.time() - t0:.0f}s", flush=True)

load_lora_adapter(
    transformer=transformer,
    lora_path=args.lora_path,
    lora_alpha=args.lora_alpha,
    lora_scale=1.0,
    fuse_lora=True,
)
print(f"fused LoRA at {time.time() - t0:.0f}s", flush=True)

transformer.save_pretrained(args.output, safe_serialization=True, max_shard_size="10GB")
print(f"saved fused transformer to {args.output} at {time.time() - t0:.0f}s", flush=True)
