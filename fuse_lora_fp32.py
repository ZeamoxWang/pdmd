"""Fuse a plain (non-PEFT) LoRA checkpoint into the base H3 transformer in fp32.

For LoRA files like the PDMD students (`lora_model_0.safetensors`), whose keys look like
`transformer.<module>.lora_A.weight` / `.lora_B.weight` and whose safetensors metadata
states the fusion rule `W_base += lora_scale * (lora_B @ lora_A)` together with
`lora_scale`. Each target weight is fused in fp32 and cast back to the base dtype, one
shard at a time, and every LoRA pair must be consumed exactly once.
"""
import argparse
import glob
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--lora", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--base",
    type=Path,
    default=None,
    help="Base transformer directory (default: MiniMaxAI/MiniMax-H3 transformer/ in the HF cache).",
)
parser.add_argument(
    "--lora-scale",
    type=float,
    default=None,
    help="Override the lora_scale stored in the LoRA metadata.",
)
args = parser.parse_args()

t0 = time.time()
base = args.base or Path(
    glob.glob("/pv/h3/hf_cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/*/transformer")[0]
)

with safe_open(args.lora, "pt") as f:
    meta = f.metadata() or {}
    lora = {k: f.get_tensor(k) for k in f.keys()}
print(f"LoRA metadata: {json.dumps(meta, indent=1)}", flush=True)
if meta.get("fuse") != "W_base += lora_scale * (lora_B @ lora_A)":
    raise ValueError(f"Unexpected fusion rule in LoRA metadata: {meta.get('fuse')!r}")
scale = args.lora_scale if args.lora_scale is not None else float(meta["lora_scale"])
prefix = "transformer."

pairs = {}
for key in lora:
    for part in ("lora_A", "lora_B"):
        suffix = f".{part}.weight"
        if key.endswith(suffix):
            module = key[len(prefix):-len(suffix)] if key.startswith(prefix) else None
            if module is None:
                raise ValueError(f"LoRA key without the {prefix!r} prefix: {key}")
            pairs.setdefault(module, {})[part] = lora[key]
incomplete = [m for m, p in pairs.items() if set(p) != {"lora_A", "lora_B"}]
if incomplete:
    raise ValueError(f"LoRA modules missing A or B: {incomplete[:5]}")
print(f"{len(pairs)} LoRA pairs, scale={scale}, base={base}", flush=True)

index = json.load(open(base / "diffusion_pytorch_model.safetensors.index.json"))
args.output.mkdir(parents=True, exist_ok=True)
used = set()
stats = []
for shard in sorted(set(index["weight_map"].values())):
    tensors = {}
    with safe_open(base / shard, "pt") as f:
        shard_meta = f.metadata()
        for name in f.keys():
            weight = f.get_tensor(name)
            module = name[: -len(".weight")] if name.endswith(".weight") else None
            if module in pairs:
                lora_a, lora_b = pairs[module]["lora_A"].float(), pairs[module]["lora_B"].float()
                expected = (lora_b.shape[0], lora_a.shape[1])
                if tuple(weight.shape) != expected or lora_a.shape[0] != lora_b.shape[1]:
                    raise ValueError(
                        f"Shape mismatch for {name}: weight {tuple(weight.shape)}, "
                        f"A {tuple(lora_a.shape)}, B {tuple(lora_b.shape)}"
                    )
                delta = scale * (lora_b @ lora_a)
                fused = weight.float() + delta
                stats.append((delta.norm() / weight.float().norm()).item())
                weight = fused.to(weight.dtype)
                used.add(module)
            tensors[name] = weight.contiguous()
    save_file(tensors, args.output / shard, metadata=shard_meta)
    print(f"[{time.time() - t0:5.0f}s] wrote {shard}", flush=True)

unused = sorted(set(pairs) - used)
if unused:
    raise ValueError(f"{len(unused)} LoRA pairs did not match any base weight: {unused[:5]}")
shutil.copy(base / "config.json", args.output / "config.json")
shutil.copy(
    base / "diffusion_pytorch_model.safetensors.index.json",
    args.output / "diffusion_pytorch_model.safetensors.index.json",
)
stats.sort()
print(
    f"fused {len(used)}/{len(pairs)} pairs; relative delta norm "
    f"median {stats[len(stats) // 2]:.2e}, max {stats[-1]:.2e}; "
    f"done in {time.time() - t0:.0f}s -> {args.output}",
    flush=True,
)
