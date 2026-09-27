# MiniMax-H3 Turbo LoRA on a single 24GB GPU

Configs and scripts for running [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) with the
[Turbo LoRA](https://github.com/ModelTC/Minimax-H3-Turbo) (FL2VA Turbo 4-step v0.1) on a **single
GPU with 24GB of VRAM** (an NVIDIA A10 on the NRP Nautilus cluster).

> ⚠️ **This version is optimized for 24GB of VRAM.**
>
> On an **80GB card** (A100 80G, H100, etc.), this script still works, but **it is not
> recommended**. The better choice there is the Turbo repo's official
> [`inference_minimax_h3.py`](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py)
> with `--fuse-lora`. It uses `ComponentsManager` automatic offload by default, which moves whole
> components on and off the GPU and keeps bf16 precision. The int8 quantization and fine-grained
> offloading used here to fit into 24GB only add loading time, slow down VAE decoding, and cost a
> little quality on an 80GB card (see [Efficiency on 80GB cards](#efficiency-on-80gb-cards)).

## Why special handling is needed

In bf16, the H3 transformer is 61.7GB and the Qwen3-VL-32B text encoder is 62.1GB, and running it
unoptimized peaks above 50GB of VRAM. This repo follows the 24–32GB recipe from the Diffusers
docs, with a few additions:

| Component | Handling |
|---|---|
| Transformer | The Turbo LoRA is first fused into the bf16 weights (it cannot be fused into int8 weights), then quantized to int8 (torchao weight-only) at load time and streamed from CPU to GPU block by block on a CUDA stream |
| Text encoder | int8 quantization, leaf-level offload |
| Video VAE | Leaf-level offload, **not resident on the GPU**; keeping it resident makes the FFN activations OOM during denoising |
| Audio VAE | Resident on the GPU (it is small) |

The weights live mostly in host RAM, so the pod needs about 160Gi of memory.

## Files

| File | Purpose |
|---|---|
| `deployment_a10_h3.yaml` | Nautilus deployment: 1× A10, 160Gi RAM, `zmw-vol-haosu` PVC mounted at `/pv` |
| `fuse_lora.py` | One-off: fuses the Turbo LoRA into the bf16 transformer and saves it (runs on CPU, ~20 min) |
| `run_a10.py` | Persistent inference worker: loads the model once, then polls a queue directory for jobs |
| `jobs/giant_cat_harbor.json` | Example job (a 14-second, three-shot T2VA prompt) |

## Usage

All paths below live on the PVC under `/pv/h3`.

### 1. Start the pod

```bash
kubectl apply -f deployment_a10_h3.yaml
```

The startup command installs the dependencies. **torch must be ≥ 2.11**: the latest torchao fails to
import on the torch 2.8 that ships with the base image, so the yaml upgrades torch first.

### 2. Fetch code and weights

```bash
# inside the pod
export HF_HOME=/pv/h3/hf_cache
git clone --depth 1 https://github.com/ModelTC/Minimax-H3-Turbo.git /pv/h3/Minimax-H3-Turbo
hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v0.1.safetensors --local-dir /pv/h3/loras
hf download MiniMaxAI/MiniMax-H3 --exclude "transformer_ref/*"
```

Copy `fuse_lora.py` and `run_a10.py` from this repo to `/pv/h3/` (e.g. with `kubectl cp`).

> The base model repo also ships original-format `FL2VA/` and `Ref2VA/` directories (~124GB) that
> Diffusers does not use; you can add them to `--exclude` as well.

### 3. Fuse the LoRA (once)

```bash
cd /pv/h3
python fuse_lora.py \
  --lora-path /pv/h3/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --lora-alpha 8 \
  --output /pv/h3/transformer_turbo4step_v0.1_fused
```

The v0.1 4-step LoRA has rank 128 and alpha 8, matching the official script's default.

### 4. Start the inference worker

```bash
cd /pv/h3
export HF_HOME=/pv/h3/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python run_a10.py \
  --transformer-path /pv/h3/transformer_turbo4step_v0.1_fused \
  --inference-steps 4 > run.log 2>&1 &
```

Once loading finishes, the log prints `watching /pv/h3/queue`.

### 5. Submit jobs

Drop a jobs JSON (same format as `examples/prompts_t2va_test.json` in the Turbo repo) into the queue
directory:

```bash
cp jobs/giant_cat_harbor.json /pv/h3/queue/
```

- On success, the video is written to `/pv/h3/outputs/` and the JSON moves to `queue/done/`.
- On failure (e.g. OOM), the JSON moves to `queue/failed/` and the worker keeps waiting for the next
  job, **without reloading the model**.

## Measured on an A10 (960×544, 345 frames ≈ 14.4 s, 4 NFE)

| Stage | Time |
|---|---|
| Loading + int8 quantization | ~23.5 min |
| Offload setup (pinned memory) | ~4–8 min |
| 4 denoising steps | 7 min 28 s (~112 s/step), ~18.4GB VRAM |
| VAE decoding + muxing | ~48 min; the bottleneck is synchronous layer-by-layer transfer under leaf-level offload (~300 decoder forwards: 20 temporal chunks × 15 spatial tiles), not compute |
| **Total per video (model already loaded)** | **~57 min**; peak allocated GPU memory 12.4GiB for the whole job (~18.4GB reserved per nvidia-smi) |

Loading is paid once when the worker starts; after that, each video costs denoising + VAE decoding.

## Known issues and patches

- **torchao int8 + group offload stream**: with `use_stream` enabled, Diffusers group offloading
  calls `tensor.to(device, non_blocking=True)`, but torchao int8 tensors only accept
  `dtype/layout/device` and raise an `AssertionError`. `run_a10.py` patches this at the top by using
  a synchronous copy for torchao tensors (the speed impact is negligible).
- **`low_cpu_mem_usage=False`**: the Diffusers docs example passes this argument, but the current
  Diffusers version rejects it when loading with quantization, so it is omitted here.
- Text encoding prints many "not executed" warnings for `visual.*` layers: a text-only prompt does
  not go through Qwen3-VL's vision encoder, so these can be ignored.

## Efficiency on 80GB cards

This script **runs fine** on an 80GB card (nothing in it depends on VRAM size), but it is noticeably
less efficient than it could be:

| Stage | This script | Reasonable approach on 80GB |
|---|---|---|
| Loading | Read bf16 → quantize to int8 → pin memory, ~28 min | Read bf16 directly, ~5–10 min |
| Text encoding | Leaf-level synchronous offload | Whole model on GPU, a few seconds |
| Denoising | int8 weight-only has to dequantize first; typically 1.2–1.5× slower on long sequences, with a slight quality cost | bf16 |
| VAE decoding | Leaf-level synchronous offload, bound by PCIe and Python hooks; a faster GPU barely helps | VAE resident on GPU, 1–2 min |

So on an 80GB card it still works, but we recommend the official
`inference_minimax_h3.py --fuse-lora` instead.
