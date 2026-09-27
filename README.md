# PDMD × MiniMax-H3: inference on a single 24GB GPU

Inference for the official **PDMD** (Projected Distribution Matching Distillation) checkpoints of
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) joint video–audio generation, on a
**single GPU with 24GB of VRAM** (tested on an NVIDIA A10).

> ⚠️ **This version is optimized for GPUs without enough VRAM, and trades that for high host
> memory.**
>
> - **GPU:** 24GB is enough. Peak allocated GPU memory is ~21GiB at 1344×768 (~22GB in
>   `nvidia-smi`), which is close to the A10's limit.
> - **Host RAM: 128GB** (verified; our first runs used 160GB). The int8 weights live in host RAM
>   (~52GiB resident), and loading memory-maps the bf16 checkpoints on top of that. We used 8 CPU
>   cores.
> - **Speed:** loading and quantizing the model takes ~25–30 min, once per process start.
>
> On an **80GB card** (A100 80G, H100, etc.), this code still works, but **it is not
> recommended**. The better choice there is to keep bf16 and let `ComponentsManager` automatic
> offload move whole components on and off the GPU, as the official
> [`inference_minimax_h3.py`](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py)
> of the MiniMax-H3-Turbo repo does. The int8 quantization and fine-grained offloading used here
> to fit into 24GB only add loading time and cost a little quality on an 80GB card (see
> [Efficiency on 80GB cards](#efficiency-on-80gb-cards)).

## Checkpoints

| Model | NFE | Type | Hugging Face |
|---|---|---|---|
| PDMD full transformer | 4 | full weights (bf16, 7 shards, 66 GB) | [pdmd2026/pdmd_4NFE_full](https://huggingface.co/pdmd2026/pdmd_4NFE_full) |
| PDMD LoRA | 2 | LoRA rank 128 (fp32, 2.8 GB) | [pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) |

Both are trained on the base `transformer/` of MiniMax-H3 (the FL2VA/T2VA partition).

### Download

```bash
pip install -U huggingface_hub
hf download pdmd2026/pdmd_4NFE_full --local-dir ckpt/pdmd_4NFE_full
hf download pdmd2026/pdmd_2NFE_lora --local-dir ckpt/pdmd_2NFE_lora
```

### Load

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel
from huggingface_hub import hf_hub_download

# 4-NFE: full transformer, a drop-in replacement for the base model's transformer
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    "pdmd2026/pdmd_4NFE_full",
    torch_dtype=torch.bfloat16,
)

# 2-NFE: a LoRA to fuse into the base transformer
lora_path = hf_hub_download("pdmd2026/pdmd_2NFE_lora", "lora_model_0.safetensors")
```

The 2-NFE LoRA is **not** in PEFT format. Its keys are `transformer.<module>.lora_A.weight` /
`.lora_B.weight` (no `.default`), it has rank 128 and alpha 128 (scale 1.0), and its safetensors
metadata states the fusion rule:

```
W_base += lora_scale * (lora_B @ lora_A)
```

It covers `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj` and `ff.net.2` in all 50
transformer blocks and both token-refiner blocks (312 pairs). `worker/fuse_lora_fp32.py` applies
this rule in fp32 and casts back to bf16, and fails if any pair is left unused.

## Inference command

`worker/run_a10.py` loads the base model with the given PDMD transformer and generates the videos
described by a jobs JSON. It needs a checkout of
[ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) for job parsing and
audio/video muxing. On any Linux machine with a 24GB GPU and 128GB of RAM:

The 4-NFE and 2-NFE checkpoints are independent; each only needs the base model. To run only
2 NFE, skip the `pdmd_4NFE_full` download and the 4-NFE command below. The 2-NFE LoRA is fused
into the original (non-LoRA) transformer, `transformer/` of `MiniMaxAI/MiniMax-H3` (~62GB), which
the base model download already includes; `fuse_lora_fp32.py` finds it in the Hugging Face cache.

```bash
# Environment (the versions verified on the A10)
pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0
pip install "git+https://github.com/huggingface/diffusers.git@e0abab83b5df05de9e7abd788643c1a7c1e42e28" \
  transformers==5.17.0 accelerate==1.15.0 peft==0.21.0 torchao==0.18.0 \
  safetensors==0.8.0 av==18.1.0 huggingface_hub==1.33.0 pillow numpy
git clone https://github.com/ModelTC/Minimax-H3-Turbo.git
git -C Minimax-H3-Turbo checkout 02e26d591f7a04d5d1a074c9566d5dd4f22f6225

# Base model (text encoder, VAEs, schedulers, base transformer), needed by both
hf download MiniMaxAI/MiniMax-H3 --exclude "transformer_ref/*" "FL2VA/*" "Ref2VA/*"
# PDMD checkpoints: download only the one(s) you run
hf download pdmd2026/pdmd_4NFE_full --local-dir ckpt/pdmd_4NFE_full
hf download pdmd2026/pdmd_2NFE_lora --local-dir ckpt/pdmd_2NFE_lora

# 4 NFE
python worker/run_a10.py \
  --transformer-path ckpt/pdmd_4NFE_full --inference-steps 4 \
  --jobs-json jobs/giant_cat_harbor_768p_4nfe.json \
  --turbo-repo Minimax-H3-Turbo --output-dir outputs

# 2 NFE: fuse the LoRA into the base transformer once (CPU, ~15 min), then generate
python worker/fuse_lora_fp32.py \
  --lora ckpt/pdmd_2NFE_lora/lora_model_0.safetensors --output ckpt/pdmd_2NFE_fused
python worker/run_a10.py \
  --transformer-path ckpt/pdmd_2NFE_fused --inference-steps 2 \
  --jobs-json jobs/giant_cat_harbor_768p_2nfe.json \
  --turbo-repo Minimax-H3-Turbo --output-dir outputs
```

The videos are written as `outputs/<job name>_<index>_<N>nfe_seed<seed>.mp4`. Each command loads
the model (~25–30 min) before generating. To avoid paying that for every video, pass several files
to `--jobs-json`, or omit it and the script keeps running as a worker that polls `--queue-dir` for
job files (finished ones move to `done/`, failed ones such as OOMs to `failed/`).

## Test settings

Both test jobs use the same three-shot prompt (`jobs/*.json`): a building-sized orange tabby
over a harbor promenade, with soundscape and music descriptions.

| Setting | Value |
|---|---|
| Resolution | 1344×768 (`megapixels: 0.98`, `aspect_ratio: "16:9"`) |
| Length | 345 frames at 24 fps (~14.4 s), with stereo audio |
| NFE | 4 (`pdmd_4NFE_full`) or 2 (`pdmd_2NFE_lora` fused) |
| Time shift | 12 for video, 3 for audio (fixed in `worker/run_a10.py`) |
| Seed | 42 |
| Guidance | none (H3 is guidance-distilled) |

A job JSON has the format of `examples/prompts_t2va_test.json` in the MiniMax-H3-Turbo repo, plus
an optional `inference_steps` field that overrides the worker's default step count.

## How it fits into 24GB

In bf16, the H3 transformer is 61.7GB and the Qwen3-VL-32B text encoder is 62.1GB. The worker
follows the 24–32GB recipe of the Diffusers docs, with a few additions:

| Component | Handling |
|---|---|
| Transformer | Loaded from a full bf16 checkpoint, quantized to int8 (torchao weight-only) at load time, and streamed from CPU to GPU block by block on a CUDA stream |
| Text encoder | int8 quantization, leaf-level offload |
| Video VAE | Stays on CPU during denoising (keeping it resident makes the FFN activations OOM) and moves to the GPU as a whole only for the decode call, with the default 256px tiling |
| Audio VAE | Resident on the GPU (it is small) |

## Measured on an A10 (1344×768, 345 frames)

| Stage | 4 NFE | 2 NFE |
|---|---|---|
| Load + int8 quantization + offload setup (once per worker start) | ~25–28 min | ~25–28 min |
| Denoising | ~25.5 min (~6.4 min/step) | ~13 min |
| VAE decoding + muxing | ~5 min | ~5 min |
| **Per video, model already loaded** | **~31 min** | **~16–19 min** |
| Peak allocated GPU memory | 21.0 GiB | 21.0 GiB |

## Repository layout

| Path | Purpose |
|---|---|
| `worker/run_a10.py` | Inference: runs the given `--jobs-json` files, or keeps running as a worker that polls a queue directory |
| `worker/fuse_lora_fp32.py` | Fuses the 2-NFE LoRA into the base transformer in fp32 |
| `jobs/*.json` | The test jobs |
| `k8s/`, `tools/`, `worker/setup_pv.sh` | Optional: the Kubernetes setup we used to run the tests |

## Efficiency on 80GB cards

This code **runs fine** on an 80GB card (nothing in it depends on VRAM size), but it is noticeably
less efficient than it could be:

| Stage | This code | Reasonable approach on 80GB |
|---|---|---|
| Loading | Read bf16 → quantize to int8 → pin memory, ~25–28 min | Read bf16 directly, ~5–10 min |
| Text encoding | Leaf-level synchronous offload | Whole model on GPU, a few seconds |
| Denoising | int8 weight-only has to dequantize first; typically 1.2–1.5× slower on long sequences, with a slight quality cost | bf16 |
| VAE decoding | Whole VAE moved to the GPU per decode call | VAE resident on GPU |

So on an 80GB card it still works, but we recommend keeping bf16 with `ComponentsManager`
automatic offload instead, as the official `inference_minimax_h3.py` does.
