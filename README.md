# PDMD × MiniMax-H3: inference on a single 24GB GPU

Inference for the official **PDMD** (Projected Distribution Matching Distillation) checkpoints of
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) joint video–audio generation, on a
**single GPU with 24GB of VRAM** (an NVIDIA A10 on the NRP Nautilus Kubernetes cluster).

> ⚠️ **This version is optimized for GPUs without enough VRAM, and trades that for high host
> memory.**
>
> - **GPU:** 24GB is enough. Peak allocated GPU memory is ~21GiB at 1344×768 (~22GB in
>   `nvidia-smi`), which is close to the A10's limit.
> - **Host RAM: 128Gi** (the pod request we verified; the first runs used 160Gi). The int8
>   weights live in host RAM (~52GiB resident), and loading memory-maps the bf16 checkpoints on
>   top of that. The pod also needs 8 CPU cores.
> - **Speed:** loading and quantizing the model takes ~25–30 min, once per worker start.
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

## Reproduce our test on Nautilus

The steps below run everything on the cluster; all data lives on a PVC mounted at `/pv`, under
`/pv/h3`. Before starting, set the PVC name (`claimName`, here `zmw-vol-haosu`) in the three files
under `k8s/`, and, if you like, the resource names (`zimo-...`). The tools read the deployment
names from `GPU_DEPLOY` / `DEPLOY`.

```bash
# 1. Publish worker/ as a ConfigMap (mounted at /opt/h3 in the pods)
tools/upload_scripts.sh

# 2. One-off setup on CPU (~1 h): base model, PDMD checkpoints, fused 2-NFE transformer
kubectl apply -f k8s/setup_job.yaml
kubectl logs -f job/zimo-h3-setup

# 3. Start the GPU worker; it serves the 4-NFE checkpoint by default and is ready when the log
#    prints "watching /pv/h3/queue" (~30 min)
kubectl apply -f k8s/deployment_a10_h3.yaml
kubectl logs -f deploy/zimo-deployment-h3-a10

# 4. Generate and download the 4-NFE video (~31 min)
tools/submit.sh jobs/giant_cat_harbor_768p_4nfe.json
kubectl apply -f k8s/deployment_cpu.yaml   # small CPU pod used for downloads
tools/fetch.sh giant_cat_harbor_768p_4nfe_00_4nfe_seed42.mp4 outputs/

# 5. Switch to the 2-NFE checkpoint (the worker reloads, ~30 min), then generate (~17 min)
tools/switch_model.sh 2nfe
tools/submit.sh jobs/giant_cat_harbor_768p_2nfe.json
tools/fetch.sh giant_cat_harbor_768p_2nfe_00_2nfe_seed42.mp4 outputs/

# 6. Release the GPU when done (Nautilus flags idle GPU pods)
kubectl delete -f k8s/deployment_a10_h3.yaml
```

- A job that fails (e.g. OOM) moves to `/pv/h3/queue/failed/` and the worker keeps waiting for the
  next one; a finished job moves to `queue/done/`.
- If the worker crashes, its container exits and Kubernetes restarts it with the same
  `/pv/h3/serve_args`.
- After editing anything in `worker/`, rerun `tools/upload_scripts.sh` and restart the worker
  (`tools/switch_model.sh <current model>`).

### Test settings

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
| `worker/run_a10.py` | Persistent inference worker: loads the model once, then polls `/pv/h3/queue` |
| `worker/fuse_lora_fp32.py` | Fuses the 2-NFE LoRA into the base transformer in fp32 |
| `worker/setup_pv.sh` | One-off, re-runnable preparation of `/pv/h3` |
| `k8s/setup_job.yaml` | CPU Job that runs `setup_pv.sh` |
| `k8s/deployment_a10_h3.yaml` | GPU worker: 1× A10, 128Gi RAM, 8 CPUs, pinned dependency versions |
| `k8s/deployment_cpu.yaml` | Small CPU pod for browsing and downloading files on the PVC |
| `tools/*.sh` | Local helpers: publish scripts, switch model, submit a job, fetch a video |
| `jobs/*.json` | The test jobs |

Layout of `/pv/h3` after setup:

```
/pv/h3/
├── hf_cache/                   # MiniMax-H3 base model (HF cache)
├── Minimax-H3-Turbo/           # helper code for job parsing and muxing (pinned commit)
├── ckpt/pdmd_4NFE_full/        # 4-NFE transformer
├── ckpt/pdmd_2NFE_lora/        # 2-NFE LoRA
├── ckpt/pdmd_2NFE_fused/       # 2-NFE LoRA fused into the base transformer
├── serve_args                  # worker arguments (served checkpoint, default step count)
├── queue/{done,failed}/        # job queue
├── outputs/                    # generated videos
└── run.log                     # worker log
```

## Known issues and patches

- **torchao int8 + group offload stream**: with `use_stream` enabled, Diffusers group offloading
  calls `tensor.to(device, non_blocking=True)`, but torchao int8 tensors only accept
  `dtype/layout/device` and raise an `AssertionError`. `run_a10.py` patches this at the top by using
  a synchronous copy for torchao tensors (the speed impact is negligible).
- **`low_cpu_mem_usage=False`**: the Diffusers docs example passes this argument, but the pinned
  Diffusers version rejects it when loading with quantization, so it is omitted.
- **Do not decode a whole frame as one VAE tile.** The decoder is a ViT whose RoPE normalizes token
  positions to [-1, 1] over the tile, and it is used with 256px tiles (16×16 latent tokens).
  Enlarging the tile to cover a 1344×768 frame makes the position density far denser and the output
  shows a grid of misaligned 16px blocks. Keep the default tiling.
- **Scheduling**: the deployment uses the `Recreate` strategy, because a rolling update needs a
  second A10 while the old pod still holds one. Larger requests (160Gi / 16 CPUs) can stay
  `Pending` when the cluster is busy.
- **Copying files out**: `kubectl cp` and long `kubectl exec` streams tend to drop after ~1.5MB, so
  `tools/fetch.sh` copies in md5-checked chunks.
- Text encoding prints many "not executed" warnings for `visual.*` layers: a text-only prompt does
  not go through Qwen3-VL's vision encoder, so these can be ignored.

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
