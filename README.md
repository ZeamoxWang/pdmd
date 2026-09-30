## PDMD: Projected Distribution Matching Distillation for Video Diffusion Models<br><sub>Official MiniMax-H3 checkpoints and inference</sub>

### [Paper](https://arxiv.org/abs/2609.35768) | [Project Page](https://pdmd2026.github.io/) | [Hugging Face](https://huggingface.co/pdmd2026)

![PDMD samples](visuals/teaser.jpg)

This repo contains pre-trained PDMD checkpoints of MiniMax-H3-33B for four- and two-step joint
video–audio generation, and code to run them on a single 24GB GPU. You can find more videos, with
sound, on our [project page](https://pdmd2026.github.io/).

> [**PDMD: Projected Distribution Matching Distillation for Video Diffusion Models**](https://arxiv.org/abs/2609.35768)<br>
> Zimo Wang, Junkun Yuan, Angtian Wang, Haotian Yang, Canyu Zhang, Siyuan Yuan, Xingchang Huang,
> Bo Liu, Yizhi Wang, Yiding Yang, Chongyang Ma, Gordon Guocheng Qian
> <br>UC San Diego, ByteDance<br>

Distribution Matching Distillation (DMD) reduces video diffusion models to a few network
evaluations (NFE), but its samples can degrade during training, drifting toward oversaturation and
artifacts. We trace this instability to critic errors that accumulate in successive student
updates, and introduce Projected Distribution Matching Distillation (PDMD), which projects out the
component of the DMD update parallel to the student–critic endpoint residual, an unbiased estimate
of the critic's endpoint error. PDMD is a one-line change to DMD, with no extra loss, network, data,
model pass, or training stage. With Wan2.1, PDMD reaches a VBench total score of 83.73 at 4 NFE;
on MiniMax-H3 joint video–audio generation, it reaches a VideoGen-Eval visual total of 83.17 and
the best score on all six audio metrics among the compared 4-NFE models.

This repository contains:

* 🪐 PDMD checkpoints of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) at 4 NFE (full weights and LoRA) and 2 NFE (LoRA)
* ⚡️ An [inference script](worker/run_a10.py) that runs them on a single 24GB GPU
* 💥 A [tool](worker/fuse_lora.py) that fuses the PDMD LoRAs into the base transformer

> **Note.** This codebase is optimized for GPUs with little memory, e.g. an NVIDIA A10 (24GB of GPU
> memory, with 128GB of host RAM), by quantizing the transformer and the text encoder to int8. On
> GPUs with more memory, we recommend running in 16-bit precision with `--no-int8`, which matches
> the setting of the experiments in the paper.


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/ZeamoxWang/pdmd.git
cd pdmd
```

Install the dependencies (the versions we tested):

```bash
pip install torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0
pip install "git+https://github.com/huggingface/diffusers.git@e0abab83b5df05de9e7abd788643c1a7c1e42e28" \
  transformers==5.17.0 accelerate==1.15.0 peft==0.21.0 torchao==0.18.0 \
  safetensors==0.8.0 av==18.1.0 huggingface_hub==1.33.0 pillow numpy
```

The inference script reuses the job parsing and audio/video muxing of
[MiniMax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo):

```bash
git clone https://github.com/ModelTC/Minimax-H3-Turbo.git
git -C Minimax-H3-Turbo checkout 02e26d591f7a04d5d1a074c9566d5dd4f22f6225
```


## Pre-trained checkpoints

| Model | NFE | Type | Size | Download |
|---|---|---|---|---|
| PDMD | 4 | full transformer | 66 GB | [pdmd2026/pdmd_4NFE_full](https://huggingface.co/pdmd2026/pdmd_4NFE_full) |
| PDMD | 4 | LoRA (rank 128) | 1.4 GB | [pdmd2026/pdmd_4NFE_lora](https://huggingface.co/pdmd2026/pdmd_4NFE_lora) |
| PDMD | 2 | LoRA (rank 128) | 1.4 GB | [pdmd2026/pdmd_2NFE_lora](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) |

All checkpoints are for the base `transformer/` of MiniMax-H3 (the FL2VA/T2VA partition), which
also provides the text encoder, VAEs and schedulers. Download the base model and the checkpoint(s)
you want to run; the checkpoints are independent of each other:

```bash
hf download MiniMaxAI/MiniMax-H3 --exclude "transformer_ref/*" --exclude "FL2VA/*" --exclude "Ref2VA/*"
hf download pdmd2026/pdmd_4NFE_full --local-dir ckpt/pdmd_4NFE_full
hf download pdmd2026/pdmd_4NFE_lora --local-dir ckpt/pdmd_4NFE_lora
hf download pdmd2026/pdmd_2NFE_lora --local-dir ckpt/pdmd_2NFE_lora
```

The 4-NFE full transformer is a drop-in replacement for the base model's transformer:

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel

transformer = MiniMaxH3Transformer3DModel.from_pretrained("pdmd2026/pdmd_4NFE_full", torch_dtype=torch.bfloat16)
```

The LoRAs cover the attention projections and both feed-forward layers of every transformer and
token-refiner block, and are fused into the base transformer once (on CPU, ~15 min):

```bash
python worker/fuse_lora.py --lora ckpt/pdmd_4NFE_lora/lora_model_0.safetensors --output ckpt/pdmd_4NFE_fused
python worker/fuse_lora.py --lora ckpt/pdmd_2NFE_lora/lora_model_0.safetensors --output ckpt/pdmd_2NFE_fused
```


## Sampling

Generate a video with [`worker/run_a10.py`](worker/run_a10.py), passing the PDMD transformer and the
number of steps. For example, with the 4-NFE checkpoint:

```bash
python worker/run_a10.py \
  --transformer-path ckpt/pdmd_4NFE_full --inference-steps 4 \
  --jobs-json jobs/giant_cat_harbor_768p_4nfe.json \
  --turbo-repo Minimax-H3-Turbo --output-dir outputs
```

Use `--transformer-path ckpt/pdmd_4NFE_fused` for the 4-NFE LoRA, or
`--transformer-path ckpt/pdmd_2NFE_fused --inference-steps 2 --jobs-json jobs/giant_cat_harbor_768p_2nfe.json`
for 2 NFE. Videos are written as `outputs/<job>_<index>_<N>nfe_seed<seed>.mp4`.

**Jobs.** A job file lists prompts with their length, resolution and aspect ratio, in the format of
`examples/prompts_t2va_test.json` in MiniMax-H3-Turbo, plus an optional `inference_steps`. The
example jobs in [`jobs/`](jobs) generate a 14.4-second, 1344×768 clip of a building-sized cat over a
harbor, with sound, at seed 42. Sampling uses time shift 12 for video and 3 for audio, and no
classifier-free guidance (MiniMax-H3 is guidance-distilled).

**Many videos.** Loading the model takes ~25–30 min. To pay that once, pass several files to
`--jobs-json`, or omit it and the script keeps running as a worker that picks up job files dropped
into `--queue-dir`.

**Hardware.** This code is built for GPUs with 24GB of memory and needs **128GB of host RAM**: by
default the transformer and the Qwen3-VL text encoder are quantized to int8 and streamed from CPU to
GPU, and the VAE is moved to the GPU only to decode. On a GPU with more memory, add `--no-int8` to
skip the quantization; this keeps the same offloading and needs more host RAM. On an NVIDIA A10 at
1344×768 and 345 frames (default settings):

| | 4 NFE | 2 NFE |
|---|---|---|
| Denoising | ~25.5 min | ~13 min |
| VAE decoding | ~5 min | ~5 min |
| **Per video** (model loaded) | **~31 min** | **~16–19 min** |
| Peak GPU memory | 21 GiB | 21 GiB |

On an 80GB GPU (A100 80G, H100, etc.), run with `--no-int8`. Loading the whole model with
`ComponentsManager` automatic offload, as the official
[`inference_minimax_h3.py`](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py)
of MiniMax-H3-Turbo does, can be faster there still.


## BibTeX

```bibtex
@misc{wang2026pdmdprojecteddistributionmatching,
      title={PDMD: Projected Distribution Matching Distillation for Video Diffusion Models},
      author={Zimo Wang and Junkun Yuan and Angtian Wang and Haotian Yang and Canyu Zhang and Siyuan Yuan and Xingchang Huang and Bo Liu and Yizhi Wang and Yiding Yang and Chongyang Ma and Gordon Guocheng Qian},
      year={2026},
      eprint={2609.35768},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.35768},
}
```


## Acknowledgments

We build on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), use the helpers of
[MiniMax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo) for job parsing and muxing, and
follow the low-memory recipe of the [Diffusers](https://github.com/huggingface/diffusers) MiniMax-H3
pipeline.
