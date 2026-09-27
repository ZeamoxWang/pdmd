"""MiniMax-H3 inference on a single 24GB GPU (A10).

Follows the 24-32GB recipe from the Diffusers docs: the transformer and the Qwen3-VL text
encoder are loaded as int8 (torchao weight-only), the transformer is streamed from CPU to
GPU block by block, and the text encoder uses leaf-level offload. --transformer-path points
at a full transformer checkpoint (e.g. one with a LoRA already fused by fuse_lora.py).

The model is loaded once (~30 min); the worker then polls --queue-dir. A jobs JSON dropped
there is executed and moved to done/, or to failed/ on error (e.g. OOM), and the worker
keeps waiting for the next job.
"""
import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.hooks import apply_group_offloading
from diffusers.hooks import group_offloading as _group_offloading
from torchao.quantization import Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig

sys.path.insert(0, "/pv/h3/Minimax-H3-Turbo")
from inference_minimax_h3 import FPS, build_jobs, save_result_video  # noqa: E402
from resolution_util import resolve_output_size  # noqa: E402

MODEL_ID = "MiniMaxAI/MiniMax-H3"


# torchao int8 tensors only accept dtype/layout/device in .to(); with use_stream, group
# offloading passes non_blocking=True and trips an AssertionError. Copy torchao tensors
# synchronously and leave everything else unchanged.
_original_transfer = _group_offloading.ModuleGroup._transfer_tensor_to_device


def _transfer_tensor_to_device(self, tensor, source_tensor, default_stream):
    if not _group_offloading._is_torchao_tensor(source_tensor):
        return _original_transfer(self, tensor, source_tensor, default_stream)
    moved = source_tensor.to(self.onload_device)
    _group_offloading._swap_torchao_tensor(tensor, moved)
    if self.record_stream:
        _group_offloading._record_stream_torchao_tensor(tensor, default_stream)


_group_offloading.ModuleGroup._transfer_tensor_to_device = _transfer_tensor_to_device

parser = argparse.ArgumentParser()
parser.add_argument("--queue-dir", type=Path, default=Path("/pv/h3/queue"))
parser.add_argument("--transformer-path", type=Path, required=True)
parser.add_argument("--inference-steps", type=int, default=4)
parser.add_argument("--video-shift", type=float, default=12.0)
parser.add_argument("--audio-shift", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=Path("/pv/h3/outputs"))
args = parser.parse_args()

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


pipe = ModularPipeline.from_pretrained(MODEL_ID)
pipe.update_components(
    transformer=MiniMaxH3Transformer3DModel.from_pretrained(
        args.transformer_path,
        dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                "token_refiner", "norm_out", "proj_out", "audio_proj_out",
            ],
        ),
    ),
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "model.visual", "model.language_model.embed_tokens",
                "model.language_model.norm", "lm_head",
            ],
        ),
    ),
)
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
log("components loaded (int8)")

pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
offload = dict(
    onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True
)
pipe.transformer.enable_group_offload(
    offload_type="block_level", num_blocks_per_group=1, **offload
)
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
# The video VAE stays on CPU during denoising (the A10 needs all its memory for the
# transformer activations) and moves to the GPU as a whole only for the decode call.
# Keep the default 256px tiling: the decoder's RoPE normalizes positions to [-1, 1], so
# decoding a whole frame as one tile pushes the position density out of distribution
# and produces 16px block artifacts.
_vae_decode = pipe.vae.decode


def _decode_on_gpu(*decode_args, **decode_kwargs):
    pipe.vae.to("cuda")
    try:
        return _vae_decode(*decode_args, **decode_kwargs)
    finally:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()


pipe.vae.decode = _decode_on_gpu
pipe.audio_vae.to("cuda")
pipe.scheduler.set_shift(args.video_shift)
pipe.audio_scheduler.set_shift(args.audio_shift)
log(f"offload ready, shifts video={args.video_shift} audio={args.audio_shift}")

args.output_dir.mkdir(parents=True, exist_ok=True)
for sub in ("done", "failed"):
    (args.queue_dir / sub).mkdir(parents=True, exist_ok=True)


def run_jobs_file(jobs_json):
    # Each example may override the startup shifts (video_shift / audio_shift) and the
    # step count (inference_steps)
    examples = json.loads(jobs_json.read_text())["examples"]
    for index, job in enumerate(build_jobs(jobs_json)):
        width, height = resolve_output_size(job.megapixels, job.aspect_ratio)
        seed = args.seed + index
        video_shift = float(examples[index].get("video_shift", args.video_shift))
        audio_shift = float(examples[index].get("audio_shift", args.audio_shift))
        steps = int(examples[index].get("inference_steps", args.inference_steps))
        pipe.scheduler.set_shift(video_shift)
        pipe.audio_scheduler.set_shift(audio_shift)
        log(f"{jobs_json.name} job {index}: {width}x{height}, {job.num_frames} frames, "
            f"{steps} NFE, shifts video={video_shift:g} audio={audio_shift:g}, "
            f"seed {seed}")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = pipe(
                prompt=job.prompt,
                height=height,
                width=width,
                num_frames=job.num_frames,
                # The scheduler counts the terminal sigma=0 as a grid point, so N NFEs need N + 1
                num_inference_steps=steps + 1,
                generator=torch.Generator().manual_seed(seed),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )
        output_path = args.output_dir / (
            f"{jobs_json.stem}_{index:02d}_{steps}nfe"
            f"_vs{video_shift:g}_as{audio_shift:g}_seed{seed}.mp4"
        )
        save_result_video(result, output_path, FPS)
        log(f"saved {output_path}, peak GPU mem "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


log(f"watching {args.queue_dir}")
while True:
    pending = sorted(args.queue_dir.glob("*.json"))
    if not pending:
        time.sleep(10)
        continue
    jobs_json = pending[0]
    try:
        run_jobs_file(jobs_json)
        shutil.move(jobs_json, args.queue_dir / "done" / jobs_json.name)
    except Exception:
        traceback.print_exc()
        log(f"FAILED {jobs_json.name}")
        shutil.move(jobs_json, args.queue_dir / "failed" / jobs_json.name)
    torch.cuda.empty_cache()
