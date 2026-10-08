"""PDMD inference on a single 80GB GPU (e.g. A100 80GB).

Runs every component in bf16, without quantization. The components are placed with the
automatic CPU offload of Diffusers' ComponentsManager: each model moves to the GPU as a whole
when it runs, and other models move back to CPU to make room. The 33B transformer (~62 GB) plus its
activations for long, high-resolution clips does not fit in 80GB, so by default it is streamed
to the GPU one block at a time (--transformer-offload block); with --transformer-offload none
it is moved to the GPU as a whole, like the other components. --transformer-path points at a
full transformer checkpoint (e.g. pdmd_4NFE_full, or a LoRA fused by fuse_lora.py). Sampling
uses time shift 12 for video.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# Must be set before torch initializes CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline  # noqa: E402

MODEL_ID = "MiniMaxAI/MiniMax-H3"
# Recommended time shifts
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
AUDIO_SHIFT_2NFE = 6.0  # Use 3.0 for paper metrics.

parser = argparse.ArgumentParser()
parser.add_argument("--transformer-path", type=Path, required=True)
parser.add_argument("--inference-steps", type=int, default=4)
parser.add_argument("--jobs-json", type=Path, nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
parser.add_argument("--turbo-repo", type=Path, default=Path("Minimax-H3-Turbo"),
                    help="Checkout of ModelTC/Minimax-H3-Turbo (job parsing and muxing helpers).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--transformer-offload", choices=("block", "none"), default="block",
                    help="block: stream the transformer to the GPU one block at a time; "
                    "none: move it to the GPU as a whole (short or low-resolution clips only).")
parser.add_argument("--memory-reserve-margin", default="12GB",
                    help="GPU memory left free by the automatic CPU offload (as in MiniMax-H3-Turbo).")
args = parser.parse_args()

sys.path.insert(0, str(args.turbo_repo))
from inference_minimax_h3 import FPS, build_jobs, save_result_video  # noqa: E402
from resolution_util import resolve_output_size  # noqa: E402

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def host_peak_gib():
    # ru_maxrss is in KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def load_jobs(jobs_json):
    """Parse and check a jobs file; returns (job, inference steps) pairs."""
    # build_jobs accepts either {"examples": [...]} or a bare list and returns one job per
    # example; each example may override the step count with inference_steps
    document = json.loads(jobs_json.read_text())
    examples = document["examples"] if isinstance(document, dict) else document
    jobs = build_jobs(jobs_json)
    unsupported = [f"{i}: {job.task}" for i, job in enumerate(jobs) if job.task != "t2va"]
    if unsupported:
        # Only prompts are passed to the pipeline, so image or reference conditioning would be
        # silently dropped
        raise ValueError(f"{jobs_json.name}: only t2va jobs are supported, got {unsupported}")
    steps = [int(example.get("inference_steps", args.inference_steps)) for example in examples]
    if min(steps) < 1:
        raise ValueError(f"{jobs_json.name}: inference steps must be at least 1, got {steps}")
    return list(zip(jobs, steps))


# Check every job file before loading the model
all_jobs = [(jobs_json, load_jobs(jobs_json)) for jobs_json in args.jobs_json]

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(MODEL_ID, components_manager=manager)
pipe.update_components(
    transformer=MiniMaxH3Transformer3DModel.from_pretrained(args.transformer_path, dtype=torch.bfloat16)
)
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
pipe.transformer.requires_grad_(False)
pipe.transformer.eval()
log(f"components loaded (bf16), host peak {host_peak_gib():.0f} GiB")

if args.transformer_offload == "block":
    # The manager leaves a group offloaded model in place and only moves the others aside
    pipe.transformer.enable_group_offload(
        onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        offload_type="block_level", num_blocks_per_group=1, use_stream=True,
        # Pin each block only while it is transferred instead of keeping a pinned copy of the
        # whole transformer, which would double its ~62 GB in host memory
        low_cpu_mem_usage=True,
    )
# Enabled after all components are registered, so that every model gets the same margin
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=args.memory_reserve_margin)
pipe.scheduler.set_shift(VIDEO_SHIFT)
pipe.audio_scheduler.set_shift(AUDIO_SHIFT)
log(f"offload ready (transformer: {args.transformer_offload}), shifts video="
    f"{pipe.scheduler.shift:g} audio={pipe.audio_scheduler.shift:g}, host peak {host_peak_gib():.0f} GiB")

args.output_dir.mkdir(parents=True, exist_ok=True)
for jobs_json, jobs in all_jobs:
    for index, (job, steps) in enumerate(jobs):
        pipe.audio_scheduler.set_shift(AUDIO_SHIFT_2NFE if steps == 2 else AUDIO_SHIFT)
        width, height = resolve_output_size(job.megapixels, job.aspect_ratio)
        seed = args.seed + index
        log(f"{jobs_json.name} job {index}: {width}x{height}, {job.num_frames} frames, "
            f"{steps} NFE, seed {seed}")
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
        output_path = args.output_dir / f"{jobs_json.stem}_{index:02d}_{steps}nfe_seed{seed}.mp4"
        save_result_video(result, output_path, FPS)
        log(f"saved {output_path}, peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
            f"host peak {host_peak_gib():.0f} GiB")
