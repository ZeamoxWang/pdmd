"""单张 A10 (24GB) 上的 MiniMax-H3 Turbo 推理。

按 Diffusers 文档里 24-32GB 卡的方案：transformer 和 Qwen3-VL 文本编码器以 int8
(torchao weight-only) 加载，transformer 按 block 从 CPU 流式搬运到 GPU，文本编码器
leaf 级 offload。transformer 使用 fuse_lora.py 预先融合好 Turbo LoRA 的权重。

模型只加载一次（约 30 分钟），之后常驻并轮询 --queue-dir：把 jobs JSON 放进去就会
被执行，完成后移到 done/，失败（例如 OOM）移到 failed/，进程继续等待下一个任务。
"""
import argparse
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


# torchao 的 int8 tensor 的 .to() 只接受 dtype/layout/device，group offload 开了 stream 后
# 会传 non_blocking=True 触发 AssertionError。torchao tensor 改为同步拷贝，其余不变。
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
# VAE 不常驻 GPU：去噪时 A10 的显存要全部留给 transformer 的激活值
apply_group_offloading(
    pipe.vae, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
    offload_type="leaf_level",
)
pipe.audio_vae.to("cuda")
# 瓦片大于画面 => 每个时间段只解码一整块。默认 256px 瓦片在 960x544 上要切 15 块，
# 而 leaf offload 每次前向都要重新搬运整套解码器权重，解码会慢十几倍。
pipe.vae.enable_tiling(tile_sample_min_height=4096, tile_sample_min_width=4096)
pipe.scheduler.set_shift(args.video_shift)
pipe.audio_scheduler.set_shift(args.audio_shift)
log(f"offload ready, shifts video={args.video_shift} audio={args.audio_shift}")

args.output_dir.mkdir(parents=True, exist_ok=True)
for sub in ("done", "failed"):
    (args.queue_dir / sub).mkdir(parents=True, exist_ok=True)


def run_jobs_file(jobs_json):
    for index, job in enumerate(build_jobs(jobs_json)):
        width, height = resolve_output_size(job.megapixels, job.aspect_ratio)
        seed = args.seed + index
        log(f"{jobs_json.name} job {index}: {width}x{height}, {job.num_frames} frames, "
            f"{args.inference_steps} NFE, seed {seed}")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = pipe(
                prompt=job.prompt,
                height=height,
                width=width,
                num_frames=job.num_frames,
                # 调度器的 num_inference_steps 包含终点 sigma=0，N 次 NFE 需要传 N+1
                num_inference_steps=args.inference_steps + 1,
                generator=torch.Generator().manual_seed(seed),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )
        output_path = args.output_dir / (
            f"{jobs_json.stem}_{index:02d}_{args.inference_steps}nfe_seed{seed}.mp4"
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
