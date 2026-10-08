"""Few-step sampling with a trained PDMD student, e.g. for the VideoGen-Eval protocol.

Two phases so the 33B DiT and the VAEs never share a GPU:
  denoise  load the base DiT, merge the student LoRA, sample latents (.safetensors)
  decode   load the video/audio VAEs and write <id>.mp4 with sound
Run one process per GPU with --shard i/n; each process takes every n-th prompt.

Prompts are a JSONL file of {"id", "prompt"}; their features come from a cache
written by tools/data/cache_prompts.py with the SAME H3 text encoder and
presentation used for training. The sampler is plain Euler on the shifted
trailing grid, the same as DiffSynth's MiniMax-H3 pipeline with cfg_scale=1.

VideoGen-Eval protocol of the paper: the 387 H3 prompt rewrites (ids 730-1116), seed 42 for every
prompt, 124 frames at 24 fps, 4 steps at video shift 12 / audio shift 3, and the 544p canvas of each
prompt from evaluation/data/vgeneval/vgeneval_544p_sizes.json (--size-map), which lists the sizes of the evaluated clips.
Score the resulting directory with evaluation/video/score.sh of the public PDMD repo.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pdmd.data import PromptCache  # noqa: E402

VGENEVAL_SIZES = '1120x480,480x1120,960x544,544x960,736x736,832x640,640x832'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='MiniMax-H3 snapshot root containing FL2VA/')
    p.add_argument('--prompts', required=True, help='JSONL with id and prompt')
    p.add_argument('--embeddings', required=True, help='feature cache of these prompts (cache_prompts.py)')
    p.add_argument('--output', required=True)
    p.add_argument('--lora', help='student LoRA: milestone student_lora.safetensors or a rolling latest.pt')
    p.add_argument('--lora-rank', type=int, default=128)
    p.add_argument('--lora-alpha', type=int, default=128)
    p.add_argument('--nfe', type=int, default=4)
    p.add_argument('--video-shift', type=float, default=12.0)
    p.add_argument('--audio-shift', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frames', type=int, default=124)
    p.add_argument('--sizes', default=VGENEVAL_SIZES, help='comma-separated WxH, cycled by prompt row')
    p.add_argument('--size-map', help='JSON with {"sizes": {id: "WxH"}}; overrides --sizes, e.g. evaluation/data/vgeneval/vgeneval_544p_sizes.json')
    p.add_argument('--ids', help='comma-separated subset of ids to render')
    p.add_argument('--shard', default='0/1', help='i/n: this process renders rows with row %% n == i')
    p.add_argument('--phase', choices=['denoise', 'decode', 'all'], default='all')
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpus-per-sample', type=int, default=1,
                   help='split the DiT blocks over this many visible GPUs (long or high-resolution clips)')
    return p.parse_args()


def plan(a):
    rows = [json.loads(line) for line in open(a.prompts) if line.strip()]
    sizes = [tuple(int(x) for x in s.split('x')) for s in a.sizes.split(',')]
    wanted = set(a.ids.split(',')) if a.ids else None
    shard, nshard = (int(x) for x in a.shard.split('/'))
    size_map = json.loads(open(a.size_map).read())['sizes'] if a.size_map else None
    jobs = []
    for row_index, row in enumerate(rows):
        if size_map is not None:
            width, height = (int(x) for x in size_map[str(row['id'])].split('x'))
        else:
            width, height = sizes[row_index % len(sizes)]
        if wanted is not None and str(row['id']) not in wanted:
            continue
        jobs.append({'id': row['id'], 'width': width, 'height': height})
    return jobs[shard::nshard]


@torch.no_grad()
def denoise(a, jobs, latent_dir):
    from pdmd.loading import load_dit
    from pdmd.adapters import install_adapters, load_role, merge_role
    from pdmd.backend import H3Backend
    from pdmd.schedule import student_grid
    todo = [j for j in jobs if not (latent_dir / f"{j['id']}.safetensors").exists()]
    if not todo:
        return
    cache = PromptCache(a.embeddings)
    by_id = {str(r['id']): i for i, r in enumerate(cache.rows)}
    started = time.monotonic()
    model = load_dit(Path(a.base) / 'FL2VA/transformer')
    if a.lora:
        install_adapters(model, a.lora_rank, a.lora_alpha)
        if a.lora.endswith('.pt'):
            from safetensors.torch import save_file
            state = torch.load(a.lora, map_location='cpu', weights_only=True)['adapters']
            tmp = latent_dir / '_student_from_rolling.safetensors'
            save_file({k: v.contiguous() for k, v in state.items() if k.endswith('.student')}, str(tmp))
            load_role(model, tmp, 'student')
            tmp.unlink()
        else:
            load_role(model, a.lora, 'student')
        merged = merge_role(model, 'student')
        print(json.dumps({'event': 'lora_merged', 'modules': merged, 'lora': a.lora}), flush=True)
    if a.gpus_per_sample > 1:
        # Pipeline the 33B DiT over several cards: contiguous block ranges per
        # card, everything else on the first one. accelerate's hooks move the
        # hidden states between cards, so the sampler code is unchanged.
        from accelerate import dispatch_model
        n, k = len(model.blocks), a.gpus_per_sample
        device_map = {name: 0 for name, _ in model.named_children() if name != 'blocks'}
        device_map.update({f'blocks.{i}': min(i * k // n, k - 1) for i in range(n)})
        model = dispatch_model(model.eval(), device_map=device_map, main_device=0)
        a.device = 'cuda:0'
    else:
        model = model.to(a.device).eval()
    backend = H3Backend(model, torch.device(a.device), checkpoint=False)
    vg = student_grid(a.nfe, a.video_shift, a.device)
    ag = student_grid(a.nfe, a.audio_shift, a.device)
    print(json.dumps({'event': 'denoiser_ready', 'seconds': round(time.monotonic() - started, 1),
                      'video_grid': [round(float(x), 4) for x in vg],
                      'audio_grid': [round(float(x), 4) for x in ag]}), flush=True)
    from safetensors.torch import save_file
    for j in todo:
        t0 = time.monotonic()
        emb, tags = cache.load(by_id[str(j['id'])])
        # Initial noise exactly as DiffSynth's MiniMaxH3Pipeline draws it: a CPU generator
        # seeded with the same seed for video and for audio, bf16, then moved to the GPU.
        # The same seed therefore gives the same clip as the reference pipeline.
        shapes = backend.latent_shapes(a.frames, j['height'], j['width'])
        x = tuple(torch.randn(sh, generator=torch.Generator('cpu').manual_seed(a.seed), dtype=torch.bfloat16)
                  .to(a.device).float() for sh in shapes)
        cond = backend.condition(emb, tags, x)
        for i in range(a.nfe):
            v = backend.predict('teacher', x, (vg[i], ag[i]), cond)
            x = tuple(s + (grid[i + 1] - grid[i]) * vel for s, vel, grid in zip(x, v, (vg, ag)))
        out = latent_dir / f"{j['id']}.safetensors"
        save_file({'video': x[0].to(torch.bfloat16).cpu().contiguous(),
                   'audio': x[1].to(torch.bfloat16).cpu().contiguous()}, str(out) + '.tmp')
        os.replace(str(out) + '.tmp', out)
        print(json.dumps({'event': 'denoised', 'id': j['id'], 'size': f"{j['width']}x{j['height']}",
                          'seconds': round(time.monotonic() - t0, 2)}), flush=True)
    del model, backend
    torch.cuda.empty_cache()


@torch.no_grad()
def decode(a, jobs, latent_dir, video_dir):
    from safetensors.torch import load_file
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.utils.data.audio_video import write_video_audio
    todo = [j for j in jobs if not (video_dir / f"{j['id']}.mp4").exists()]
    if not todo:
        return
    root = Path(a.base) / 'FL2VA'
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=a.device, processor_config=None,
        model_configs=[ModelConfig(path=str(root / 'video_vae/source/model.safetensors')),
                       ModelConfig(path=str(root / 'audio_vae/model.safetensors'))])
    for j in todo:
        t0 = time.monotonic()
        lat = load_file(str(latent_dir / f"{j['id']}.safetensors"), device=a.device)
        pipe.load_models_to_device(['video_vae'])
        frames = pipe.video_vae.decode_video(lat['video'], dtype=torch.bfloat16, tiled=True,
                                             tile_size=256, tile_overlap=64)
        video = pipe.vae_output_to_video(frames, min_value=0, max_value=1)
        pipe.load_models_to_device(['audio_vae'])
        audio = pipe.output_audio_format_check(pipe.audio_vae.decode_audio(lat['audio'], dtype=torch.bfloat16))
        out = video_dir / f"{j['id']}.mp4"
        tmp = video_dir / f".{j['id']}.tmp.mp4"
        write_video_audio(video=video, audio=audio, output_path=str(tmp), fps=24, audio_sample_rate=32000)
        os.replace(tmp, out)
        print(json.dumps({'event': 'decoded', 'id': j['id'], 'seconds': round(time.monotonic() - t0, 2)}), flush=True)


def main():
    a = parse_args()
    out = Path(a.output)
    latent_dir, video_dir = out / 'latents', out / 'videos'
    latent_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan(a)
    if a.phase in ('denoise', 'all'):
        denoise(a, jobs, latent_dir)
    if a.phase in ('decode', 'all'):
        decode(a, jobs, latent_dir, video_dir)
    print(json.dumps({'event': 'done', 'shard': a.shard, 'videos': len(jobs)}), flush=True)


if __name__ == '__main__':
    main()
