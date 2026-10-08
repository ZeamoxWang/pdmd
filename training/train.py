"""H3 PDMD trainer. Launch with torchrun (see training/launch.sh).

One iteration is ONE optimizer update, of either the critic or the student:
five critic updates are followed by one student update that reuses the
rollouts of the critic update right before it.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pdmd.loading import load_dit  # noqa: E402
from pdmd.adapters import install_adapters, role_parameters  # noqa: E402
from pdmd.backend import H3Backend, enable_cached_attention_bounds  # noqa: E402
from pdmd.engine import rollout, student_step_loss, critic_step_loss  # noqa: E402
from pdmd.schedule import update_kind, schedules, critic_times  # noqa: E402
from pdmd.run_log import recover_metrics  # noqa: E402
from pdmd.distributed import shard_base, reduce_gradients, adapter_state, restore_adapters  # noqa: E402
from pdmd.data import PromptCache, SampleStream, Prefetcher, stream_seed  # noqa: E402
from pdmd.checkpoint import AsyncCheckpointer  # noqa: E402
from pdmd.profiling import gpu_busy_summary  # noqa: E402
from pdmd.optimizer_state import move_moments  # noqa: E402
from pdmd import hub  # noqa: E402

# Keys that may change between a run and its resume without changing the math.
RESUMABLE_KEYS = {'iterations', 'save_every', 'milestone_every', 'log_every', 'provenance'}
REQUIRED = ['frames', 'buckets', 'nfe', 'train_video_shift', 'train_audio_shift', 'critic_steps',
            'global_batch', 'iterations', 'student_lr', 'critic_lr', 'adam_betas', 'weight_decay',
            'gradient_clip', 'loss_weights', 'loss_clamp', 'normalizer_floor', 'projection_eps',
            'seed', 'expected_prompts', 'lora_rank', 'lora_alpha']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=hub.MODEL,
                   help='MiniMax-H3 root containing FL2VA/transformer, or repo@revision on Hugging Face (default: pinned release)')
    p.add_argument('--cache', default=hub.CACHE,
                   help='prompt-feature cache directory, or repo@revision on Hugging Face (default: pinned public cache)')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', help="checkpoint path, or 'auto' for <output>/checkpoints/latest.pt if present")
    p.add_argument('--cache-audit', help='cache-audit.json receipt matching the cache manifests')
    p.add_argument('--iterations', type=int, help='override the configured iteration count (smoke tests)')
    p.add_argument('--profile', help='START:STOP iterations to trace on rank 0 with torch.profiler')
    p.add_argument('--no-attention-bounds-cache', action='store_true',
                   help='use the upstream per-call cu_seqlens.tolist() (for A/B profiling)')
    return p.parse_args()


def config_diff(a, b):
    keys = (set(a) | set(b)) - RESUMABLE_KEYS
    return sorted(k for k in keys if a.get(k) != b.get(k))


class Phases:
    """CUDA-event timers per phase, resolved once per iteration (after its sync)."""
    def __init__(self, enabled):
        self.enabled, self.events = enabled, []

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            stop.record()
            self.events.append((name, start, stop))

    def collect(self):
        out = {}
        for name, start, stop in self.events:
            out[name] = out.get(name, 0.0) + start.elapsed_time(stop) / 1000
        self.events = []
        return {k: round(v, 3) for k, v in out.items()}


def main():
    a = parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for key in REQUIRED:
        if key not in cfg:
            raise ValueError(f'Run config must explicitly specify {key}')
    iterations = a.iterations or cfg['iterations']
    snapshot_marker = Path(__file__).resolve().parents[1] / 'SNAPSHOT_READY'
    snapshot_id = snapshot_marker.read_text().strip() if snapshot_marker.exists() else None

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    if cfg['global_batch'] % world:
        raise ValueError('Global batch must divide evenly across ranks')
    accum = cfg['global_batch'] // world
    slots_of_rank = [micro * world + rank for micro in range(accum)]

    out = Path(a.output)
    cache_dir, hub_audit = hub.resolve_cache(a.cache)
    if a.cache_audit is None and hub_audit is not None:
        a.cache_audit = str(hub_audit)
    cache = PromptCache(cache_dir)
    if len(cache) != cfg['expected_prompts']:
        raise ValueError(f"Expected {cfg['expected_prompts']} prompts, found {len(cache)}")
    source_hashes = {r['source_sha256'] for r in cache.rows}
    if len(source_hashes) != 1:
        raise ValueError('Cache mixes prompt sources')
    cache_signature = cache.signature()
    provenance = cfg.get('provenance', {}).get('prompts')
    if isinstance(provenance, dict) and source_hashes != {provenance['output_sha256']}:
        raise ValueError('Cache source differs from pinned run provenance')
    if cfg.get('require_cache_audit') and not a.cache_audit:
        raise ValueError('Full training requires a successful cache audit receipt')
    if a.cache_audit:
        audit = json.loads(Path(a.cache_audit).read_text())
        if (audit['cache_signature'] != cache_signature or audit['samples'] != len(cache)
                or audit['source_sha256'] not in source_hashes):
            raise ValueError('Cache audit receipt does not match current manifests')

    resume_path = None
    if a.resume == 'auto':
        candidate = out / 'checkpoints' / 'latest.pt'
        resume_path = candidate if candidate.exists() else None
    elif a.resume:
        resume_path = Path(a.resume)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        run_file = out / 'run.json'
        if run_file.exists():
            if resume_path is None:
                raise FileExistsError('Existing run without a checkpoint: resume explicitly or choose a new output')
            diff = config_diff(json.loads(run_file.read_text()), cfg)
            if diff:
                raise ValueError(f'Resume config differs from existing run.json in {diff}')
        else:
            run_file.write_text(json.dumps(cfg, indent=2) + '\n')
    if world > 1:
        dist.barrier()

    torch.manual_seed(cfg['seed'])  # adapters initialized identically on every rank
    started = time.monotonic()
    model = load_dit(hub.resolve_model(a.base) / 'FL2VA/transformer')
    targets = install_adapters(model, cfg['lora_rank'], cfg['lora_alpha'])
    model = shard_base(model, device) if world > 1 else model.to(device)
    model.eval()  # no dropout; eval does not disable autograd
    if not a.no_attention_bounds_cache:
        enable_cached_attention_bounds()
    backend = H3Backend(model, device)
    params = {role: role_parameters(model, role) for role in ('student', 'critic')}
    opts = {role: torch.optim.AdamW(params[role], lr=cfg[role + '_lr'], betas=tuple(cfg['adam_betas']),
                                    eps=cfg.get('adam_eps', 1e-8), weight_decay=cfg['weight_decay'],
                                    # Moment offload moves state tensors between devices, which
                                    # the fused kernel does not allow.
                                    fused=not cfg.get('optimizer_state_offload', False))
            for role in params}
    load_seconds = time.monotonic() - started

    start = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=True)
        diff = config_diff(ckpt['config'], cfg)
        if diff or ckpt['cache_signature'] != cache_signature:
            raise ValueError(f'Resume config or corpus mismatch: {diff}')
        restore_adapters(model, ckpt['adapters'])
        for role, opt in opts.items():
            opt.load_state_dict(ckpt['optimizers'][role])
        start = ckpt['next_iteration']
        if rank == 0:
            recover_metrics(out / 'metrics.jsonl', start)
            if ckpt.get('world_size') != world:
                print(json.dumps({'event': 'resume_world_size_changed', 'was': ckpt.get('world_size'),
                                  'now': world}), flush=True)
        del ckpt
        if world > 1:
            dist.barrier()

    video_grid, audio_grid = schedules(cfg['nfe'], cfg['train_video_shift'], cfg['train_audio_shift'], device)
    stream = SampleStream(len(cache), cfg['global_batch'], cfg['critic_steps'],
                          [tuple(b) for b in cfg['buckets']], cfg['seed'])
    is_critic = lambda it: update_kind(it, cfg['critic_steps']) == 'critic'  # noqa: E731
    prefetch = Prefetcher(cache, stream, slots_of_rank, start, iterations, is_critic)
    checkpointer = AsyncCheckpointer(out, enabled=(rank == 0))
    phases = Phases(enabled=True)
    weights = tuple(cfg['loss_weights'])
    projected = cfg.get('method', 'projected_dmd') == 'projected_dmd'

    if rank == 0:
        print(json.dumps({'event': 'initialized', 'targets': len(targets), 'world': world, 'accum': accum,
                          'trainable': {k: sum(p.numel() for p in v) for k, v in params.items()},
                          'cached_prompts': len(cache), 'cache_signature': cache_signature,
                          'start': start, 'iterations': iterations, 'load_seconds': round(load_seconds, 1),
                          'video_grid': [round(float(x), 4) for x in video_grid],
                          'audio_grid': [round(float(x), 4) for x in audio_grid],
                          'source_snapshot_id': snapshot_id}), flush=True)

    def trajectory_for(iteration, slot):
        """Deterministic rollout for (iteration, global slot), independent of world size."""
        t0 = time.monotonic()
        emb, tags = prefetch.get(iteration, slot)
        data_seconds = time.monotonic() - t0
        h, w = stream.bucket(iteration, slot)
        g = torch.Generator(device=device).manual_seed(stream_seed(cfg['seed'], 'noise', iteration, slot))
        noise = backend.noise(cfg['frames'], h, w, g)
        cond = backend.condition(emb, tags, noise)
        with phases('rollout'):
            tr = rollout(backend, noise, cond, video_grid, audio_grid)
        tr.meta.update(data_seconds=data_seconds, height=h, width=w)
        return tr

    profile = None
    if a.profile and rank == 0:
        p_start, p_stop = (int(x) for x in a.profile.split(':'))
    else:
        p_start = p_stop = -1

    cached = {}
    for iteration in range(start, iterations):
        if iteration == p_start:
            profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profile.__enter__()
        tick = time.monotonic()
        role = update_kind(iteration, cfg['critic_steps'])
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)
        if cfg.get('optimizer_state_offload'):
            # Keep only the updating role's AdamW moments on the GPU (saves one role's
            # two moment copies, about 5.5 GB at rank 128).
            for other, opt in opts.items():
                move_moments(opt, 'cpu' if other != role else device)
        # The student's exit step is shared by every sample and rank of an update.
        index = int(torch.randint(cfg['nfe'], (1,), generator=torch.Generator().manual_seed(
            stream_seed(cfg['seed'], 'exit', iteration))))
        totals, data_seconds, shapes = {}, 0.0, []
        for micro, slot in enumerate(slots_of_rank):
            if role == 'student':
                tr = cached.pop(slot, None)
                if tr is None:  # first update after a resume: rebuild the previous critic batch
                    tr = trajectory_for(iteration - 1, slot)
            else:
                tr = trajectory_for(iteration, slot)
                data_seconds += tr.meta['data_seconds']
            shapes.append((tr.meta['height'], tr.meta['width']))
            g = torch.Generator().manual_seed(stream_seed(cfg['seed'], 'draws', iteration, slot))
            with phases('forward_backward'):
                if role == 'student':
                    ratio = float(torch.rand((), generator=g))
                    loss, stats = student_step_loss(
                        backend, tr, index, ratio, projected=projected, weights=weights,
                        clamp_max=cfg['loss_clamp'], normalizer_floor=cfg['normalizer_floor'],
                        eps=cfg['projection_eps'])
                else:
                    sigmas = tuple(s.to(device) for s in critic_times(
                        g, 'cpu', cfg.get('critic_video_threshold', 0.95), cfg.get('critic_audio_floor', 0.85)))
                    ng = torch.Generator(device=device).manual_seed(stream_seed(cfg['seed'], 'renoise', iteration, slot))
                    fresh = tuple(torch.randn(x.shape, device=device, dtype=torch.float32, generator=ng)
                                  for x in tr.states[-1])
                    loss, stats = critic_step_loss(backend, tr, sigmas, fresh, weights, cfg['loss_clamp'])
                    if update_kind(iteration + 1, cfg['critic_steps']) == 'student':
                        cached[slot] = tr
                (loss / accum).backward()
            totals['loss'] = totals.get('loss', 0.0) + loss.detach() / accum
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + (v / accum if torch.is_tensor(v) else v)
        with phases('reduce'):
            reduce_gradients(params[role])
        with phases('optimizer'):
            grad_norm = torch.nn.utils.clip_grad_norm_(params[role], cfg['gradient_clip'] or float('inf'))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'Non-finite {role} gradient norm at iteration {iteration}')
            opts[role].step()
        torch.cuda.synchronize()
        log = {'iteration': iteration + 1, 'role': role, 'index': index if role == 'student' else None,
               'grad_norm': float(grad_norm), 'seconds': round(time.monotonic() - tick, 3),
               'data_seconds': round(data_seconds, 4), 'phases': phases.collect(),
               'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2), 'shapes': shapes}
        log.update({k: (float(v) if torch.is_tensor(v) else v) for k, v in totals.items()})
        if not all(torch.isfinite(torch.tensor(v)) for k, v in log.items() if isinstance(v, float)):
            raise FloatingPointError(f'Non-finite metric at iteration {iteration}: {log}')
        if rank == 0:
            print(json.dumps(log), flush=True)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(log) + '\n')
        if profile is not None and iteration + 1 == p_stop:
            profile.__exit__(None, None, None)
            trace = out / 'profile' / f'rank0_iter{p_start}-{p_stop}.json'
            trace.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(trace))
            print(json.dumps({'event': 'profile', 'trace': str(trace), **gpu_busy_summary(trace)}), flush=True)
            profile = None
        done = iteration + 1
        state = None
        if cfg.get('save_every') and (done % cfg['save_every'] == 0 or done == iterations):
            state = {'next_iteration': done, 'adapters': adapter_state(model, device_tensors=True),
                     'optimizers': {r: o.state_dict() for r, o in opts.items()}, 'config': cfg,
                     'world_size': world, 'cache_signature': cache_signature,
                     'source_snapshot_id': snapshot_id}
            checkpointer.save_rolling(state, done)
        if (cfg.get('milestone_every') and done % cfg['milestone_every'] == 0) or done == iterations:
            checkpointer.save_lora(adapter_state(model, device_tensors=True), done,
                                   {'lora_rank': cfg['lora_rank'], 'lora_alpha': cfg['lora_alpha'],
                                    'nfe': cfg['nfe'], 'cache_signature': cache_signature,
                                    'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()})
        if state is not None and rank == 0 and checkpointer.last_write_seconds is not None:
            print(json.dumps({'event': 'checkpoint', 'iteration': done,
                              'previous_write_seconds': round(checkpointer.last_write_seconds, 1)}), flush=True)
    checkpointer.wait()
    prefetch.close()
    if rank == 0:
        (out / 'COMPLETE').write_text(json.dumps({'iterations': iterations}) + '\n')
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
