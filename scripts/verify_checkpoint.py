"""CPU-only audit of our saved adapters and AdamW states, without loading DiT."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--scratch', help='Stage checkpoint sequentially on local disk before mmap checks')
    a = p.parse_args()
    run = Path(a.run)
    path = run / f'checkpoint-{a.iteration:06d}.pt'
    load_path = path
    staged_digest = None
    if a.scratch:
        scratch = Path(a.scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        load_path = scratch / path.name
        assert load_path.resolve() != path.resolve()
        digest = hashlib.sha256()
        copied = 0
        next_log = 1024 ** 3
        before = path.stat()
        with path.open('rb') as src, load_path.open('xb') as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                dst.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if copied >= next_log:
                    print(json.dumps({'event': 'checkpoint_staging', 'bytes': copied}), flush=True)
                    next_log += 1024 ** 3
        after = path.stat()
        assert copied == before.st_size == after.st_size
        assert before.st_mtime_ns == after.st_mtime_ns
        staged_digest = digest.hexdigest()
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=True, mmap=True)
    config = json.loads((run / 'run.json').read_text())
    assert checkpoint['config'] == config
    assert checkpoint['next_iteration'] == a.iteration
    assert checkpoint['source_snapshot_id']
    assert checkpoint['cache_signature']
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert [m['iteration'] for m in metrics] == list(range(1, a.iteration + 1))
    assert all(torch.isfinite(torch.tensor([m['loss'], m['grad_norm']])).all() for m in metrics)
    adapters = checkpoint['adapters']
    roles = {}
    tensors_checked = 0
    for role in ('student', 'critic'):
        params = [t for name, t in adapters.items() if name.endswith('.' + role)]
        assert params and all(t.dtype == torch.float32 and t.ndim == 2 for t in params)
        for t in params:
            assert torch.isfinite(t).all()
            tensors_checked += 1
        # Full H3 rank128 coverage from independently counted model targets.
        if config['lora_rank'] == 128:
            assert len(params) == 416
            assert sum(t.numel() for t in params) == 691798016
        opt = checkpoint['optimizers'][role]
        ids = [pid for group in opt['param_groups'] for pid in group['params']]
        assert len(ids) == len(params) == len(set(ids))
        assert set(ids) == set(opt['state'])
        steps = sum(m['role'] == role for m in metrics)
        for group in opt['param_groups']:
            assert group['lr'] == config[role + '_lr']
            assert tuple(group['betas']) == tuple(config['adam_betas'])
            assert group['weight_decay'] == config['weight_decay']
        for pid, param in zip(ids, params):
            state = opt['state'][pid]
            assert int(state['step'].item()) == steps
            for key in ('exp_avg', 'exp_avg_sq'):
                t = state[key]
                assert t.shape == param.shape and t.dtype == param.dtype
                assert torch.isfinite(t).all()
                if key == 'exp_avg_sq':
                    assert (t >= 0).all()
                tensors_checked += 1
        roles[role] = {'parameters': sum(t.numel() for t in params), 'tensors': len(params), 'optimizer_steps': steps}
        print(json.dumps({'event': 'role_verified', 'role': role, **roles[role]}), flush=True)
    assert len(adapters) == sum(r['tensors'] for r in roles.values())
    if staged_digest is None:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        staged_digest = digest.hexdigest()
    report = {'checkpoint': str(path), 'bytes': path.stat().st_size, 'sha256': staged_digest,
              'next_iteration': a.iteration, 'world_size': checkpoint['world_size'],
              'source_snapshot_id': checkpoint['source_snapshot_id'], 'cache_signature': checkpoint['cache_signature'],
              'roles': roles, 'finite_tensors_checked': tensors_checked,
              'scope': 'deserialization, coverage, shapes, finite values, moments, step counts and metadata; not a GPU resume test'}
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
