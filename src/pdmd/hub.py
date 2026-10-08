"""Resolve the base model and the prompt-feature cache: a local path, or a pinned Hugging Face repo.

Pass a local directory to use files already on disk, or `repo_id@revision` to
download (once per node) into the Hugging Face cache. Only the files training
needs are fetched: FL2VA/transformer for the model, the whole feature cache for
the data.
"""
import os
from pathlib import Path

MODEL = 'MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe'
CACHE = 'pdmd2026/rcm-vidprom-h3-qwenvl-cache@5abcdbbe08ae594da1fe07128fac80e1a3c1ae0a'


def _split(spec):
    repo, _, rev = spec.partition('@')
    return repo, (rev or None)


def _download(spec, repo_type, patterns):
    from huggingface_hub import snapshot_download
    repo, rev = _split(spec)
    return Path(snapshot_download(repo, repo_type=repo_type, revision=rev, allow_patterns=patterns,
                                  max_workers=int(os.environ.get('PDMD_HF_WORKERS', '16'))))


def _on_node_leader(fn):
    """Run fn on local rank 0 of each node first; the other local ranks then reuse its result."""
    import torch.distributed as dist
    local = int(os.environ.get('LOCAL_RANK', 0))
    if not (dist.is_available() and dist.is_initialized()):
        return fn()
    if local == 0:
        out = fn()
    dist.barrier()
    if local != 0:
        out = fn()  # files are present now: snapshot_download only verifies them
    return out


def resolve_model(spec=MODEL):
    """Directory that contains FL2VA/transformer."""
    if Path(spec).exists():
        return Path(spec)
    return _on_node_leader(lambda: _download(spec, 'model', ['FL2VA/transformer/*']))


def resolve_cache(spec=CACHE):
    """(cache directory with rank-*.jsonl, path of its cache-audit.json or None)."""
    root = Path(spec) if Path(spec).exists() else _on_node_leader(lambda: _download(spec, 'dataset', None))
    cache = root / 'cache' if (root / 'cache').is_dir() else root
    audit = root / 'cache-audit.json'
    return cache, (audit if audit.exists() else None)
