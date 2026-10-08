"""Download the pinned MiniMax-H3 FL2VA weights and the prompt-feature cache from Hugging Face.

  python scripts/download_assets.py --model-root /data/MiniMax-H3 --cache-root /data/pdmd-cache [--eval]

Training needs FL2VA/transformer and the cache (~66 GB + ~290 GB). --eval adds
the text encoder, processor and VAEs needed to encode new prompts and decode
samples (~78 GB more). Each cache file is checked against SHA256SUMS.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REPO, MODEL_REV = 'MiniMaxAI/MiniMax-H3', '42ed227ee7df40d41602854ae760620d6eb651fe'
CACHE_REPO, CACHE_REV = 'pdmd2026/rcm-vidprom-h3-qwenvl-cache', '5abcdbbe08ae594da1fe07128fac80e1a3c1ae0a'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', required=True)
    p.add_argument('--cache-root')
    p.add_argument('--eval', action='store_true', help='also fetch text encoder, processor and VAEs')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--skip-verify', action='store_true')
    a = p.parse_args()
    patterns = ['FL2VA/transformer/*']
    if a.eval:
        patterns += ['FL2VA/text_encoder/*', 'FL2VA/processor/*', 'FL2VA/tokenizer/*',
                     'FL2VA/video_vae/*', 'FL2VA/video_vae/source/*', 'FL2VA/audio_vae/*', 'FL2VA/model_index.json']
    snapshot_download(MODEL_REPO, revision=MODEL_REV, allow_patterns=patterns,
                      local_dir=a.model_root, max_workers=a.workers)
    if a.cache_root:
        snapshot_download(CACHE_REPO, repo_type='dataset', revision=CACHE_REV,
                          local_dir=a.cache_root, max_workers=a.workers)
        if not a.skip_verify:
            root = Path(a.cache_root)
            sums = json.loads((root / 'SHA256SUMS.json').read_text())
            bad = [rel for rel, v in sums.items()
                   if (root / rel).stat().st_size != v['size'] or sha256(root / rel) != v['sha256']]
            if bad:
                raise SystemExit(f'{len(bad)} cache files fail verification, e.g. {bad[:3]}')
            print(f'cache verified: {len(sums)} files')


if __name__ == '__main__':
    main()
