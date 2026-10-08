"""Download immutable public inputs to a local directory."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    repo = 'MiniMaxAI/MiniMax-H3'
    revision = '42ed227ee7df40d41602854ae760620d6eb651fe'
    patterns = ['FL2VA/processor/*', 'FL2VA/text_encoder/*', 'FL2VA/transformer/*']
    # No VAE, Ref2VA, distilled checkpoint, or write/upload to Hugging Face.
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns,
                             local_dir=root/'base', max_workers=2)
    provenance = {'model_id': repo, 'model_revision': revision, 'model_path': path,
                  'allow_patterns': patterns}
    dataset = 'WenhaoWang/VidProM'
    dataset_revision = HfApi().dataset_info(dataset).sha
    csv = hf_hub_download(dataset, 'VidProM_unique.csv', repo_type='dataset',
                          revision=dataset_revision, local_dir=root/'prompts_source')
    digest = hashlib.sha256()
    with open(csv, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    provenance['raw_vidprom'] = {'revision': dataset_revision, 'file': csv,
                                 'sha256': digest.hexdigest(),
                                 'status': 'raw original corpus; NOT the paper 248K subset'}
    (root/'assets.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
