"""Back up the audited cache to the user-authorized public dataset only.

HF_TOKEN is read from the environment, never from a CLI argument. Staging uses
hard links on the same filesystem; original cache files are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = 'pdmd2026/rcm-vidprom-h3-qwenvl-cache'


def hashes(path):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            blob.update(chunk)
    return {'size': path.stat().st_size, 'sha256': sha.hexdigest(), 'blob_id': blob.hexdigest()}


def destination_info(api):
    info = api.repo_info(REPO, repo_type='dataset', files_metadata=True)
    if info.private is not False:
        raise RuntimeError('Destination visibility differs from the user-authorized public setting')
    return info


def matches(remote, expected):
    if remote is None or remote.size != expected['size']:
        return False
    lfs = remote.lfs
    if lfs:
        sha = lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256
        return sha == expected['sha256']
    return remote.blob_id == expected['blob_id']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--prompts-dir', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--staging', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    destination_info(api)
    cache, prompts, stage = Path(a.cache), Path(a.prompts_dir), Path(a.staging)
    audit = json.loads(Path(a.audit).read_text())
    rows = []
    manifests = sorted(cache.glob('rank-*.jsonl'))
    for path in manifests:
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    rows.sort(key=lambda row: row['source_row'])
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if audit['samples'] != 248221 or len(rows) != 248221 or audit['cache_signature'] != signature:
        raise ValueError('Missing, stale or incomplete full-cache audit')
    if hashes(prompts/'prompts.jsonl')['sha256'] != audit['source_sha256']:
        raise ValueError('Prompt source does not match cache audit')
    files = {f'cache/{name}': cache/name for name in {r['shard'] for r in rows}}
    for path in manifests:
        files['cache/'+path.name] = path
    for rank in range(audit['workers']):
        path = cache/f'COMPLETE-rank-{rank:04d}.json'
        if json.loads(path.read_text())['source_sha256'] != audit['source_sha256']:
            raise ValueError('Worker completion source mismatch')
        files['cache/'+path.name] = path
    for name in ('source.txt','prompts.jsonl','provenance.json','token-statistics.json'):
        files['prompts/'+name] = prompts/name
    files['cache-audit.json'] = Path(a.audit)
    stage.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for i, (name, source) in enumerate(sorted(files.items())):
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
        if not os.path.samefile(source, target):
            raise ValueError('Staging path is not linked to the audited source: '+name)
        inventory[name] = hashes(target)
        if i % 100 == 0:
            print(json.dumps({'event':'hashing','files':i+1,'total':len(files)}), flush=True)
    (stage/'SHA256SUMS.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (stage/'README.md').write_text('''---
license: other
license_name: upstream-terms-apply
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
---
# rCM prompt list encoded with MiniMax-H3 QwenVL

Public cache for Projected DMD training on MiniMax-H3.
248,221 published rCM-linked prompts, preserved verbatim without additional
rewriting. BF16 token features have dimension 5120; integer token tags are 1.
H3 FL2VA encoder uses 50 layers and identity final normalization. Exact model,
encoder implementation and source versions are recorded in the cache metadata,
`prompts/provenance.json`, and `cache-audit.json`.

This is not a model checkpoint. `SHA256SUMS.json` records the file hashes.
Source text and model-derived artifacts remain subject to their upstream terms:
[VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM),
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

`BACKUP_COMPLETE.json` is published only after all data files pass remote
inventory/hash checks. An interrupted upload without it is incomplete.
''')
    for name in ('SHA256SUMS.json','README.md'):
        inventory[name] = hashes(stage/name)
    info = destination_info(api)
    remote = {f.rfilename: f for f in info.siblings}
    pending = [name for name in sorted(inventory) if not matches(remote.get(name), inventory[name])]
    for offset in range(0, len(pending), 64):
        destination_info(api)
        batch = pending[offset:offset+64]
        api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=stage,
                          allow_patterns=batch, commit_message='Back up audited H3 QwenVL cache shards')
        print(json.dumps({'event':'uploaded','files':min(offset+64,len(pending)),
                          'total':len(pending)}), flush=True)
    info = destination_info(api)
    remote = {f.rfilename: f for f in info.siblings}
    failed = [name for name, meta in inventory.items() if not matches(remote.get(name), meta)]
    if failed:
        raise RuntimeError('Remote hash/size verification failed: '+str(failed[:10]))
    receipt = {'repo_id': REPO, 'private': False, 'verified_data_revision': info.sha,
               'files': len(inventory), 'bytes': sum(v['size'] for v in inventory.values()),
               'cache_signature': signature, 'samples': len(rows), 'source_copy_retained': True}
    completion = stage/'BACKUP_COMPLETE.json'
    completion.write_text(json.dumps(receipt, indent=2)+'\n')
    api.upload_file(repo_id=REPO, repo_type='dataset', path_or_fileobj=completion,
                    path_in_repo=completion.name, commit_message='Record verified cache backup')
    final_info = destination_info(api)
    receipt['revision'] = final_info.sha
    target = Path(a.receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt, indent=2)+'\n')
    tmp.replace(target)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
