"""Audit every cached feature against its exact original prompt and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open


def verify(cache, prompts, expected_count):
    cache, prompts = Path(cache), Path(prompts)
    source_hash = hashlib.sha256(prompts.read_bytes()).hexdigest()
    source = [json.loads(line) for line in prompts.read_text().splitlines()]
    if len(source) != expected_count:
        raise ValueError(f'Original prompt count {len(source)} != {expected_count}')
    records = []
    worlds = set()
    for manifest in sorted(cache.glob('rank-*.jsonl')):
        rank = int(manifest.stem.split('-')[1])
        complete = json.loads((cache/f'COMPLETE-rank-{rank:04d}.json').read_text())
        if complete['source_sha256'] != source_hash or complete['limit'] not in (None, expected_count):
            raise ValueError(f'Incomplete or wrong source: {manifest}')
        worlds.add(complete['world_size'])
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row['world_size'] != complete['world_size'] or row['source_row'] % row['world_size'] != rank:
                raise ValueError('Cache worker assignment mismatch')
            records.append(row)
    if len(worlds) != 1:
        raise ValueError('Missing or inconsistent cache worker count')
    world = worlds.pop()
    if {p.name for p in cache.glob('rank-*.jsonl')} != {f'rank-{r:04d}.jsonl' for r in range(world)}:
        raise ValueError('Missing worker manifest')
    if len(records) != expected_count or {r['source_row'] for r in records} != set(range(expected_count)):
        raise ValueError('Missing or duplicate cache rows')
    shards = {}
    for row in records:
        original = source[row['source_row']]
        if (row['source_sha256'] != source_hash or row['id'] != original['id'] or
            row['prompt_sha256'] != hashlib.sha256(original['prompt'].encode()).hexdigest()):
            raise ValueError(f"Original prompt identity mismatch: {row['source_row']}")
        name = row['shard']
        if Path(name).name != name:
            raise ValueError('Shard path must be a filename')
        shards.setdefault(name, []).append(row)
    provenance = None
    for name, rows in shards.items():
        with safe_open(cache/name, framework='pt', device='cpu') as f:
            metadata = f.metadata()
            if (metadata['source_sha256'] != source_hash or metadata['encoder_layer'] != '50' or
                metadata['final_norm'] != 'identity' or metadata['prompt_rewrite'] != 'false' or
                metadata['base_revision'] != '42ed227ee7df40d41602854ae760620d6eb651fe' or
                metadata['diffsynth_revision'] != '974cfa37f27ac55eba3b6d10efa21f876900572d'):
                raise ValueError(f'Incorrect encoder provenance: {name}')
            if provenance is not None and metadata != provenance:
                raise ValueError('Mixed encoder provenance')
            provenance = metadata
            for row in rows:
                hidden, tags = f.get_tensor(row['key']), f.get_tensor(row['key']+'_tags')
                if hidden.shape != (row['tokens'], 5120) or tags.shape != (row['tokens'],):
                    raise ValueError('Embedding/tag shape mismatch')
                if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all():
                    raise ValueError('Invalid cached embedding values')
                if tags.dtype != torch.int64 or not (tags == 1).all():
                    raise ValueError('Invalid text-only token tags')
    records.sort(key=lambda row: row['source_row'])
    return {'samples': len(records), 'shards': len(shards), 'workers': world,
            'cache_signature': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            'source_sha256': source_hash, 'max_tokens': max(r['tokens'] for r in records),
            'total_tokens': sum(r['tokens'] for r in records), 'provenance': provenance}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--expected-count', required=True, type=int)
    p.add_argument('--report', help='Atomically publish the successful audit receipt')
    a = p.parse_args()
    report = json.dumps(verify(a.cache, a.prompts, a.expected_count), indent=2)+'\n'
    if a.report:
        target = Path(a.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(report)
        temporary.replace(target)
    print(report, end='')


if __name__ == '__main__':
    main()
