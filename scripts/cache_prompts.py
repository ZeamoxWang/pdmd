"""Dump H3 layer-50 Qwen3-VL features. Requires an explicit original prompt file."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoTokenizer
from pdmd_training.loading import load_encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--prompts', required=True, help='JSONL rows: id, prompt; text preserved verbatim')
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--shard-size', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    root, out = Path(a.base)/'FL2VA', Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    source_hash = hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest()
    provenance = {'source_sha256':source_hash,'encoder_layer':'50',
                  'final_norm':'identity','prompt_rewrite':'false',
                  'base_revision':'42ed227ee7df40d41602854ae760620d6eb651fe',
                  'diffsynth_revision':'974cfa37f27ac55eba3b6d10efa21f876900572d',
                  'encoder_index_sha256':hashlib.sha256((root/'text_encoder/model.safetensors.index.json').read_bytes()).hexdigest(),
                  'tokenizer_sha256':hashlib.sha256((root/'processor/tokenizer.json').read_bytes()).hexdigest()}
    manifest = out/f'rank-{rank:04d}.jsonl'
    if manifest.exists() and not a.resume:
        raise FileExistsError(f'Use a new output directory; manifest exists: {manifest}')
    completed = set()
    if manifest.exists():
        previous = [json.loads(line) for line in manifest.read_text().splitlines()]
        for row in previous:
            if row['source_sha256'] != source_hash or row['world_size'] != world:
                raise ValueError('Resume source or worker count changed')
            if row['source_row'] in completed:
                raise ValueError('Duplicate cache row')
            completed.add(row['source_row'])
        by_shard = {}
        for row in previous:
            by_shard.setdefault(row['shard'],[]).append(row)
        for name, shard_rows in by_shard.items():
            with safe_open(out/name,framework='pt',device='cpu') as f:
                if f.metadata() != provenance:
                    raise ValueError(f'Cache provenance mismatch: {name}')
                for row in shard_rows:
                    if f.get_slice(row['key']).get_shape() != [row['tokens'],5120]:
                        raise ValueError(f'Corrupt cached shape: {name}')
    load_started = time.monotonic()
    encoder = load_encoder(root/'text_encoder', device)
    torch.cuda.synchronize()
    print(json.dumps({'event':'encoder_loaded','seconds':time.monotonic()-load_started}),flush=True)
    tokenizer = AutoTokenizer.from_pretrained(root/'processor', local_files_only=True)
    existing = list(out.glob(f'rank-{rank:04d}-shard-*.safetensors'))
    # A shard whose manifest append was interrupted is left as an unreferenced
    # artifact; choose a fresh name rather than overwrite it.
    shard = max((int(x.stem.rsplit('-',1)[1]) for x in existing),default=-1)+1
    tensors, rows = {}, []
    encode_started, encoded = time.monotonic(), 0
    def flush():
        nonlocal tensors, rows, shard, encoded
        if not rows:
            return
        name = f'rank-{rank:04d}-shard-{shard:06d}.safetensors'
        tmp = out/(name+'.tmp')
        save_file(tensors, tmp, metadata=provenance)
        # Publish the manifest only after the shard's data is durable. A node
        # failure must not leave durable rows pointing to unwritten file pages.
        with tmp.open('rb') as f:
            os.fsync(f.fileno())
        tmp.rename(out/name)
        directory_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        with manifest.open('a') as f:
            for row in rows:
                row['shard'] = name
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        encoded += len(rows)
        elapsed = time.monotonic()-encode_started
        print(json.dumps({'rank':rank,'shard':shard,'samples':len(rows),
                          'encoded_this_run':encoded,'encoding_seconds':elapsed,
                          'samples_per_second':encoded/elapsed,
                          'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}), flush=True)
        tensors, rows, shard = {}, [], shard+1
    with open(a.prompts) as f, torch.inference_mode():
        for i, line in enumerate(f):
            if a.limit is not None and i >= a.limit:
                break
            if i % world != rank:
                continue
            if i in completed:
                continue
            row = json.loads(line)
            prompt = row['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Invalid prompt at source row {i}')
            ids, tags = presentation_t2va(tokenizer, prompt)
            ids = ids[None].to(device)
            hidden = encoder(input_ids=ids, attention_mask=torch.ones_like(ids))
            if hidden.shape != (ids.shape[1], 5120) or not torch.isfinite(hidden).all():
                raise ValueError(f'Invalid encoder output at row {i}')
            key = f'sample_{i:08d}'
            tensors[key] = hidden.to('cpu', torch.bfloat16).contiguous()
            tensors[key+'_tags'] = tags.contiguous()
            rows.append({'id':row['id'], 'source_row':i, 'key':key, 'tokens':len(tags),
                         'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                         'source_sha256':source_hash,'world_size':world})
            if len(rows) == a.shard_size:
                flush()
    flush()
    (out/f'COMPLETE-rank-{rank:04d}.json').write_text(json.dumps({
        'source_sha256':source_hash,'world_size':world,'limit':a.limit})+'\n')


if __name__ == '__main__':
    main()
