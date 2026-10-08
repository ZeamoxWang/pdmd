"""Strict, shard-streamed loading; never instantiate full random CPU weights."""
import json
from pathlib import Path
import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open


def load_indexed(model, directory, dtype=torch.bfloat16, device='cpu', allow_extra=None):
    directory = Path(directory)
    index = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    expected = dict(model.named_parameters())
    missing = set(expected) - set(index)
    extra = [k for k in set(index)-set(expected) if not (allow_extra and allow_extra(k))]
    if missing or extra:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)[:12]}, extra={sorted(extra)[:12]}')
    by_shard = {}
    for key in expected:
        by_shard.setdefault(index[key], []).append(key)
    for shard, keys in sorted(by_shard.items()):
        with safe_open(directory/shard, framework='pt', device='cpu') as f:
            for key in keys:
                value = f.get_tensor(key)
                if value.shape != expected[key].shape:
                    raise ValueError(f'Wrong shape: {key}')
                set_module_tensor_to_device(model, key, device, value=value, dtype=dtype)
        print(f'loaded {shard}', flush=True)
    # Buffers created outside meta by accelerate.init_empty_weights stay real.
    for name, buffer in model.named_buffers():
        if buffer.is_meta:
            raise ValueError(f'Unmaterialized buffer {name}')
    return model


def load_dit(directory):
    from accelerate import init_empty_weights
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
    config = json.loads((Path(directory)/'config.json').read_text())
    with init_empty_weights():
        model = MiniMaxH3DiT(**config)
    return load_indexed(model, directory).requires_grad_(False)


def load_encoder(directory, device='cuda'):
    from accelerate import init_empty_weights
    from diffsynth.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
    def omitted(k):
        prefix = 'model.language_model.layers.'
        return (k == 'lm_head.weight' or k.startswith('model.language_model.norm.') or
                (k.startswith(prefix) and int(k[len(prefix):].split('.')[0]) >= 50))
    with init_empty_weights():
        model = MiniMaxH3TextEncoder(num_retained_layers=50)
    model = load_indexed(model, directory, device=device, allow_extra=omitted)
    return model.to(device).eval().requires_grad_(False)
