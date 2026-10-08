"""Export our student adapters to H3 Diffusers naming, without changing any base."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file


def convert(state):
    output={}
    for key,value in state.items():
        if not key.endswith('.student'):continue
        name,kind,_ = key.rsplit('.',2)
        name=name.replace('token_refiner.blocks.','token_refiner.refiner_blocks.')
        if name.startswith('blocks.'):name='transformer_blocks.'+name[len('blocks.'):]
        if name.endswith('.qkv_proj'):
            for qkv,part in zip('qkv',value.chunk(3,dim=0)):
                output['transformer.'+name.removesuffix('.qkv_proj')+f'.to_{qkv}.{kind}.weight']=part.contiguous()
        else:
            if name.endswith('.attn.out_proj'):
                name=name.removesuffix('.out_proj')+'.to_out.0'
            elif name.endswith('.mlp.fc1'):
                name=name.removesuffix('.mlp.fc1')+'.ff.net.0.proj'
                if kind=='lora_B':
                    gate,up=value.chunk(2,dim=0)
                    value=torch.cat((up,gate),dim=0)
            elif name.endswith('.mlp.fc2'):
                name=name.removesuffix('.mlp.fc2')+'.ff.net.2'
            else:raise ValueError(name)
            output['transformer.'+name+f'.{kind}.weight']=value.contiguous()
    if not output:raise ValueError('No student adapters')
    return output


def from_diffusers(tensors, role='student'):
    """Inverse of convert(): H3 Diffusers LoRA naming -> this repo's adapter keys.

    Accepts the released PDMD LoRAs (e.g. pdmd2026/pdmd_4NFE_lora). Per-projection
    to_q/to_k/to_v pairs are stacked into the fused-QKV adapter, and the FFN
    input projection's B halves are swapped back from (up, gate) to (gate, up).
    """
    groups = {}
    for key, value in tensors.items():
        if not key.startswith('transformer.') or not key.endswith('.weight'):
            raise ValueError(f'Not an H3 Diffusers LoRA key: {key}')
        name, kind = key[len('transformer.'):-len('.weight')].rsplit('.', 1)
        groups[(name, kind)] = value
    out = {}
    for (name, kind), value in groups.items():
        base = name.replace('token_refiner.refiner_blocks.', 'token_refiner.blocks.')
        if base.startswith('transformer_blocks.'):
            base = 'blocks.' + base[len('transformer_blocks.'):]
        if base.endswith(('.attn.to_k', '.attn.to_v')):
            continue
        if base.endswith('.attn.to_q'):
            stem = name.removesuffix('.to_q')
            parts = [groups[(f'{stem}.to_{x}', kind)] for x in 'qkv']
            out[base.removesuffix('.to_q') + f'.qkv_proj.{kind}.{role}'] = torch.cat(parts, 0).contiguous()
        elif base.endswith('.attn.to_out.0'):
            out[base.removesuffix('.to_out.0') + f'.out_proj.{kind}.{role}'] = value.contiguous()
        elif base.endswith('.ff.net.0.proj'):
            if kind == 'lora_B':
                up, gate = value.chunk(2, dim=0)
                value = torch.cat((gate, up), dim=0)
            out[base.removesuffix('.ff.net.0.proj') + f'.mlp.fc1.{kind}.{role}'] = value.contiguous()
        elif base.endswith('.ff.net.2'):
            out[base.removesuffix('.ff.net.2') + f'.mlp.fc2.{kind}.{role}'] = value.contiguous()
        else:
            raise ValueError(f'Unexpected LoRA module {name}')
    return out


def _iteration_from_dir(name):
    """'iter_002500' -> '2500'; any other directory name gives 'unknown' instead of leaking the path."""
    import re
    m = re.fullmatch(r'iter_(\d+)', name)
    return str(int(m.group(1))) if m else 'unknown'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    if a.checkpoint.endswith('.safetensors'):
        # A milestone student_lora.safetensors: rank/alpha/iteration live in its metadata.
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(a.checkpoint,'pt') as f:
            meta=f.metadata() or {}
        ckpt={'adapters':load_file(a.checkpoint),
              'config':{'lora_rank':int(meta.get('lora_rank',128)),'lora_alpha':int(meta.get('lora_alpha',128))},
              'next_iteration':meta.get('iteration',_iteration_from_dir(Path(a.checkpoint).parent.name))}
    else:
        ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    tensors=convert(ckpt['adapters'])
    config=ckpt['config']
    out.parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,out,metadata={'format':'pt','role':'student',
        'lora_rank':str(config['lora_rank']),'lora_alpha':str(config['lora_alpha']),
        'lora_scale':str(config['lora_alpha']/config['lora_rank']),
        'fuse':'W_base += lora_scale * (lora_B @ lora_A)',
        'method':'Projected DMD (pdmd_training)',
        'iteration':str(ckpt['next_iteration'])})


if __name__=='__main__':main()
