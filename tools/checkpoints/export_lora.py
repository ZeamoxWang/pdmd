"""Export our student adapters to H3 Diffusers naming, without changing any base."""
import argparse
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pdmd.lora_conversion import convert  # noqa: E402


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
        'method':'Projected DMD (pdmd)',
        'iteration':str(ckpt['next_iteration'])})


if __name__=='__main__':main()
