"""Two independent LoRAs on one frozen H3 base; no upstream source patches."""
import math
import torch
from torch import nn


class DualLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.active = 'teacher'
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for role in ('student', 'critic'):
            # fp32 optimizer/master parameters, cast activations at the boundary.
            self.lora_A[role] = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
            self.lora_B[role] = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A[role], a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        if self.active != 'teacher':
            a, b = self.lora_A[self.active], self.lora_B[self.active]
            # Casting weights preserves their fp32 master copy and gradients.
            delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a.to(x.dtype)), b.to(x.dtype))
            result = result + delta * self.scale
        return result


class SplitQKVDualLoRA(DualLoRA):
    """Independent rank-r Q/K/V adapters inside DiffSynth's fused QKV linear.

    Public PDMD weights have separate to_q/to_k/to_v rank-128 pairs. A single
    rank-128 fused QKV adapter would share A and change the parameterization.
    """
    def __init__(self, base, rank, alpha, head_dim):
        nn.Module.__init__(self)
        self.base = base.requires_grad_(False)
        self.scale, self.active = alpha/rank, 'teacher'
        self.rank, self.head_dim = rank, head_dim
        self.inner = base.out_features//3
        if base.out_features % 3 or self.inner % head_dim:
            raise ValueError('Invalid fused QKV dimensions')
        self.lora_A, self.lora_B = nn.ParameterDict(), nn.ParameterDict()
        for role in ('student','critic'):
            self.lora_A[role] = nn.Parameter(torch.empty(3*rank,base.in_features,device=base.weight.device))
            self.lora_B[role] = nn.Parameter(torch.zeros(3*self.inner,rank,device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A[role],a=math.sqrt(5))

    def forward(self,x):
        result = self.base(x)
        if self.active != 'teacher':
            a,b = self.lora_A[self.active],self.lora_B[self.active]
            low = torch.nn.functional.linear(x,a.to(x.dtype)).reshape(*x.shape[:-1],3,self.rank)
            delta = torch.einsum('...gr,gor->...go',low,b.to(x.dtype).reshape(3,self.inner,self.rank))
            # Original Q,K,V output chunks -> head-interleaved fused projection.
            delta = delta.reshape(*x.shape[:-1],3,self.inner//self.head_dim,self.head_dim)
            delta = delta.transpose(-3,-2).flatten(-3)
            result = result+delta*self.scale
        return result


def install_adapters(model, rank=128, alpha=128):
    targets = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
    matched = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(targets):
            parent, _, leaf = name.rpartition('.')
            owner = model.get_submodule(parent)
            wrapped = (SplitQKVDualLoRA(module,rank,alpha,owner.head_dim) if leaf=='qkv_proj'
                       else DualLoRA(module,rank,alpha))
            setattr(owner, leaf, wrapped)
            matched.append(name)
    if not matched:
        raise ValueError('No H3 adapter targets found')
    return matched


def set_role(model, role):
    if role not in ('teacher', 'student', 'critic'):
        raise ValueError(role)
    for module in model.modules():
        if isinstance(module, DualLoRA):
            module.active = role


def role_parameters(model, role):
    return [p for n, p in model.named_parameters() if f'lora_A.{role}' in n or f'lora_B.{role}' in n]


@torch.no_grad()
def merge_role(model, role='student'):
    """Fold one role's adapters into the frozen base weights and unwrap the modules.

    Used for sampling: the merged model has the base's memory footprint and no
    adapter overhead. For the fused QKV projection the three per-projection
    deltas are interleaved per head exactly as SplitQKVDualLoRA.forward does.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, DualLoRA):
            continue
        a = module.lora_A[role].float()
        b = module.lora_B[role].float()
        if isinstance(module, SplitQKVDualLoRA):
            r, inner, hd = module.rank, module.inner, module.head_dim
            blocks = torch.stack([b[g*inner:(g+1)*inner] @ a[g*r:(g+1)*r] for g in range(3)])
            delta = blocks.reshape(3, inner // hd, hd, -1).transpose(0, 1).reshape(3 * inner, -1)
        else:
            delta = b @ a
        weight = module.base.weight
        weight.copy_((weight.float() + module.scale * delta.to(weight.device)).to(weight.dtype))
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, leaf, module.base)
        merged += 1
    return merged


@torch.no_grad()
def load_role(model, path, role='student'):
    """Load '<module>.lora_{A,B}.<role>' tensors (a milestone file) into installed adapters."""
    from safetensors.torch import load_file
    state = load_file(str(path))
    if all(k.startswith('transformer.') for k in state):
        # A released H3 Diffusers LoRA (e.g. pdmd2026/pdmd_4NFE_lora): convert names.
        from .lora_conversion import from_diffusers
        state = from_diffusers(state, role)
    used = set()
    for name, module in model.named_modules():
        if isinstance(module, DualLoRA):
            for kind in ('lora_A', 'lora_B'):
                key = f'{name}.{kind}.{role}'
                if key not in state:
                    raise KeyError(f'Missing adapter tensor {key}')
                p = getattr(module, kind)[role]
                if p.shape != state[key].shape:
                    raise ValueError(f'Adapter shape mismatch: {key}')
                p.copy_(state[key].to(p.device, p.dtype))
                used.add(key)
    extra = set(state) - used
    if extra:
        raise ValueError(f'Unexpected adapter tensors, e.g. {sorted(extra)[:3]}')
    return len(used)
