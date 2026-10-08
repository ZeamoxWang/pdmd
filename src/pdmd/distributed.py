"""Shard the frozen base; replicate and explicitly reduce only the two LoRAs."""
import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from pdmd.adapters import DualLoRA


def shard_base(model, device, forward_prefetch=True):
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock
    # Ignored adapters retain fp32 master weights and normal optimizer state.
    # Only frozen bf16 base weights are flattened, avoiding mixed-dtype handles.
    ignored = []
    for module in model.modules():
        if isinstance(module, DualLoRA):
            module.lora_A.to(device)
            module.lora_B.to(device)
            ignored.extend(module.lora_A.parameters())
            ignored.extend(module.lora_B.parameters())
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={MiniMaxH3DiTBlock, MiniMaxH3TokenRefinerBlock})
    # Shard the immutable 66GB base WITHIN each node, so its all-gathers stay on
    # the fast intra-node links. Only adapter gradients cross nodes.
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE',dist.get_world_size()))
    if dist.get_world_size() % local_world:
        raise ValueError('Unequal local worker counts are unsupported')
    base_group = None
    for start in range(0,dist.get_world_size(),local_world):
        ranks = list(range(start,start+local_world))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            base_group = group
    # forward_prefetch issues block i+1's all-gather before block i computes,
    # which matters most in the four no-grad rollout passes per update; the
    # default only prefetches in backward.
    return FSDP(model, auto_wrap_policy=policy, ignored_states=ignored,
                device_id=device, use_orig_params=True, limit_all_gathers=True,
                forward_prefetch=forward_prefetch,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                sync_module_states=False, process_group=base_group)


def reduce_gradients(parameters, bucket_bytes=256 * 1024 * 1024):
    # Coalesce small LoRA tensors: hundreds of separate all-reduces make each
    # optimizer step pay hundreds of network round trips across nodes.
    world = dist.get_world_size() if dist.is_initialized() else 1
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for grad in bucket:
            grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
            offset += grad.numel()

    for p in parameters:
        # Explicit zero keeps collective ordering identical for unused targets.
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if world == 1:
            continue
        grad = p.grad
        nbytes = grad.numel() * grad.element_size()
        if bucket and (size + nbytes > bucket_bytes or
                       grad.dtype != bucket[0].dtype or grad.device != bucket[0].device):
            flush()
            bucket, size = [], 0
        bucket.append(grad)
        size += nbytes
    flush()


def adapter_state(model, device_tensors=False):
    """Adapter tensors keyed '<module>.lora_{A,B}.<role>', without FSDP wrapper names.

    device_tensors=True returns the live (detached) parameters so an
    asynchronous writer can copy them into its own pinned buffers; otherwise
    independent CPU clones are returned.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, DualLoRA):
            name = name.replace('_fsdp_wrapped_module.', '')
            for role in ('student', 'critic'):
                for kind in ('lora_A', 'lora_B'):
                    t = getattr(module, kind)[role].detach()
                    state[f'{name}.{kind}.{role}'] = t if device_tensors else t.cpu().clone()
    return state


def restore_adapters(model, state):
    used = set()
    for name, module in model.named_modules():
        if isinstance(module, DualLoRA):
            name = name.replace('_fsdp_wrapped_module.', '')
            for kind in ('lora_A', 'lora_B'):
                for role in ('student', 'critic'):
                    key = f'{name}.{kind}.{role}'
                    p = getattr(module,kind)[role]
                    if p.shape != state[key].shape:
                        raise ValueError(f'Adapter shape mismatch: {key}')
                    with torch.no_grad():
                        p.copy_(state[key])
                    used.add(key)
    if used != set(state):
        raise ValueError('Unexpected checkpoint adapter keys')
