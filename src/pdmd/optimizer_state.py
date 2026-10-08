"""Move AdamW moments without changing parameters, gradients, or step counters."""
import torch


def move_moments(optimizer, device):
    if any(g.get('capturable', False) or g.get('fused', False) for g in optimizer.param_groups):
        raise ValueError('Moment offload requires ordinary non-capturable AdamW')
    for state in optimizer.state.values():
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state:
                state[name] = state[name].to(device)
