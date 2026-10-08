"""GPU idle-time ("bubble") summary from a PyTorch profiler chrome trace."""
import json


def _union(intervals):
    total, end = 0.0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def gpu_busy_summary(trace_path):
    """Fraction of wall time with any kernel / any compute kernel running on the GPU.

    Communication kernels (NCCL) overlap compute on their own streams, so both
    unions are reported: `busy` counts any kernel, `compute` excludes NCCL.
    The window spans the first to the last GPU activity in the trace.
    """
    with open(trace_path) as f:
        events = json.load(f).get('traceEvents', [])
    kernels, compute = [], []
    for ev in events:
        if ev.get('ph') != 'X' or ev.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        s, d = float(ev['ts']), float(ev.get('dur', 0))
        kernels.append((s, s + d))
        if 'nccl' not in ev.get('name', '').lower() and ev.get('cat') == 'kernel':
            compute.append((s, s + d))
    if not kernels:
        return {'busy': None, 'compute': None, 'window_s': 0.0}
    start = min(s for s, _ in kernels)
    end = max(e for _, e in kernels)
    window = end - start
    return {'busy': _union(kernels) / window, 'compute': _union(compute) / window,
            'window_s': window / 1e6, 'kernels': len(kernels)}
