"""CPU distributed test for the exact gradient averaging used across nodes."""
from datetime import timedelta
import torch
import torch.distributed as dist
from pdmd_training.distributed import reduce_gradients


def main():
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    params = [torch.nn.Parameter(torch.zeros(n, dtype=dtype))
              for n, dtype in [(3, torch.float32), (7, torch.float32),
                               (11, torch.float32), (2, torch.float64)]]
    for i, p in enumerate(params):
        if i != 1:
            p.grad = torch.arange(p.numel(), dtype=p.dtype) + rank * (i + 1)
    expected = []
    for p in params:
        grad = torch.zeros_like(p) if p.grad is None else p.grad.clone()
        dist.all_reduce(grad)
        expected.append(grad / world)
    reduce_gradients(params, bucket_bytes=32)
    for p, reference in zip(params, expected):
        torch.testing.assert_close(p.grad, reference, rtol=0, atol=0)
    print(f'rank={rank} bucketed gradients match individual all-reduces exactly', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
