"""Two-GPU real-H3-architecture test at tiny size, no downloaded weights."""
import os
import torch
import torch.distributed as dist
from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
from pdmd_training.adapters import install_adapters, role_parameters
from pdmd_training.backend import H3Backend
from pdmd_training.distributed import shard_base, reduce_gradients, adapter_state, restore_adapters
from pdmd_training.engine import rollout, student_step_loss, critic_step_loss


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    # Known rank-dependent gradients test averaging, absent gradients and buckets.
    probe = [torch.nn.Parameter(torch.zeros(n, device=torch.device('cuda', local_rank))) for n in (3, 7, 11)]
    probe[0].grad = torch.full_like(probe[0], float(rank + 1))
    probe[2].grad = torch.full_like(probe[2], float(2 * rank))
    reduce_gradients(probe, bucket_bytes=32)
    torch.testing.assert_close(probe[0].grad, torch.full_like(probe[0], (world + 1) / 2))
    torch.testing.assert_close(probe[1].grad, torch.zeros_like(probe[1]))
    torch.testing.assert_close(probe[2].grad, torch.full_like(probe[2], float(world - 1)))
    torch.manual_seed(100)
    dit = MiniMaxH3DiT(num_layers=2,token_refiner_num_layers=1,hidden_size=32,
                      num_attention_heads=2,attention_head_dim=16,ffn_hidden_size=64,
                      text_dim=16,timestep_input_dim=16,time_embed_hidden_size=32,
                      time_embed_dim=16,adaln_out_features=576,
                      final_adaln_out_features=64,rope_inv_freq_len=2).to(torch.bfloat16).requires_grad_(False)
    install_adapters(dit,rank=2,alpha=2)
    device=torch.device('cuda',local_rank)
    dit=shard_base(dit,device)
    backend=H3Backend(dit,device)
    params={r:role_parameters(dit,r) for r in ('student','critic')}
    opts={r:torch.optim.AdamW(ps,lr=.01) for r,ps in params.items()}
    gen=torch.Generator(device=device).manual_seed(101+rank)
    # Exercise a complete TTUR cycle plus the next student update. Repeated
    # forwards/backwards and accumulation catch FSDP lifetime errors that a
    # single update cannot expose.
    accumulation = int(os.environ.get('PDMD_TEST_ACCUMULATION', '2'))
    for iteration in range(7):
        role = 'student' if iteration % 6 == 0 else 'critic'
        for opt in opts.values():opt.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            noise=backend.noise(5,32,32,gen)
            cond=backend.condition(torch.randn(4,16),torch.ones(4,dtype=torch.long),noise)
            tr=rollout(backend,noise,cond)
            loss=(critic_step_loss(backend,tr,.6) if role=='critic' else
                  student_step_loss(backend,tr,2,.4))
            (loss/accumulation).backward()
            assert torch.isfinite(loss), role
        other='student' if role=='critic' else 'critic'
        assert all(p.grad is None for p in params[other]), 'gradient leaked'
        reduce_gradients(params[role])
        # At initialization teacher and critic coincide, so the first PDMD
        # student direction is exactly zero. Critic updates break the equality.
        if iteration > 0:
            assert any(p.grad.abs().sum()>0 for p in params[role]), 'missing gradient'
        opts[role].step()
        print(f'rank={rank} iteration={iteration+1} role={role} accumulation={accumulation} passed',flush=True)
    state=adapter_state(dit)
    restore_adapters(dit,state)
    for p in params['student']+params['critic']:
        expected=p.detach().clone()
        dist.broadcast(expected,src=0)
        torch.testing.assert_close(p,expected)
    print(f'rank={rank} base_shard_world={os.environ.get("LOCAL_WORLD_SIZE")} FSDP wrapper, PDMD, gradient isolation and adapter sync PASSED',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
