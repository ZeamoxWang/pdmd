"""CUDA test: alternating role updates and checkpoint restore match GPU moments."""
import copy
import torch
from pdmd_training.optimizer_state import move_moments


def main():
    torch.manual_seed(8)
    params = [{r: torch.nn.Parameter(torch.randn(17, 11, device='cuda'))
               for r in ('student', 'critic')} for _ in range(2)]
    for r in params[0]:
        with torch.no_grad(): params[1][r].copy_(params[0][r])
    opts = [{r: torch.optim.AdamW([p], lr=.01, betas=(0., .9), foreach=False)
             for r, p in ps.items()} for ps in params]
    for step in range(13):
        role = 'student' if step % 6 == 0 else 'critic'
        for i in range(2):
            for opt in opts[i].values(): opt.zero_grad(set_to_none=True)
            if i:
                for other, opt in opts[i].items():
                    if other != role: move_moments(opt, 'cpu')
                move_moments(opts[i][role], 'cuda')
            loss = (params[i][role].sin() * (step+1)).sum()
            loss.backward()
            opts[i][role].step()
        for r in params[0]:
            torch.testing.assert_close(params[0][r], params[1][r], rtol=0, atol=0)
        if step == 5:
            # Simulate checkpoint restore into fresh optimizers, including CPU moments.
            saved = {r: copy.deepcopy(o.state_dict()) for r, o in opts[1].items()}
            for r in opts[1]:
                opts[1][r] = torch.optim.AdamW([params[1][r]], foreach=False)
                opts[1][r].load_state_dict(saved[r])
                move_moments(opts[1][r], 'cpu')
    print('CUDA AdamW offload, TTUR switching and restore: exact parameter equality PASSED', flush=True)


if __name__ == '__main__': main()
