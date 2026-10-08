"""Real DiffSynth H3 architecture at tiny dimensions, no pretrained weights."""
import unittest
import torch
from pdmd_training.adapters import install_adapters, role_parameters
from pdmd_training.backend import H3Backend, enable_cached_attention_bounds
from pdmd_training.engine import rollout, student_step_loss, critic_step_loss
from pdmd_training.schedule import schedules


def tiny_dit():
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
    return MiniMaxH3DiT(num_layers=2, token_refiner_num_layers=1, hidden_size=32,
                        num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64,
                        text_dim=16, timestep_input_dim=16, time_embed_hidden_size=32,
                        time_embed_dim=16, adaln_out_features=576,
                        final_adaln_out_features=64, rope_inv_freq_len=2).requires_grad_(False)


class H3IntegrationTest(unittest.TestCase):
    def test_pdmd_backward_with_checkpointing(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        dit = tiny_dit()
        install_adapters(dit, rank=2, alpha=2)
        backend = H3Backend(dit, 'cpu', torch.float32, checkpoint=True)
        noise = backend.noise(5, 32, 32, torch.Generator().manual_seed(22))
        cond = backend.condition(torch.randn(4, 16), torch.ones(4, dtype=torch.long), noise)
        vg, ag = schedules(4, 1., 1.)
        tr = rollout(backend, noise, cond, vg, ag)
        fresh = tuple(torch.randn_like(x) for x in tr.states[-1])
        sig = (torch.tensor(.6), torch.tensor(.6))
        # Initially teacher==critic, so DMD is exactly zero. First fit the critic.
        opt = torch.optim.SGD(role_parameters(dit, 'critic'), lr=.01)
        critic_step_loss(backend, tr, sig, fresh)[0].backward()
        opt.step()
        dit.zero_grad(set_to_none=True)
        loss, stats = student_step_loss(backend, tr, index=2, ratio=.4)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['nonfinite_update'], 0)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in role_parameters(dit, 'student')))
        self.assertTrue(all(p.grad is None for p in role_parameters(dit, 'critic')))
        dit.zero_grad(set_to_none=True)
        loss, _ = critic_step_loss(backend, tr, sig, fresh)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in role_parameters(dit, 'critic')))
        self.assertTrue(all(p.grad is None for p in role_parameters(dit, 'student')))
        self.assertTrue(all(p.grad is None for n, p in dit.named_parameters() if 'lora_' not in n))

    def test_cached_attention_bounds_match_upstream(self):
        torch.manual_seed(5)
        dit = tiny_dit()
        backend = H3Backend(dit, 'cpu', torch.float32, checkpoint=False)
        noise = backend.noise(5, 32, 32, torch.Generator().manual_seed(6))
        emb, tags = torch.randn(4, 16), torch.ones(4, dtype=torch.long)
        sig = (torch.tensor(.5), torch.tensor(.5))
        before = backend.predict('teacher', noise, sig, backend.condition(emb, tags, noise))
        enable_cached_attention_bounds()
        after = backend.predict('teacher', noise, sig, backend.condition(emb, tags, noise))
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
