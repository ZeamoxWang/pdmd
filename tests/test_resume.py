"""A resumed student update must match reuse of the last critic trajectory."""
import copy
import io
import unittest
import torch
from pdmd_training.adapters import install_adapters, role_parameters
from pdmd_training.backend import H3Backend
from pdmd_training.distributed import adapter_state, restore_adapters
from pdmd_training.engine import rollout, critic_step_loss, student_step_loss
from pdmd_training.schedule import schedules
from tests.test_h3_integration import tiny_dit


class ResumeTest(unittest.TestCase):
    def test_optimizer_restore_and_trajectory_reconstruction(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        model = tiny_dit()
        install_adapters(model, rank=2, alpha=2)
        backend = H3Backend(model, 'cpu', torch.float32, checkpoint=True)
        optimizers = {r: torch.optim.AdamW(role_parameters(model, r), lr=.001,
                                          betas=(0., .9), weight_decay=0.)
                      for r in ('student', 'critic')}
        emb = torch.randn(4, 16)
        vg, ag = schedules(4, 1., 1.)
        sig = (torch.tensor(.6), torch.tensor(.3))

        def trajectory(b, seed):
            noise = b.noise(5, 32, 32, torch.Generator().manual_seed(seed))
            cond = b.condition(emb, torch.ones(4, dtype=torch.long), noise)
            return rollout(b, noise, cond, vg, ag)

        def fresh(tr, seed):
            g = torch.Generator().manual_seed(seed)
            return tuple(torch.randn(x.shape, generator=g) for x in tr.states[-1])

        for role, seed in [('critic', 92), ('student', 93), ('critic', 94)]:
            model.zero_grad(set_to_none=True)
            tr = trajectory(backend, seed)
            loss = (critic_step_loss(backend, tr, sig, fresh(tr, seed))[0] if role == 'critic'
                    else student_step_loss(backend, tr, 2, .4)[0])
            loss.backward()
            optimizers[role].step()
        checkpoint = io.BytesIO()
        torch.save({'adapters': adapter_state(model),
                    'optimizers': {r: o.state_dict() for r, o in optimizers.items()}}, checkpoint)
        fresh_model = copy.deepcopy(model)
        with torch.no_grad():
            for role in optimizers:
                for p in role_parameters(fresh_model, role):
                    p.zero_()
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restore_adapters(fresh_model, saved['adapters'])
        resumed_opts = {r: torch.optim.AdamW(role_parameters(fresh_model, r), lr=.123)
                        for r in optimizers}
        for r, opt in resumed_opts.items():
            opt.load_state_dict(saved['optimizers'][r])
        fresh_backend = H3Backend(fresh_model, 'cpu', torch.float32, checkpoint=True)
        reconstructed = trajectory(fresh_backend, 94)
        for cached, restored in zip(tr.states, reconstructed.states):
            for x, y in zip(cached, restored):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        for b, t, opts in [(backend, tr, optimizers), (fresh_backend, reconstructed, resumed_opts)]:
            b.dit.zero_grad(set_to_none=True)
            student_step_loss(b, t, 1, .35)[0].backward()
            opts['student'].step()
        expected, actual = adapter_state(model), adapter_state(fresh_model)
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
