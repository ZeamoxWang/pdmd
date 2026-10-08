import unittest
import torch
from pdmd_training.objectives import (project_update, endpoint, noise_estimate, endpoint_update,
                                      surrogate_loss, renoise, critic_loss)
from pdmd_training.schedule import schedules, update_kind, critic_times, score_times, student_grid
from pdmd_training.data import SampleStream


class ObjectivesTest(unittest.TestCase):
    def test_projection_is_per_sample_and_orthogonal(self):
        torch.manual_seed(12)
        d, r = torch.randn(3, 2, 13), torch.randn(3, 2, 13)
        p = project_update(d, r)
        torch.testing.assert_close((p*r).sum((1, 2)), torch.zeros(3), atol=2e-6, rtol=0)
        self.assertTrue(torch.all(p.square().sum((1, 2)) <= d.square().sum((1, 2))+1e-5))
        torch.testing.assert_close(project_update(p, r), p, atol=1e-6, rtol=1e-5)
        for i in range(3):
            torch.testing.assert_close(p[i:i+1], project_update(d[i:i+1], r[i:i+1]))

    def test_zero_residual_leaves_update_unchanged(self):
        d = torch.tensor([[2., 3.]])
        torch.testing.assert_close(project_update(d, torch.zeros_like(d)), d)

    def test_velocity_sign_and_noise_estimate(self):
        clean, noise = torch.randn(2, 10), torch.randn(2, 10)
        x = renoise(clean, noise, .7)
        torch.testing.assert_close(endpoint(x, noise-clean, .7), clean)
        torch.testing.assert_close(noise_estimate(x, noise-clean, .7), noise)
        self.assertEqual(critic_loss(noise-clean, clean, noise).item(), 0.)

    def test_projection_matches_student_perpendicular_form(self):
        torch.manual_seed(3)
        x, t, c = torch.randn(1, 4, 9), torch.randn(1, 4, 9), torch.randn(1, 4, 9)
        u, bad = endpoint_update(x, t, c, projected=True)
        p_real, p_fake = x - t, x - c
        d = p_real - p_fake
        expected = (d - (d*p_fake).sum()/((p_fake*p_fake).sum()+1e-8)*p_fake) / p_real.abs().mean()
        torch.testing.assert_close(u, expected)
        self.assertEqual(bad, 0)
        u_dmd, _ = endpoint_update(x, t, c, projected=False)
        torch.testing.assert_close(u_dmd, d / p_real.abs().mean())

    def test_detached_surrogate_gradient(self):
        x = torch.randn(2, 10, requires_grad=True)
        t, c = torch.randn_like(x, requires_grad=True), torch.randn_like(x, requires_grad=True)
        u, _ = endpoint_update(x, t, c)
        surrogate_loss(x, u, clamp_max=None).backward()
        torch.testing.assert_close(x.grad, 2*u/x.numel())
        self.assertIsNone(t.grad)
        self.assertIsNone(c.grad)

    def test_clamp_turns_off_gradient(self):
        x = torch.zeros(1, 4, requires_grad=True)
        u = torch.full((1, 4), 10.)
        surrogate_loss(x, u, clamp_max=5.).backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_stereo_is_projected_as_one_sample(self):
        d = torch.tensor([[[1., 0.], [0., 2.]]])
        r = torch.tensor([[[1., 0.], [0., 1.]]])
        torch.testing.assert_close(project_update(d, r), torch.tensor([[[-.5, 0.], [0., .5]]]), atol=1e-6, rtol=0)

    def test_grids_and_update_order(self):
        v, a = schedules()
        torch.testing.assert_close(v, torch.tensor([1., .75, .5, .25, 0.]))
        torch.testing.assert_close(a, v)
        torch.testing.assert_close(student_grid(4, 12.), torch.tensor([1., 36/37, 12/13, .8, 0.]))
        self.assertEqual([update_kind(i) for i in range(12)], (['critic']*5+['student'])*2)

    def test_score_and_critic_times(self):
        v, a = schedules(4, 12., 3.)
        sv, sa = score_times(v, a, 1, .25)
        self.assertAlmostEqual(float(sv), float(v[2] + .25*(v[1]-v[2])), places=6)
        self.assertAlmostEqual(float(sa), float(a[2] + .25*(a[1]-a[2])), places=6)
        g = torch.Generator().manual_seed(0)
        draws = [critic_times(g, 'cpu') for _ in range(4000)]
        for tv, ta in draws:
            self.assertTrue(0 <= tv <= 1 and 0 <= ta <= 1)
            if tv > .95:
                self.assertGreaterEqual(float(ta), .85)

    def test_sample_stream_is_world_size_independent(self):
        s = SampleStream(100, 16, 5, [(544, 960)], seed=1)
        self.assertEqual(s.critic_updates_before(6), 5)
        first = [s.index(0, slot) for slot in range(16)]
        self.assertEqual(len(set(first)), 16)
        # Student iterations consume no prompts: iteration 6 continues after 5 critic updates.
        self.assertEqual(s.position(6, 0), 5*16)
        seen = [s.index(it, slot) for it in range(0, 6) if update_kind(it) == 'critic' for slot in range(16)]
        self.assertEqual(len(seen), 80)
        self.assertEqual(len(set(seen)), 80)


if __name__ == '__main__':
    unittest.main()
