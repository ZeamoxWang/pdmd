"""Merging the student adapters into the base must reproduce the adapted model."""
import unittest
import torch
from pdmd_training.adapters import install_adapters, set_role, merge_role, role_parameters
from pdmd_training.backend import H3Backend
from tests.test_h3_integration import tiny_dit


class MergeTest(unittest.TestCase):
    def test_merged_matches_adapted(self):
        torch.manual_seed(7)
        dit = tiny_dit()
        install_adapters(dit, rank=2, alpha=4)
        with torch.no_grad():
            for p in role_parameters(dit, 'student'):
                p.normal_(0, .2)
        backend = H3Backend(dit, 'cpu', torch.float32, checkpoint=False)
        noise = backend.noise(5, 32, 32, torch.Generator().manual_seed(8))
        cond = backend.condition(torch.randn(4, 16), torch.ones(4, dtype=torch.long), noise)
        sig = (torch.tensor(.7), torch.tensor(.4))
        adapted = backend.predict('student', noise, sig, cond)
        teacher = backend.predict('teacher', noise, sig, cond)
        self.assertGreater(float((adapted[0] - teacher[0]).abs().max()), 1e-3)
        n = merge_role(dit, 'student')
        self.assertGreater(n, 0)
        merged = backend.predict('teacher', noise, sig, cond)
        for x, y in zip(adapted, merged):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
