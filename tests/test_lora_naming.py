"""Released-LoRA naming: export then import must be the identity and give the same model."""
import unittest
import torch
from pdmd_training.adapters import install_adapters, role_parameters, merge_role, load_role
from pdmd_training.backend import H3Backend
from pdmd_training.distributed import adapter_state
from scripts.export_lora import convert, from_diffusers
from tests.test_h3_integration import tiny_dit


class LoraNamingTest(unittest.TestCase):
    def test_round_trip_and_load(self):
        import tempfile, os
        from safetensors.torch import save_file
        torch.manual_seed(11)
        dit = tiny_dit()
        install_adapters(dit, rank=2, alpha=2)
        with torch.no_grad():
            for p in role_parameters(dit, 'student'):
                p.normal_(0, .2)
        state = {k: v for k, v in adapter_state(dit).items() if k.endswith('.student')}
        back = from_diffusers(convert(state))
        self.assertEqual(set(back), set(state))
        for k in state:
            torch.testing.assert_close(back[k], state[k], rtol=0, atol=0)
        # Loading the Diffusers-named file into a fresh model reproduces the outputs.
        torch.manual_seed(11)
        fresh = tiny_dit()  # same seed -> same frozen base as `dit`
        install_adapters(fresh, rank=2, alpha=2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'lora.safetensors')
            save_file({k: v.contiguous() for k, v in convert(state).items()}, path)
            load_role(fresh, path, 'student')
        b1, b2 = H3Backend(dit, 'cpu', torch.float32, False), H3Backend(fresh, 'cpu', torch.float32, False)
        noise = b1.noise(5, 32, 32, torch.Generator().manual_seed(3))
        emb, tags = torch.randn(4, 16), torch.ones(4, dtype=torch.long)
        sig = (torch.tensor(.6), torch.tensor(.6))
        for x, y in zip(b1.predict('student', noise, sig, b1.condition(emb, tags, noise)),
                        b2.predict('student', noise, sig, b2.condition(emb, tags, noise))):
            torch.testing.assert_close(x, y)


if __name__ == '__main__':
    unittest.main()
