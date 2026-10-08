import unittest
import torch
from pdmd_training.adapters import DualLoRA, SplitQKVDualLoRA


class AdaptersTest(unittest.TestCase):
    def test_independent_qkv_matches_three_projections(self):
        layer=SplitQKVDualLoRA(torch.nn.Linear(7,24,bias=False),rank=2,alpha=2,head_dim=4)
        layer.active='student'
        with torch.no_grad():layer.lora_B['student'].normal_()
        x=torch.randn(5,7)
        parts=[]
        for a,b in zip(layer.lora_A['student'].chunk(3),layer.lora_B['student'].chunk(3)):
            parts.append(torch.nn.functional.linear(torch.nn.functional.linear(x,a),b))
        expected=torch.stack(parts,dim=-2).reshape(5,3,2,4).transpose(1,2).flatten(1)
        torch.testing.assert_close(layer(x)-layer.base(x),expected)

    def test_teacher_unchanged_and_critic_isolated(self):
        layer = DualLoRA(torch.nn.Linear(4, 3), rank=2, alpha=2)
        x = torch.randn(5, 4)
        base = layer(x).detach().clone()
        opt = torch.optim.SGD([layer.lora_A['student'], layer.lora_B['student']], lr=.1)
        layer.active = 'student'
        layer(x).square().sum().backward()
        opt.step()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNone(layer.lora_B['critic'].grad)
        self.assertGreater((layer(x)-base).abs().max().item(), 0)
        layer.active = 'teacher'
        torch.testing.assert_close(layer(x), base)
        layer.active = 'critic'
        torch.testing.assert_close(layer(x), base)


if __name__ == '__main__':
    unittest.main()
