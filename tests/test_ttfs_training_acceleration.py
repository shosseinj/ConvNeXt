import sys
import types
import unittest

import torch
from torch import nn


if "timm" not in sys.modules:
    timm = types.ModuleType("timm")
    timm.layers = types.ModuleType("timm.layers")
    timm.models = types.ModuleType("timm.models")

    class DropPath(nn.Identity):
        pass

    timm.layers.DropPath = DropPath
    timm.layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm.models.register_model = lambda function: function
    sys.modules["timm"] = timm
    sys.modules["timm.layers"] = timm.layers
    sys.modules["timm.models"] = timm.models


from models.convnext import (  # noqa: E402
    Block,
    SpikingBlock,
    call_spiking_linear,
    call_spiking_torch,
)


class TTFSTrainingAccelerationTests(unittest.TestCase):
    def make_block(self):
        dense = Block(dim=4, dw_kernel_size=3)
        return SpikingBlock(
            dense,
            t_min=0.0,
            t_max=1.0,
            init_delay=0.05,
            pw2_mode="ttfs",
            ttfs_norm_mode="score_layernorm",
            dwconv_mode="ttfs",
        )

    def test_native_linear_layout_matches_legacy_spiking_transform(self):
        torch.manual_seed(3)
        spike_times = torch.rand(13, 4, requires_grad=True)
        native_weight = torch.randn(7, 4, requires_grad=True)
        delay = torch.rand(7, requires_grad=True)

        legacy = call_spiking_torch(
            spike_times,
            native_weight.t().contiguous(),
            delay,
            None,
            0.0,
            1.0,
        )
        optimized = call_spiking_linear(
            spike_times,
            native_weight,
            delay,
            0.0,
            1.0,
        )

        torch.testing.assert_close(optimized, legacy)
        legacy.sum().backward(retain_graph=True)
        legacy_grads = (
            spike_times.grad.clone(),
            native_weight.grad.clone(),
            delay.grad.clone(),
        )
        spike_times.grad = native_weight.grad = delay.grad = None
        optimized.sum().backward()
        optimized_grads = (
            spike_times.grad,
            native_weight.grad,
            delay.grad,
        )
        for actual, expected in zip(optimized_grads, legacy_grads):
            torch.testing.assert_close(actual, expected)

    def test_training_does_not_retain_pointwise_spike_outputs(self):
        block = self.make_block()
        block.train()
        output = block(torch.rand(2, 4, 8, 8, requires_grad=True))
        output.sum().backward()

        self.assertIsNone(block.t_mid_spike)
        self.assertIsNone(block.t_out_spike)

    def test_evaluation_retains_detached_pointwise_spike_outputs(self):
        block = self.make_block()
        block.eval()
        with torch.no_grad():
            block(torch.rand(2, 4, 8, 8))

        self.assertIsNotNone(block.t_mid_spike)
        self.assertIsNotNone(block.t_out_spike)
        self.assertFalse(block.t_mid_spike.requires_grad)
        self.assertFalse(block.t_out_spike.requires_grad)


if __name__ == "__main__":
    unittest.main()
