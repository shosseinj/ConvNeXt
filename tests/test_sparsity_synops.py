import unittest
from argparse import Namespace

import torch

from Evaluation.evaluate_sparsity import (
    SparsityCounter,
    convolution_synops,
    linear_synops,
)
from train_continuous_ttfs_cifar10_32x32_stem1 import make_model


def model_args():
    return Namespace(
        dataset="cifar10", num_classes=10,
        dims=(8, 16, 32, 64), depths=(1, 1, 1, 1),
        dw_kernel_size=3, drop_path=0.0, t_min=0.0, t_max=1.0,
        head_dropout=0.0, spike_dropout=0.0, pw1_mode="ttfs",
        pw2_mode="ttfs", ttfs_norm_mode="score_layernorm",
        final_score_norm=True, dwconv_mode="ttfs",
        downsample_mode="ttfs", residual_operator="min",
        force_positive_weights=False, force_positive_pointwise_weights=False,
        init_delay=0.0, stage_delays="0.05,0.02,0.01,0.01",
        input_resolution=32,
    )


class SparsitySynOpsTests(unittest.TestCase):
    def test_eval_forward_retains_actual_pointwise_inputs_for_synops(self):
        args = model_args()
        model = make_model(args).eval()

        with torch.no_grad():
            model(torch.rand(2, 3, 32, 32))

        block = model.stages[0][0]
        self.assertIsNotNone(block.t_pw1_input_spike)
        self.assertIsNotNone(block.t_pw2_input_spike)
        self.assertEqual(block.t_pw1_input_spike.shape[-1], block.pw1.in_features)
        self.assertEqual(block.t_pw2_input_spike.shape[-1], block.pw2.in_features)

    def test_linear_synops_counts_active_inputs_times_nonzero_outgoing_weights(self):
        times = torch.tensor([[0.2, 1.0, 0.4], [1.0, 0.3, 0.7]])
        weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])

        self.assertEqual(linear_synops(times, weight, 1.0), 6)

    def test_convolution_synops_respects_padding_stride_and_nonzero_weights(self):
        times = torch.tensor([[[[0.2, 1.0], [0.3, 0.4]]]])
        weight = torch.ones(1, 1, 2, 2)

        self.assertEqual(
            convolution_synops(
                times,
                weight,
                t_max=1.0,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
            ),
            3,
        )

    def test_counter_reconciles_layer_and_global_synops(self):
        counter = SparsityCounter()
        counter.add("a", torch.tensor([0.2, 1.0]), 1.0, synops=7)
        counter.add("b", torch.tensor([0.3, 0.4]), 1.0, synops=11)

        self.assertEqual(counter.data["a"]["synops"], 7)
        self.assertEqual(counter.global_synops(), 18)


if __name__ == "__main__":
    unittest.main()
