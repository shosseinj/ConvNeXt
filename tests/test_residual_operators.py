import unittest
from argparse import Namespace

import torch

from Evaluation.evaluate_sparsity import get_checkpoint_residual_operator
from train_continuous_ttfs_cifar10_32x32_stem1 import (
    architecture_metadata,
    make_model,
)


def model_args(residual_operator):
    return Namespace(
        num_classes=10,
        dims=(8, 16, 32, 64),
        depths=(1, 1, 1, 1),
        dw_kernel_size=3,
        drop_path=0.0,
        t_min=0.0,
        t_max=1.0,
        head_dropout=0.0,
        spike_dropout=0.0,
        pw2_mode="ttfs",
        ttfs_norm_mode="score_layernorm",
        final_score_norm=True,
        dwconv_mode="dense",
        downsample_mode="dense",
        force_positive_weights=False,
        init_delay=0.0,
        stage_delays="0.05,0.02,0.01,0.01",
        input_resolution=32,
        residual_operator=residual_operator,
    )


class ResidualOperatorTests(unittest.TestCase):
    def test_min_preserves_existing_earliest_spike_behavior(self):
        model = make_model(model_args("min"))
        block = model.stages[0][0]
        identity = torch.tensor([[[[0.2]], [[0.8]]]])
        branch = torch.tensor([[[[0.6]], [[0.3]]]])

        output = block.combine_residual(identity, branch)

        torch.testing.assert_close(output, torch.minimum(identity, branch))
        self.assertNotIn("raw_residual_gate", dict(block.named_parameters()))

    def test_mean_is_bounded_arithmetic_time_average(self):
        model = make_model(model_args("mean"))
        block = model.stages[0][0]
        identity = torch.tensor([[[[0.2]], [[0.8]]]])
        branch = torch.tensor([[[[0.6]], [[0.3]]]])

        output = block.combine_residual(identity, branch)

        torch.testing.assert_close(output, (identity + branch) / 2.0)
        self.assertTrue(torch.all((0.0 <= output) & (output <= 1.0)))
        self.assertNotIn("raw_residual_gate", dict(block.named_parameters()))

    def test_learnable_gate_is_per_channel_bounded_and_trainable(self):
        model = make_model(model_args("learnable_gate"))
        block = model.stages[0][0]
        identity = torch.linspace(0.1, 0.8, 8).reshape(1, 8, 1, 1).requires_grad_()
        branch = torch.linspace(0.9, 0.2, 8).reshape(1, 8, 1, 1).requires_grad_()

        output = block.combine_residual(identity, branch)
        output.sum().backward()

        torch.testing.assert_close(output.detach(), (identity.detach() + branch.detach()) / 2.0)
        self.assertEqual(tuple(block.raw_residual_gate.shape), (8,))
        self.assertIsNotNone(block.raw_residual_gate.grad)
        self.assertTrue(torch.isfinite(block.raw_residual_gate.grad).all())
        self.assertGreater(block.raw_residual_gate.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.all((0.0 <= output.detach()) & (output.detach() <= 1.0)))

    def test_checkpoint_metadata_records_residual_operator(self):
        metadata = architecture_metadata(model_args("learnable_gate"))

        self.assertEqual(metadata["residual_operator"], "learnable_gate")

    def test_min_checkpoint_state_remains_strictly_compatible(self):
        original = make_model(model_args("min"))
        restored = make_model(model_args("min"))

        incompatible = restored.load_state_dict(original.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_evaluator_defaults_legacy_checkpoint_to_min(self):
        self.assertEqual(get_checkpoint_residual_operator({"architecture": {}}), "min")

    def test_evaluator_reads_supported_residual_operator(self):
        checkpoint = {"architecture": {"residual_operator": "learnable_gate"}}
        self.assertEqual(
            get_checkpoint_residual_operator(checkpoint),
            "learnable_gate",
        )

    def test_evaluator_rejects_invalid_residual_operator(self):
        checkpoint = {"architecture": {"residual_operator": "sum"}}
        with self.assertRaisesRegex(ValueError, "residual_operator"):
            get_checkpoint_residual_operator(checkpoint)


if __name__ == "__main__":
    unittest.main()
