import unittest
from argparse import Namespace

import torch

from evaluate_ttfs_cifar10_tta import (
    architecture_from_checkpoint,
    build_model,
    checkpoint_integrity,
    evaluate,
    make_views,
)
from train_continuous_ttfs_cifar10_32x32_stem1 import (
    architecture_metadata,
    make_model,
)


def model_args(operator="min", pointwise_constraint=False):
    return Namespace(
        dataset="cifar10", num_classes=10,
        dims=(8, 16, 32, 64), depths=(1, 1, 1, 1),
        dw_kernel_size=3, drop_path=0.0, t_min=0.0, t_max=1.0,
        head_dropout=0.0, spike_dropout=0.0, pw1_mode="ttfs",
        pw2_mode="ttfs", ttfs_norm_mode="score_layernorm",
        final_score_norm=True, dwconv_mode="ttfs",
        downsample_mode="ttfs", residual_operator=operator,
        force_positive_weights=False,
        force_positive_pointwise_weights=pointwise_constraint,
        init_delay=0.0, stage_delays="0.05,0.02,0.01,0.01",
        input_resolution=32,
    )


def checkpoint_for(args):
    model = make_model(args)
    return {
        "model": model.state_dict(),
        "architecture": architecture_metadata(args),
        "args": vars(args),
    }


class TTAResidualAblationTests(unittest.TestCase):
    def test_reconstructs_all_residual_operators_strictly(self):
        for operator in ("min", "mean", "learnable_gate"):
            with self.subTest(operator=operator):
                checkpoint = checkpoint_for(model_args(operator))
                restored, architecture = build_model(checkpoint)
                integrity = checkpoint_integrity(restored, checkpoint["model"])

                self.assertEqual(architecture["residual_operator"], operator)
                self.assertEqual(restored.residual_operator, operator)
                self.assertEqual(integrity["missing_keys"], [])
                self.assertEqual(integrity["unexpected_keys"], [])
                self.assertEqual(integrity["shape_mismatches"], [])

    def test_reconstructs_pointwise_constraint(self):
        checkpoint = checkpoint_for(model_args("mean", True))

        restored, _ = build_model(checkpoint)

        self.assertTrue(restored.force_positive_pointwise_weights)

    def test_rejects_missing_or_invalid_residual_metadata(self):
        checkpoint = checkpoint_for(model_args("min"))
        checkpoint["architecture"].pop("residual_operator")
        with self.assertRaisesRegex(RuntimeError, "residual_operator"):
            architecture_from_checkpoint(checkpoint)

        checkpoint["architecture"]["residual_operator"] = "sum"
        with self.assertRaisesRegex(RuntimeError, "residual_operator"):
            architecture_from_checkpoint(checkpoint)

    def test_bounded_flip_shift_evaluation_uses_ten_finite_views(self):
        checkpoint = checkpoint_for(model_args("mean"))
        model, _ = build_model(checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        images = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([0, 1])

        self.assertEqual(len(make_views(images, "flip_shift")), 10)
        metrics = evaluate(
            [model], [(images, labels)], "flip_shift", torch.device("cpu"),
            False, 0.0, 1.0, tuple(str(index) for index in range(10)), False,
        )

        self.assertEqual(metrics["views_per_model"], 10)
        self.assertEqual(metrics["forward_passes_per_sample"], 10)
        self.assertEqual(metrics["samples"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
