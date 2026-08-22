import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from Evaluation.evaluate_sparsity import (
    build_model as build_evaluation_model,
    get_checkpoint_pointwise_constraint,
    get_checkpoint_pointwise_parameterization,
)
from models.convnext import Block, ContinuousTTFSConv2d, SpikingBlock
from train_continuous_ttfs_cifar10_32x32_stem1 import (
    architecture_metadata,
    args_parser,
    initialize_constrained_finetune,
    make_model,
    save_checkpoint,
    validate_constrained_finetune_architecture,
)


def model_args(**overrides):
    values = {
        "dataset": "cifar100",
        "num_classes": 100,
        "dims": (8, 16, 32, 64),
        "depths": (1, 1, 1, 1),
        "dw_kernel_size": 3,
        "drop_path": 0.0,
        "t_min": 0.0,
        "t_max": 1.0,
        "head_dropout": 0.0,
        "spike_dropout": 0.0,
        "pw1_mode": "ttfs",
        "pw2_mode": "ttfs",
        "ttfs_norm_mode": "score_layernorm",
        "final_score_norm": True,
        "dwconv_mode": "ttfs",
        "downsample_mode": "ttfs",
        "residual_operator": "min",
        "force_positive_weights": False,
        "force_positive_pointwise_weights": False,
        "init_delay": 0.0,
        "stage_delays": "0.05,0.02,0.01,0.01",
        "input_resolution": 32,
    }
    values.update(overrides)
    return Namespace(**values)


def checkpoint_from_model(model, args):
    return {
        "model": {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        },
        "ema": None,
        "architecture": architecture_metadata(args),
        "args": vars(args),
    }


class PointwiseConstraintTests(unittest.TestCase):
    def test_softplus_initialization_matches_absolute_original_weights(self):
        dense = Block(dim=2, dw_kernel_size=3)
        original_pw1 = dense.pwconv1.weight.detach().abs().clamp_min(1e-6).clone()
        original_pw2 = dense.pwconv2.weight.detach().abs().clamp_min(1e-6).clone()
        block = SpikingBlock(
            dense,
            init_delay=0.05,
            pointwise_weight_parameterization="softplus",
        )

        self.assertTrue(torch.all(block.effective_pointwise_weight(block.pw1.weight) > 0))
        self.assertTrue(torch.all(block.effective_pointwise_weight(block.pw2.weight) > 0))
        torch.testing.assert_close(
            block.effective_pointwise_weight(block.pw1.weight), original_pw1
        )
        torch.testing.assert_close(
            block.effective_pointwise_weight(block.pw2.weight), original_pw2
        )

    def test_softplus_negative_raw_weights_keep_nonzero_gradients(self):
        dense = Block(dim=2, dw_kernel_size=3)
        block = SpikingBlock(
            dense,
            init_delay=0.05,
            pointwise_weight_parameterization="softplus",
        )
        raw = nn.Parameter(torch.tensor([[-8.0, -2.0]]))

        block.effective_pointwise_weight(raw).sum().backward()

        self.assertTrue(torch.all(raw.grad > 0))

    def test_pointwise_constraint_does_not_constrain_ttfs_convolutions(self):
        args = model_args(force_positive_pointwise_weights=True)
        model = make_model(args)
        block = model.stages[0][0]

        self.assertTrue(block.force_positive_pointwise_weights)
        self.assertFalse(block.dwconv.force_positive_weights)
        self.assertFalse(
            model.downsample_layers[1][0].force_positive_weights
        )

        weight = torch.tensor([[-2.0, 3.0]])
        torch.testing.assert_close(
            block.effective_pointwise_weight(weight),
            torch.tensor([[0.0, 3.0]]),
        )

    def test_negative_pointwise_weights_have_zero_relu_gradient(self):
        dense = Block(dim=2, dw_kernel_size=3)
        block = SpikingBlock(
            dense,
            init_delay=0.05,
            force_positive_pointwise_weights=True,
        )
        weight = nn.Parameter(torch.tensor([[-2.0, 3.0]]))

        block.effective_pointwise_weight(weight).sum().backward()

        torch.testing.assert_close(weight.grad, torch.tensor([[0.0, 1.0]]))

    def test_negative_ttfs_convolution_weight_remains_trainable(self):
        convolution = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        convolution.weight.data.fill_(-0.01)
        layer = ContinuousTTFSConv2d(
            convolution,
            init_delay=0.05,
            force_positive_weights=False,
        )

        layer(torch.full((1, 1, 2, 2), 0.5)).sum().backward()

        self.assertLess(convolution.weight.item(), 0.0)
        self.assertNotEqual(convolution.weight.grad.item(), 0.0)


class ConstrainedCheckpointTests(unittest.TestCase):
    def test_cli_rejects_softplus_pretrained_conversion(self):
        argv = [
            "trainer.py",
            "--experiment_name", "test",
            "--pointwise_weight_parameterization", "softplus",
            "--pretrained_checkpoint", "signed.pth",
        ]
        with patch("sys.argv", argv), self.assertRaises(SystemExit) as error:
            args_parser()
        self.assertEqual(error.exception.code, 2)

    def test_cli_maps_legacy_positive_flag_to_relu(self):
        argv = [
            "trainer.py",
            "--experiment_name", "test",
            "--force_positive_pointwise_weights", "true",
        ]
        with patch("sys.argv", argv):
            args = args_parser()
        self.assertEqual(args.pointwise_weight_parameterization, "relu")

    def test_checkpoint_arguments_are_mutually_exclusive(self):
        argv = [
            "trainer.py",
            "--experiment_name", "test",
            "--resume", "resume.pth",
            "--constrained_finetune_checkpoint", "source.pth",
        ]
        with patch("sys.argv", argv), self.assertRaises(SystemExit) as error:
            args_parser()
        self.assertEqual(error.exception.code, 2)

    def test_initializes_strictly_from_unconstrained_fully_ttfs_source(self):
        source_args = model_args()
        target_args = model_args(force_positive_pointwise_weights=True)
        source = make_model(source_args)
        source.head.bias.data.fill_(2.75)
        target = make_model(target_args)
        checkpoint = checkpoint_from_model(source, source_args)

        diagnostics = initialize_constrained_finetune(
            target, checkpoint, target_args
        )

        torch.testing.assert_close(
            target.head.bias,
            torch.full_like(target.head.bias, 2.75),
        )
        self.assertEqual(diagnostics["missing_keys"], [])
        self.assertEqual(diagnostics["unexpected_keys"], [])
        self.assertEqual(diagnostics["source_state"], "model")

    def test_saved_checkpoint_records_constrained_source_lineage(self):
        args = model_args(force_positive_pointwise_weights=True)
        args.constrained_finetune_initialization = {
            "source_checkpoint": "fully_ttfs/seed_42/best_checkpoint.pth"
        }
        model = make_model(args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                path, model, optimizer, scheduler, scaler, None,
                -1, -1.0, -1, 0, args,
            )
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )

        self.assertEqual(
            checkpoint["constrained_finetune_initialization"],
            args.constrained_finetune_initialization,
        )

    def test_rejects_dense_already_constrained_and_wrong_dataset_sources(self):
        target_args = model_args(force_positive_pointwise_weights=True)
        invalid_sources = (
            model_args(dwconv_mode="dense", downsample_mode="dense"),
            model_args(force_positive_pointwise_weights=True),
            model_args(dataset="cifar10", num_classes=10),
        )
        expected = ("fully TTFS", "unconstrained", "dataset")

        for source_args, message in zip(invalid_sources, expected):
            with self.subTest(message=message):
                checkpoint = checkpoint_from_model(
                    make_model(source_args), source_args
                )
                with self.assertRaisesRegex(ValueError, message):
                    validate_constrained_finetune_architecture(
                        checkpoint, target_args
                    )


class EvaluatorConstraintTests(unittest.TestCase):
    def test_evaluator_reconstructs_softplus_parameterization(self):
        checkpoint = {
            "architecture": {
                "pointwise_weight_parameterization": "softplus"
            }
        }
        parameterization = get_checkpoint_pointwise_parameterization(checkpoint)
        self.assertEqual(parameterization, "softplus")
        args = Namespace(
            dataset="cifar100", dw_kernel_size=3, cifar_stem=True
        )
        model = build_evaluation_model(
            args,
            {"dwconv_mode": "ttfs", "downsample_mode": "ttfs"},
            pointwise_weight_parameterization=parameterization,
        )
        self.assertEqual(model.pointwise_weight_parameterization, "softplus")
        self.assertEqual(
            model.stages[0][0].pointwise_weight_parameterization, "softplus"
        )

    def test_legacy_checkpoint_defaults_to_unconstrained(self):
        self.assertFalse(get_checkpoint_pointwise_constraint({}))

    def test_evaluator_reconstructs_pointwise_constraint(self):
        checkpoint = {
            "architecture": {"force_positive_pointwise_weights": True}
        }
        enabled = get_checkpoint_pointwise_constraint(checkpoint)
        args = Namespace(
            dataset="cifar100", dw_kernel_size=3, cifar_stem=True
        )
        model = build_evaluation_model(
            args,
            {"dwconv_mode": "ttfs", "downsample_mode": "ttfs"},
            force_positive_pointwise_weights=enabled,
        )

        self.assertTrue(model.force_positive_pointwise_weights)
        self.assertTrue(model.stages[0][0].force_positive_pointwise_weights)
        self.assertFalse(model.stages[0][0].dwconv.force_positive_weights)


if __name__ == "__main__":
    unittest.main()
