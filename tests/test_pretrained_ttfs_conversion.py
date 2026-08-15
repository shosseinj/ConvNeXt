import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from train_continuous_ttfs_cifar10_32x32_stem1 import (
    architecture_metadata,
    apply_pretrained_lineage,
    args_parser,
    apply_warmup_learning_rates,
    build_optimizer,
    convert_dense_checkpoint_to_ttfs,
    make_model,
    validate_pretrained_architecture,
)


def model_args(dwconv_mode="dense", downsample_mode="dense"):
    return Namespace(
        dataset="cifar10",
        num_classes=10,
        dims=(8, 16, 32, 64),
        depths=(1, 1, 1, 1),
        dw_kernel_size=3,
        drop_path=0.0,
        t_min=0.0,
        t_max=1.0,
        head_dropout=0.0,
        spike_dropout=0.0,
        pw1_mode="ttfs",
        pw2_mode="ttfs",
        ttfs_norm_mode="score_layernorm",
        final_score_norm=True,
        dwconv_mode=dwconv_mode,
        downsample_mode=downsample_mode,
        residual_operator="min",
        force_positive_weights=False,
        init_delay=0.0,
        stage_delays="0.05,0.02,0.01,0.01",
        input_resolution=32,
    )


def checkpoint_from_model(model, args, ema=None):
    return {
        "model": model.state_dict(),
        "ema": ema,
        "architecture": architecture_metadata(args),
        "args": vars(args),
    }


class DenseToTTFSConversionTests(unittest.TestCase):
    def test_differential_optimizer_isolates_new_convolution_delays(self):
        args = model_args("ttfs", "ttfs")
        args.lr = 2e-5
        args.conv_delay_lr = 1e-4
        args.weight_decay = 0.05
        model = make_model(args)

        optimizer = build_optimizer(model, args)

        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["transferred", "conv_delays"],
        )
        self.assertEqual(
            [group["target_lr"] for group in optimizer.param_groups],
            [2e-5, 1e-4],
        )
        delay_parameters = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name.endswith("D_conv")
        }
        optimizer_delay_parameters = {
            id(parameter) for parameter in optimizer.param_groups[1]["params"]
        }
        self.assertEqual(optimizer_delay_parameters, delay_parameters)
        self.assertEqual(len(delay_parameters), 7)

    def test_warmup_preserves_differential_learning_rate_ratio(self):
        args = model_args("ttfs", "ttfs")
        args.lr = 2e-5
        args.conv_delay_lr = 1e-4
        args.weight_decay = 0.05
        optimizer = build_optimizer(make_model(args), args)

        apply_warmup_learning_rates(optimizer, epoch=0, warmup_epochs=3)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-5 / 3)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-4 / 3)

    def test_resume_restores_pretrained_lineage(self):
        args = Namespace(pretrained_checkpoint="")
        initialization = {
            "source_checkpoint": "dense/best_checkpoint.pth",
            "source_state": "ema",
        }

        apply_pretrained_lineage(
            args,
            {"pretrained_initialization": initialization},
        )

        self.assertEqual(args.pretrained_initialization, initialization)
        self.assertEqual(
            args.pretrained_checkpoint,
            "dense/best_checkpoint.pth",
        )

    def test_resume_and_pretrained_checkpoint_are_mutually_exclusive(self):
        argv = [
            "trainer.py",
            "--experiment_name",
            "test",
            "--resume",
            "resume.pth",
            "--pretrained_checkpoint",
            "dense.pth",
        ]
        with patch("sys.argv", argv), self.assertRaises(SystemExit) as error:
            args_parser()
        self.assertEqual(error.exception.code, 2)

    def test_transfers_all_dense_parameters_and_only_initializes_delays(self):
        dense_args = model_args()
        target_args = model_args("ttfs", "ttfs")
        dense = make_model(dense_args)
        target = make_model(target_args)
        checkpoint = checkpoint_from_model(dense, dense_args)

        diagnostics = convert_dense_checkpoint_to_ttfs(
            target,
            checkpoint,
            target_args,
        )

        target_state = target.state_dict()
        source_state = dense.state_dict()
        self.assertEqual(diagnostics["source_state"], "model")
        self.assertEqual(len(diagnostics["initialized_delay_keys"]), 7)
        self.assertEqual(diagnostics["missing_keys"], [])
        self.assertEqual(diagnostics["unexpected_keys"], [])
        torch.testing.assert_close(
            target_state["stages.0.0.dwconv.conv.weight"],
            source_state["stages.0.0.dwconv.weight"],
        )
        torch.testing.assert_close(
            target_state["downsample_layers.1.0.conv.weight"],
            source_state["downsample_layers.1.0.weight"],
        )

    def test_prefers_ema_state(self):
        dense_args = model_args()
        target_args = model_args("ttfs", "ttfs")
        dense = make_model(dense_args)
        ema = {name: tensor.clone() for name, tensor in dense.state_dict().items()}
        ema["head.bias"].fill_(3.25)
        checkpoint = checkpoint_from_model(dense, dense_args, ema=ema)
        target = make_model(target_args)

        diagnostics = convert_dense_checkpoint_to_ttfs(
            target,
            checkpoint,
            target_args,
        )

        self.assertEqual(diagnostics["source_state"], "ema")
        torch.testing.assert_close(
            target.head.bias,
            torch.full_like(target.head.bias, 3.25),
        )

    def test_rejects_non_dense_source_architecture(self):
        source_args = model_args("ttfs", "dense")
        target_args = model_args("ttfs", "ttfs")
        source = make_model(source_args)
        checkpoint = checkpoint_from_model(source, source_args)

        with self.assertRaisesRegex(ValueError, "dense depthwise"):
            validate_pretrained_architecture(checkpoint, target_args)

    def test_rejects_incompatible_classifier(self):
        dense_args = model_args()
        target_args = model_args("ttfs", "ttfs")
        dense = make_model(dense_args)
        checkpoint = checkpoint_from_model(dense, dense_args)
        checkpoint["architecture"]["num_classes"] = 100

        with self.assertRaisesRegex(ValueError, "num_classes"):
            validate_pretrained_architecture(checkpoint, target_args)

    def test_rejects_incompatible_stage_delays(self):
        dense_args = model_args()
        target_args = model_args("ttfs", "ttfs")
        target_args.stage_delays = "0.04,0.02,0.01,0.01"
        checkpoint = checkpoint_from_model(
            make_model(dense_args),
            dense_args,
        )

        with self.assertRaisesRegex(ValueError, "stage_delays"):
            validate_pretrained_architecture(checkpoint, target_args)

    def test_rejects_unconsumed_source_parameter(self):
        dense_args = model_args()
        target_args = model_args("ttfs", "ttfs")
        dense = make_model(dense_args)
        checkpoint = checkpoint_from_model(dense, dense_args)
        checkpoint["model"] = dict(checkpoint["model"])
        checkpoint["model"]["unexpected.weight"] = torch.ones(1)

        with self.assertRaisesRegex(ValueError, "unused source parameters"):
            convert_dense_checkpoint_to_ttfs(
                make_model(target_args),
                checkpoint,
                target_args,
            )


if __name__ == "__main__":
    unittest.main()
