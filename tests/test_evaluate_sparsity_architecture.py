import gc
import io
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


if "torchvision" not in sys.modules:
    torchvision = types.ModuleType("torchvision")
    torchvision.datasets = types.ModuleType("torchvision.datasets")
    torchvision.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.datasets"] = torchvision.datasets
    sys.modules["torchvision.transforms"] = torchvision.transforms

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

from Evaluation.evaluate_sparsity import (  # noqa: E402
    build_model,
    evaluate_sparsity,
    find_checkpoint,
    get_checkpoint_convolution_modes,
    main,
    markdown_output,
    parse_args,
    report_path_for_checkpoint,
)


class CheckpointConvolutionModeTests(unittest.TestCase):
    def test_builds_model_for_every_checkpoint_mode_combination(self):
        args = Namespace(
            dataset="cifar100",
            dw_kernel_size=3,
            cifar_stem=True,
        )

        for dwconv_mode, downsample_mode in (
            ("ttfs", "ttfs"),
            ("dense", "dense"),
            ("dense", "ttfs"),
            ("ttfs", "dense"),
        ):
            with self.subTest(
                dwconv_mode=dwconv_mode,
                downsample_mode=downsample_mode,
            ):
                checkpoint = {
                    "architecture": {
                        "dwconv_mode": dwconv_mode,
                        "downsample_mode": downsample_mode,
                    }
                }
                modes = get_checkpoint_convolution_modes(checkpoint)
                model = build_model(args, modes)

                self.assertEqual(model.dwconv_mode, dwconv_mode)
                self.assertEqual(model.downsample_mode, downsample_mode)
                downsample = model.downsample_layers[1][0]
                if downsample_mode == "ttfs":
                    downsample = downsample.conv
                self.assertEqual(downsample.kernel_size, (3, 3))

                del model
                gc.collect()

    def test_infers_missing_modes_from_state_dict(self):
        cases = (
            (
                "dense",
                "dense",
                {
                    "stages.0.0.dwconv.weight": torch.empty(1),
                    "downsample_layers.1.0.weight": torch.empty(1),
                },
            ),
            (
                "ttfs",
                "ttfs",
                {
                    "stages.0.0.dwconv.D_conv": torch.empty(1),
                    "downsample_layers.1.0.D_conv": torch.empty(1),
                },
            ),
            (
                "dense",
                "ttfs",
                {
                    "stages.0.0.dwconv.weight": torch.empty(1),
                    "downsample_layers.1.0.D_conv": torch.empty(1),
                },
            ),
            (
                "ttfs",
                "dense",
                {
                    "stages.0.0.dwconv.D_conv": torch.empty(1),
                    "downsample_layers.1.0.weight": torch.empty(1),
                },
            ),
        )

        for expected_dwconv, expected_downsample, state_dict in cases:
            with self.subTest(
                dwconv_mode=expected_dwconv,
                downsample_mode=expected_downsample,
            ):
                modes = get_checkpoint_convolution_modes(
                    {"architecture": {}},
                    state_dict,
                )
                self.assertEqual(modes["dwconv_mode"], expected_dwconv)
                self.assertEqual(
                    modes["downsample_mode"],
                    expected_downsample,
                )

    def test_metadata_takes_precedence_over_state_dict_markers(self):
        modes = get_checkpoint_convolution_modes(
            {
                "architecture": {
                    "dwconv_mode": "dense",
                    "downsample_mode": "dense",
                }
            },
            {
                "stages.0.0.dwconv.D_conv": torch.empty(1),
                "downsample_layers.1.0.D_conv": torch.empty(1),
            },
        )

        self.assertEqual(
            modes,
            {"dwconv_mode": "dense", "downsample_mode": "dense"},
        )

    def test_rejects_ambiguous_or_unrecognized_state_dict_markers(self):
        invalid_state_dicts = (
            {},
            {
                "stages.0.0.dwconv.weight": torch.empty(1),
                "stages.0.0.dwconv.D_conv": torch.empty(1),
                "downsample_layers.1.0.weight": torch.empty(1),
            },
        )

        for state_dict in invalid_state_dicts:
            with self.subTest(state_dict=tuple(state_dict)):
                with self.assertRaisesRegex(ValueError, "infer"):
                    get_checkpoint_convolution_modes(
                        {"architecture": {}},
                        state_dict,
                    )

    def test_rejects_invalid_mode_metadata(self):
        for field in ("dwconv_mode", "downsample_mode"):
            checkpoint = {
                "architecture": {
                    "dwconv_mode": "ttfs",
                    "downsample_mode": "dense",
                }
            }
            checkpoint["architecture"][field] = "automatic"

            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    get_checkpoint_convolution_modes(checkpoint)

    def test_collects_all_39_ttfs_sparsity_points(self):
        args = Namespace(
            dataset="cifar100",
            dw_kernel_size=3,
            cifar_stem=True,
        )
        model = build_model(
            args,
            {
                "dwconv_mode": "ttfs",
                "downsample_mode": "ttfs",
            },
        )
        loader = [
            (
                torch.rand(1, 3, 32, 32),
                torch.zeros(1, dtype=torch.long),
            )
        ]

        counter, _, expected_points = evaluate_sparsity(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            t_min=0.0,
            t_max=1.0,
        )

        self.assertEqual(len(counter.data), 39)
        self.assertEqual(expected_points, 39)
        self.assertEqual(
            sum(name.endswith(".pw1_ttfs") for name in counter.data),
            12,
        )
        self.assertEqual(
            sum(name.endswith(".pw2_ttfs") for name in counter.data),
            12,
        )
        blocks = [
            block
            for stage in model.stages
            for block in stage
        ]
        self.assertTrue(
            all(not block.t_mid_spike.requires_grad for block in blocks)
        )
        self.assertTrue(
            all(not block.t_out_spike.requires_grad for block in blocks)
        )
        self.assertFalse(
            any(
                "t_mid_spike" in key or "t_out_spike" in key
                for key in model.state_dict()
            )
        )

    def test_dense_convolutions_collect_24_expected_ttfs_points(self):
        args = Namespace(
            dataset="cifar10",
            dw_kernel_size=3,
            cifar_stem=True,
        )
        model = build_model(
            args,
            {
                "dwconv_mode": "dense",
                "downsample_mode": "dense",
            },
        )
        loader = [
            (
                torch.rand(1, 3, 32, 32),
                torch.zeros(1, dtype=torch.long),
            )
        ]

        counter, _, expected_points = evaluate_sparsity(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            t_min=0.0,
            t_max=1.0,
        )

        self.assertEqual(len(counter.data), 24)
        self.assertEqual(expected_points, 24)


class MarkdownReportTests(unittest.TestCase):
    def test_report_path_is_beside_selected_checkpoint(self):
        checkpoint = Path("results/cifar100/experiment/seed_42/best_checkpoint.pth")

        self.assertEqual(
            report_path_for_checkpoint(checkpoint),
            checkpoint.parent / "activation_sparsity.md",
        )

    def test_markdown_output_mirrors_terminal_and_overwrites_old_report(self):
        terminal = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "activation_sparsity.md"
            report_path.write_text("stale report", encoding="utf-8")

            with markdown_output(report_path, terminal):
                print("Dataset: cifar100")
                print("TTFS layers/points: 39")

            self.assertEqual(
                terminal.getvalue(),
                "Dataset: cifar100\nTTFS layers/points: 39\n",
            )
            self.assertEqual(
                report_path.read_text(encoding="utf-8"),
                "# Activation Sparsity Evaluation\n\n"
                "```text\n"
                "Dataset: cifar100\n"
                "TTFS layers/points: 39\n"
                "```\n",
            )

    def test_main_saves_report_in_selected_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "seed_42" / "best_checkpoint.pth"
            checkpoint.parent.mkdir()
            checkpoint.touch()

            with (
                patch(
                    "Evaluation.evaluate_sparsity.parse_args",
                    return_value=Namespace(checkpoint=str(checkpoint)),
                ),
                patch(
                    "Evaluation.evaluate_sparsity.find_checkpoint",
                    return_value=checkpoint,
                ),
                patch(
                    "Evaluation.evaluate_sparsity.run_evaluation",
                    side_effect=lambda args, path: print("TTFS layers/points: 39"),
                ),
            ):
                main()

            report = checkpoint.parent / "activation_sparsity.md"
            self.assertTrue(report.is_file())
            self.assertIn(
                "TTFS layers/points: 39",
                report.read_text(encoding="utf-8"),
            )


class DefaultArgumentTests(unittest.TestCase):
    def test_default_checkpoint_resolves_to_existing_file(self):
        with patch.object(sys, "argv", ["evaluate_sparsity.py"]):
            args = parse_args()

        checkpoint = find_checkpoint(args.checkpoint)

        self.assertTrue(checkpoint.is_file())
        self.assertEqual(checkpoint.name, "best_checkpoint.pth")
        self.assertEqual(checkpoint.parent.name, "seed_42")


if __name__ == "__main__":
    unittest.main()
