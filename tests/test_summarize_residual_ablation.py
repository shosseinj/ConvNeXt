import json
import tempfile
import unittest
from pathlib import Path

import torch

from Evaluation.summarize_residual_ablation import build_report


OPERATORS = ("min", "mean", "learnable_gate")
SEEDS = (42, 6543, 7777)


def write_run(root, operator, seed, recorded_test_offset=0.0):
    if operator == "min":
        directory = root / "cifar100" / "fully_ttfs" / f"seed_{seed}"
    else:
        directory = (
            root / "cifar100" / "ablation_residual" / operator / f"seed_{seed}"
        )
    directory.mkdir(parents=True)
    (directory / "training_summary.json").write_text(
        json.dumps({
            "best_validation_accuracy": 70.0 + seed / 10000,
            "test_metrics": {
                "accuracy": 69.0 + seed / 10000 + recorded_test_offset
            },
        }),
        encoding="utf-8",
    )
    lines = [
        f"Detected residual operator: {operator}",
        "Measured TTFS points:     39",
        "Expected TTFS points:     39",
        f"Classification accuracy: {69.0 + seed / 10000:.4f}%",
        "Activation sparsity:  32.00%",
        "Theoretical SynOps:   4,641 per sample",
    ]
    for index in range(39):
        lines.append(
            f"stages.0.{index}.pw1_ttfs pw1 1 2 50.00% {100 + index:,}"
        )
    (directory / "activation_sparsity.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    state = {}
    if operator == "learnable_gate":
        for index in range(12):
            state[f"stages.0.{index}.raw_residual_gate"] = torch.zeros(2)
    torch.save({"model": state}, directory / "best_checkpoint.pth")
    tta_directory = directory / "evaluation_tta"
    tta_directory.mkdir()
    checkpoint_path = (directory / "best_checkpoint.pth").resolve()
    (tta_directory / "tta_evaluation.json").write_text(
        json.dumps({
            "dataset": "CIFAR-100 test",
            "checkpoints": [{
                "path": str(checkpoint_path),
                "integrity": {
                    "missing_keys": [], "unexpected_keys": [],
                    "shape_mismatches": [],
                },
            }],
            "architecture": {"residual_operator": operator},
            "results": [
                {
                    "mode": "none", "models": 1, "views_per_model": 1,
                    "forward_passes_per_sample": 1, "samples": 10000,
                    "accuracy": 69.0 + seed / 10000,
                },
                {
                    "mode": "flip_shift", "models": 1,
                    "views_per_model": 10, "forward_passes_per_sample": 10,
                    "samples": 10000, "accuracy": 70.0 + seed / 10000,
                },
            ],
        }),
        encoding="utf-8",
    )


class ResidualAblationSummaryTests(unittest.TestCase):
    def test_report_uses_uniform_evaluator_accuracy_with_small_cuda_variation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for operator in OPERATORS:
                for seed in SEEDS:
                    write_run(
                        root,
                        operator,
                        seed,
                        recorded_test_offset=0.12 if seed == 42 else 0.0,
                    )

            report = build_report(root)

            self.assertIn("70.48", report)
            self.assertIn("Standard test accuracy", report)

    def test_report_requires_all_nine_runs_and_labels_normalized_sum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for operator in OPERATORS:
                for seed in SEEDS:
                    write_run(root, operator, seed)

            report = build_report(root)

            self.assertIn("Normalized Sum (Mean)", report)
            self.assertIn("Learnable Gate", report)
            self.assertIn("70.48", report)
            self.assertIn("SynOps/sample", report)
            self.assertIn("Reviewer 1, Comment 4", report)
            self.assertIn("0.5000", report)
            self.assertIn("10-view TTA test accuracy", report)

    def test_report_refuses_partial_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "min", 42)

            with self.assertRaisesRegex(FileNotFoundError, "nine-run"):
                build_report(root)

    def test_report_rejects_non_ten_view_tta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for operator in OPERATORS:
                for seed in SEEDS:
                    write_run(root, operator, seed)
            path = (
                root / "cifar100" / "fully_ttfs" / "seed_42"
                / "evaluation_tta" / "tta_evaluation.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["results"][1]["views_per_model"] = 9
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "10 views"):
                build_report(root)


if __name__ == "__main__":
    unittest.main()
