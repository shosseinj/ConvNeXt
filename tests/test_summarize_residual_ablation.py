import json
import tempfile
import unittest
from pathlib import Path

import torch

from Evaluation.summarize_residual_ablation import build_report


OPERATORS = ("min", "mean", "learnable_gate")
SEEDS = (42, 6543, 7777)


def write_run(root, operator, seed):
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
            "test_metrics": {"accuracy": 69.0 + seed / 10000},
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


class ResidualAblationSummaryTests(unittest.TestCase):
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
            self.assertIn("Theoretical SynOps", report)
            self.assertIn("Reviewer 1, Comment 4", report)
            self.assertIn("0.5000", report)

    def test_report_refuses_partial_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "min", 42)

            with self.assertRaisesRegex(FileNotFoundError, "nine-run"):
                build_report(root)


if __name__ == "__main__":
    unittest.main()
