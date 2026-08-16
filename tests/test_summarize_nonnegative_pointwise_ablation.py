import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Evaluation.summarize_nonnegative_pointwise_ablation import (
    summarize_condition,
)


def report_text(accuracy, sparsity, offset=0.0):
    rows = []
    for index in range(39):
        if index < 3:
            name = f"downsample_layers.{index + 1}.0"
            kind = "downsample"
        elif index < 15:
            name = f"stages.0.{index - 3}.dwconv"
            kind = "dwconv"
        elif index < 27:
            name = f"stages.0.{index - 15}.pw1_ttfs"
            kind = "pw1"
        else:
            name = f"stages.0.{index - 27}.pw2_ttfs"
            kind = "pw2"
        rows.append(f"{name} {kind} 1 10 {index + offset:.2f}%")
    return "\n".join(
        [
            "Detected depthwise convolution mode: ttfs (metadata)",
            "Detected downsampling convolution mode: ttfs (metadata)",
            *rows,
            f"Classification accuracy: {accuracy:.2f}%",
            "Measured TTFS points: 39",
            "Expected TTFS points: 39",
            f"Activation sparsity: {sparsity:.2f}%",
        ]
    )


class NonnegativePointwiseSummaryTests(unittest.TestCase):
    def test_script_help_runs_from_repository_root(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "Evaluation"
            / "summarize_nonnegative_pointwise_ablation.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_summarizes_test_validation_global_and_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_paths = []
            summary_paths = []
            for seed, offset in ((1, 0.0), (2, 2.0), (3, 4.0)):
                report = root / f"report_{seed}.md"
                report.write_text(
                    report_text(70.0 + offset, 30.0 + offset, offset),
                    encoding="utf-8",
                )
                summary = root / f"summary_{seed}.json"
                summary.write_text(
                    json.dumps({"best_validation_accuracy": 75.0 + offset}),
                    encoding="utf-8",
                )
                report_paths.append(report)
                summary_paths.append(summary)

            result = summarize_condition(report_paths, summary_paths)

        self.assertEqual(result["test_accuracy_mean"], 72.0)
        self.assertEqual(result["test_accuracy_sample_std"], 2.0)
        self.assertEqual(result["best_validation_accuracy_mean"], 77.0)
        self.assertEqual(result["global_sparsity_mean"], 32.0)
        self.assertEqual(
            result["layers"]["downsample_layers.1.0"]["sample_std"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
