import tempfile
import unittest
from pathlib import Path

from Evaluation.summarize_finetuned_ttfs import (
    aggregate_dataset,
    parse_sparsity_report,
)


def report_text(offset=0.0):
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
            "Classification accuracy: 70.00%",
            "Measured TTFS points:     39",
            "Expected TTFS points:     39",
            "Activation sparsity:  30.00%",
        ]
    )


class FineTunedTTFSSummaryTests(unittest.TestCase):
    def test_parses_exactly_39_fully_ttfs_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activation_sparsity.md"
            path.write_text(report_text(), encoding="utf-8")

            report = parse_sparsity_report(path)

        self.assertEqual(len(report["layers"]), 39)
        self.assertEqual(report["accuracy"], 70.0)
        self.assertEqual(report["global_sparsity"], 30.0)

    def test_rejects_dense_convolution_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activation_sparsity.md"
            path.write_text(
                report_text().replace(
                    "depthwise convolution mode: ttfs",
                    "depthwise convolution mode: dense",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "depthwise.*TTFS"):
                parse_sparsity_report(path)

    def test_aggregates_layer_mean_and_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, offset in ((1, 0.0), (2, 2.0), (3, 4.0)):
                path = root / f"seed_{seed}.md"
                path.write_text(report_text(offset), encoding="utf-8")
                paths.append(path)

            summary = aggregate_dataset(paths)

        first = summary["layers"]["downsample_layers.1.0"]
        self.assertEqual(first["mean"], 2.0)
        self.assertEqual(first["sample_std"], 2.0)


if __name__ == "__main__":
    unittest.main()
