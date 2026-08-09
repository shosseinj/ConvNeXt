import json
import tempfile
import unittest
from pathlib import Path

from summarize_validation_accuracy import (
    calculate_summary,
    extract_best_validation_accuracy,
    format_summary,
    load_seed_accuracies,
)


class ExtractBestValidationAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.seed_directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_json(self, relative_path, value):
        path = self.seed_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_reads_training_summary_first(self):
        self.write_json("training_summary.json", {"best_validation_accuracy": 94.48})
        self.write_json(
            "experiment_report.json",
            {"results": {"best_validation_accuracy": 90.0}},
        )

        self.assertEqual(extract_best_validation_accuracy(self.seed_directory), 94.48)

    def test_reads_nested_experiment_report(self):
        self.write_json(
            "experiment_report.json",
            {"results": {"best_validation_accuracy": 94.50}},
        )

        self.assertEqual(extract_best_validation_accuracy(self.seed_directory), 94.50)

    def test_reads_highest_value_from_train_log(self):
        rows = [
            {"epoch": 0, "best_validation_accuracy": 90.0},
            {"epoch": 1, "best_validation_accuracy": 94.4},
        ]
        (self.seed_directory / "train_log.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        self.assertEqual(extract_best_validation_accuracy(self.seed_directory), 94.4)

    def test_reads_checkpoint_metadata_from_nested_tta_report(self):
        self.write_json(
            "evaluation/run/tta_evaluation.json",
            {"checkpoints": [{"best_validation_accuracy": 94.4}]},
        )

        self.assertEqual(extract_best_validation_accuracy(self.seed_directory), 94.4)

    def test_missing_accuracy_has_actionable_error(self):
        self.write_json("training_summary.json", {"best_epoch": 10})

        with self.assertRaisesRegex(ValueError, "best_validation_accuracy"):
            extract_best_validation_accuracy(self.seed_directory)

    def test_missing_seed_directory_has_actionable_error(self):
        missing = self.seed_directory / "seed_999"

        with self.assertRaisesRegex(FileNotFoundError, "Seed directory not found"):
            extract_best_validation_accuracy(missing)


class RealResultsTests(unittest.TestCase):
    def test_loads_real_seed_accuracies(self):
        base_directory = Path(
            "results/cifar10/clean_finetune_from_94_36"
        )

        values = load_seed_accuracies(base_directory, (42, 6543, 7777))

        self.assertEqual(values[42], 94.4)
        self.assertGreaterEqual(values[6543], 94.5)
        self.assertEqual(values[7777], 94.48)


class SummaryCalculationTests(unittest.TestCase):
    def test_uses_sample_standard_deviation_and_formats_two_decimals(self):
        values = {42: 94.4, 6543: 94.5, 7777: 94.48}

        mean_accuracy, sample_std = calculate_summary(values)
        output = format_summary(
            "CIFAR-10", values, mean_accuracy, sample_std
        )

        self.assertAlmostEqual(mean_accuracy, 94.46)
        self.assertAlmostEqual(sample_std, 0.05291502622129385)
        self.assertEqual(output.splitlines()[0], "Dataset: CIFAR-10")
        self.assertEqual(
            output.splitlines()[-1],
            "Final: 94.46 ± 0.05",
        )


if __name__ == "__main__":
    unittest.main()
