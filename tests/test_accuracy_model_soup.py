import tempfile
import unittest
from pathlib import Path

import torch

from Evaluation.build_accuracy_model_soup import (
    average_states,
    candidate_weights,
    load_sources,
)


class AccuracyModelSoupTests(unittest.TestCase):
    def test_candidate_grid_is_complete_and_normalized(self):
        candidates = candidate_weights()
        self.assertEqual(len(candidates), 66)
        self.assertIn((1.0, 0.0, 0.0), candidates)
        self.assertIn((0.0, 1.0, 0.0), candidates)
        self.assertIn((0.0, 0.0, 1.0), candidates)
        self.assertTrue(all(abs(sum(weights) - 1.0) < 1e-8 for weights in candidates))

    def test_weighted_state_arithmetic_is_exact(self):
        states = [
            {"weight": torch.tensor([0.0, 2.0]), "counter": torch.tensor(4)},
            {"weight": torch.tensor([2.0, 4.0]), "counter": torch.tensor(4)},
            {"weight": torch.tensor([4.0, 6.0]), "counter": torch.tensor(4)},
        ]
        result = average_states(states, (0.2, 0.3, 0.5))
        self.assertTrue(torch.allclose(result["weight"], torch.tensor([2.6, 4.6])))
        self.assertEqual(result["counter"].item(), 4)

    def test_source_validation_rejects_split_mismatch(self):
        architecture = {
            "model_type": "fully_dense_ann",
            "num_classes": 10,
            "depths": [2, 2, 6, 2],
            "dims": [96, 192, 384, 768],
            "kernel_size": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, split_seed in enumerate((2026, 2026, 7)):
                path = Path(directory) / f"source_{index}.pth"
                torch.save({
                    "architecture": architecture,
                    "args": {"dataset": "cifar10", "split_seed": split_seed},
                    "ema": {"weight": torch.tensor([float(index)])},
                }, path)
                paths.append(path)
            with self.assertRaisesRegex(RuntimeError, "split_seed mismatch"):
                load_sources(paths, 2026)


if __name__ == "__main__":
    unittest.main()
