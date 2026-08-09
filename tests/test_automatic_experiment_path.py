import unittest

from train_continuous_ttfs_cifar10_32x32_stem1 import (
    resolve_experiment_identity,
)


class AutomaticExperimentPathTests(unittest.TestCase):
    def test_derives_full_name_and_output_path(self):
        experiment_name, output_dir = resolve_experiment_identity(
            dataset="cifar10",
            experiment_name="ttfs_dwconv_downsample",
            seed=8888,
            output_dir="",
        )

        self.assertEqual(
            experiment_name,
            "cifar10_ttfs_dwconv_downsample_seed8888",
        )
        self.assertEqual(
            output_dir,
            "results/cifar10/ttfs_dwconv_downsample/seed_8888",
        )

    def test_explicit_output_directory_remains_an_override(self):
        experiment_name, output_dir = resolve_experiment_identity(
            dataset="cifar10",
            experiment_name="ttfs_dwconv_downsample",
            seed=7,
            output_dir="custom/run",
        )

        self.assertEqual(
            experiment_name,
            "cifar10_ttfs_dwconv_downsample_seed7",
        )
        self.assertEqual(output_dir, "custom/run")

    def test_rejects_empty_or_path_like_experiment_names(self):
        invalid_names = ("", " ", ".", "..", "folder/name", "folder\\name", "C:\\run")

        for invalid_name in invalid_names:
            with self.subTest(experiment_name=invalid_name):
                with self.assertRaisesRegex(ValueError, "experiment_name"):
                    resolve_experiment_identity(
                        dataset="cifar10",
                        experiment_name=invalid_name,
                        seed=8888,
                        output_dir="",
                    )


if __name__ == "__main__":
    unittest.main()
