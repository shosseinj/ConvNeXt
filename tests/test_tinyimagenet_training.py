import unittest
from argparse import Namespace
from pathlib import Path

import torch

from train_continuous_ttfs_cifar10_32x32_stem1 import (
    architecture_metadata,
    build_loaders,
    dataset_metadata,
    dataset_name,
    make_model,
)
from Evaluation.evaluate_sparsity import build_loader as build_evaluation_loader


TINY_ROOT = Path(
    r"C:\Users\jafari.h\Desktop\ai_project\cifar_data\tiny-imagenet-200"
)


class TinyImageNetTrainingTests(unittest.TestCase):
    def test_dataset_alias_and_metadata(self):
        self.assertEqual(dataset_name("tiny-imagenet"), "tinyimagenet")
        self.assertEqual(dataset_name("tiny_imagenet"), "tinyimagenet")

        metadata = dataset_metadata("tinyimagenet")
        self.assertEqual(metadata["display_name"], "Tiny ImageNet")
        self.assertEqual(metadata["num_classes"], 200)
        self.assertEqual(metadata["input_resolution"], 64)
        self.assertEqual(metadata["default_val_size"], 10000)

    @unittest.skipUnless(TINY_ROOT.is_dir(), "Tiny ImageNet dataset is unavailable")
    def test_official_layout_builds_90k_10k_10k_loaders(self):
        args = Namespace(
            dataset="tinyimagenet",
            data_path=str(TINY_ROOT),
            download=False,
            randaugment=False,
            randaugment_num_ops=2,
            randaugment_magnitude=9,
            random_erasing=0.0,
            seed=42,
            val_size=10000,
            batch_size=4,
            num_workers=0,
        )

        train_loader, validation_loader, test_loader = build_loaders(args)

        self.assertEqual(len(train_loader.dataset), 90000)
        self.assertEqual(len(validation_loader.dataset), 10000)
        self.assertEqual(len(test_loader.dataset), 10000)
        images, labels = next(iter(test_loader))
        self.assertEqual(tuple(images.shape), (4, 3, 64, 64))
        self.assertTrue(torch.all((0 <= labels) & (labels < 200)))

    def test_model_and_metadata_use_64px_and_200_classes(self):
        args = Namespace(
            num_classes=200,
            dims=(8, 16, 32, 64),
            depths=(1, 1, 1, 1),
            dw_kernel_size=3,
            drop_path=0.0,
            t_min=0.0,
            t_max=1.0,
            head_dropout=0.0,
            spike_dropout=0.0,
            pw2_mode="ttfs",
            ttfs_norm_mode="score_layernorm",
            final_score_norm=True,
            dwconv_mode="dense",
            downsample_mode="ttfs",
            force_positive_weights=False,
            init_delay=0.0,
            stage_delays="0.05,0.02,0.01,0.01",
            input_resolution=64,
        )

        model = make_model(args)
        logits = model(torch.rand(1, 3, 64, 64))
        metadata = architecture_metadata(args)

        self.assertEqual(tuple(logits.shape), (1, 200))
        self.assertEqual(metadata["num_classes"], 200)
        self.assertEqual(metadata["input_resolution"], [64, 64])

    @unittest.skipUnless(TINY_ROOT.is_dir(), "Tiny ImageNet dataset is unavailable")
    def test_evaluator_uses_all_200_official_validation_labels(self):
        dataset, loader = build_evaluation_loader(
            dataset_name="tinyimagenet",
            data_path=str(TINY_ROOT),
            batch_size=4,
            workers=0,
        )

        self.assertEqual(len(dataset), 10000)
        self.assertEqual(len({target for _, target in dataset.samples}), 200)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (4, 3, 64, 64))
        self.assertTrue(torch.all((0 <= labels) & (labels < 200)))


if __name__ == "__main__":
    unittest.main()
