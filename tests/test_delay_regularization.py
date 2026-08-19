import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.convnext import Block, SpikingBlock
from train_continuous_ttfs_cifar10_32x32_stem1 import (
    effective_delay_regularization,
    run_epoch,
    validate_resume_training_configuration,
)


class _DelayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [nn.Sequential(SpikingBlock(Block(dim=2), init_delay=0.05))]
        )
        self.head = nn.Linear(2, 2)

    def forward(self, images):
        pooled = images.mean(dim=(2, 3))[:, :2]
        return self.head(pooled)


def epoch_args(weight):
    return Namespace(
        amp=False,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        t_min=0.0,
        t_max=1.0,
        grad_clip=0.0,
        print_freq=100,
        delay_regularization_weight=weight,
    )


class DelayRegularizationTests(unittest.TestCase):
    def test_zero_weight_preserves_classification_loss(self):
        model = _DelayModel()
        loader = DataLoader(
            TensorDataset(torch.rand(2, 3, 4, 4), torch.tensor([0, 1])),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        metrics = run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            epoch_args(0.0),
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
        )
        self.assertEqual(metrics["loss"], metrics["classification_loss"])
        self.assertEqual(metrics["total_loss"], metrics["classification_loss"])
        self.assertEqual(metrics["weighted_delay_regularization"], 0.0)

    def test_positive_penalty_pushes_raw_delays_down(self):
        model = _DelayModel()
        penalty = effective_delay_regularization(model)
        penalty.backward()
        block = model.stages[0][0]
        self.assertTrue(torch.all(block.D_mid.grad > 0))
        self.assertTrue(torch.all(block.D_out.grad > 0))

    def test_validation_excludes_delay_regularization(self):
        model = _DelayModel()
        loader = DataLoader(
            TensorDataset(torch.rand(2, 3, 4, 4), torch.tensor([0, 1])),
            batch_size=2,
        )
        metrics = run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            epoch_args(0.1),
        )
        self.assertEqual(metrics["loss"], metrics["classification_loss"])
        self.assertEqual(metrics["delay_regularization"], 0.0)
        self.assertEqual(metrics["weighted_delay_regularization"], 0.0)

    def test_resume_rejects_different_regularization_weight(self):
        checkpoint = {"args": {"delay_regularization_weight": 0.01}}
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_resume_training_configuration(
                checkpoint, Namespace(delay_regularization_weight=0.1)
            )

    def test_legacy_checkpoint_is_zero_weight(self):
        validate_resume_training_configuration(
            {"args": {}}, Namespace(delay_regularization_weight=0.0)
        )

    def test_negative_cli_weight_is_rejected(self):
        from train_continuous_ttfs_cifar10_32x32_stem1 import args_parser

        with patch.object(
            sys,
            "argv",
            [
                "trainer",
                "--experiment_name",
                "delay_regularization_test",
                "--delay_regularization_weight",
                "-0.1",
            ],
        ), self.assertRaises(SystemExit):
            args_parser()


if __name__ == "__main__":
    unittest.main()
