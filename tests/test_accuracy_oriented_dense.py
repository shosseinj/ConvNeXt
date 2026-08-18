import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from evaluate_accuracy_oriented_dense import integrity, make_views
from models.accuracy_convnext import AccuracyConvNeXt, DenseConvNeXtBlock, architecture_metadata
from train_accuracy_oriented_dense import (
    initialize_refinement,
    optimizer_state_to_device,
    resize_conv_kernel,
    scheduled_augmentation,
    transfer_imagenet_weights,
    update_overfitting_state,
)


class AccuracyOrientedDenseTests(unittest.TestCase):
    def test_restored_optimizer_state_moves_to_requested_device(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        optimizer_state_to_device(optimizer, torch.device("cpu"))

        self.assertTrue(all(
            not torch.is_tensor(value) or value.device.type == "cpu"
            for state in optimizer.state.values()
            for value in state.values()
        ))

    def test_model_is_dense_additive_and_forward_backward_is_finite(self):
        for classes in (10, 100):
            model = AccuracyConvNeXt(classes, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
            self.assertEqual(architecture_metadata(model)["residual_operator"], "sum")
            self.assertFalse(any("ttfs" in type(module).__name__.lower() for module in model.modules()))
            self.assertTrue(all(isinstance(block, DenseConvNeXtBlock) for stage in model.stages for block in stage))
            output = model(torch.randn(2, 3, 32, 32))
            output.square().mean().backward()
            self.assertEqual(tuple(output.shape), (2, classes))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))

    def test_exact_partial_transfer_excludes_convolutions_and_classifier(self):
        model = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        original_stem = model.downsample_layers[0][0].weight.detach().clone()
        source = {key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pth"
            torch.save({"model": source}, path)
            report = transfer_imagenet_weights(model, str(path))
        self.assertTrue(torch.equal(model.stages[0][0].pwconv1.weight, source["stages.0.0.pwconv1.weight"]))
        self.assertTrue(torch.equal(model.downsample_layers[0][0].weight, original_stem))
        self.assertNotIn("head.weight", report["transferred_keys"])
        self.assertFalse(any("dwconv" in key for key in report["transferred_keys"]))

    def test_interpolated_convolution_transfer_preserves_filter_norms(self):
        model = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        source = {key: value.detach().clone() for key, value in model.state_dict().items()}
        source["downsample_layers.0.0.weight"] = torch.randn(8, 3, 4, 4)
        source["stages.0.0.dwconv.weight"] = torch.randn(8, 1, 7, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pth"
            torch.save({"model": source}, path)
            report = transfer_imagenet_weights(
                model, str(path), interpolate_convolutions=True
            )
        self.assertIn(
            "downsample_layers.0.0.weight", report["interpolated_convolution_keys"]
        )
        self.assertIn(
            "stages.0.0.dwconv.weight", report["interpolated_convolution_keys"]
        )
        expected = resize_conv_kernel(
            source["stages.0.0.dwconv.weight"], (8, 1, 3, 3)
        )
        torch.testing.assert_close(
            model.state_dict()["stages.0.0.dwconv.weight"], expected
        )
        torch.testing.assert_close(
            source["stages.0.0.dwconv.weight"].flatten(1).norm(dim=1),
            expected.flatten(1).norm(dim=1),
        )

    def test_strict_round_trip_and_ten_views(self):
        model = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        clone = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        self.assertFalse(any(integrity(clone, model.state_dict()).values()))
        clone.load_state_dict(model.state_dict(), strict=True)
        views = make_views(torch.rand(2, 3, 32, 32), "flip_shift")
        self.assertEqual(len(views), 10)
        self.assertTrue(all(tuple(view.shape) == (2, 3, 32, 32) for view in views))

    def test_refinement_strictly_uses_ema_and_records_lineage(self):
        source = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        ema_state = {key: torch.full_like(value, 0.125) for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best_checkpoint.pth"
            torch.save({
                "model": source.state_dict(),
                "ema": ema_state,
                "architecture": architecture_metadata(source),
                "args": {"dataset": "cifar10"},
                "best_epoch": 17,
                "best_validation_accuracy": 95.0,
            }, checkpoint_path)
            target = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
            lineage = initialize_refinement(target, checkpoint_path, "cifar10")
        self.assertTrue(all(torch.equal(target.state_dict()[key], ema_state[key]) for key in ema_state))
        self.assertEqual(lineage["source_weights"], "ema")
        self.assertTrue(lineage["fresh_training_state"])

    def test_refinement60_transitions_and_one_way_restore(self):
        args = type("Args", (), {"augmentation_schedule": "refinement60"})()
        state = {
            "restored_phase": None,
            "overfit_counter": 0,
            "previous_validation_loss": 1.0,
        }
        self.assertEqual(scheduled_augmentation(args, 9, state)[0], "strong")
        self.assertEqual(scheduled_augmentation(args, 10, state)[0], "middle")
        self.assertEqual(scheduled_augmentation(args, 45, state)[0], "clean")
        for epoch, loss in ((10, 1.1), (11, 1.2), (12, 1.3)):
            update_overfitting_state(
                state, epoch, "middle",
                {"loss": loss, "accuracy": 74.0}, 75.0,
            )
        self.assertEqual(state["restored_phase"], "strong")
        self.assertEqual(scheduled_augmentation(args, 45, state)[0], "strong")
        update_overfitting_state(
            state, 46, "strong", {"loss": 0.9, "accuracy": 76.0}, 76.0
        )
        self.assertEqual(state["restored_phase"], "strong")

    def test_lowaug30_transition_and_restore(self):
        args = type("Args", (), {"augmentation_schedule": "lowaug30"})()
        state = {
            "restored_phase": None,
            "overfit_counter": 0,
            "previous_validation_loss": 0.20,
        }
        phase, augmentation = scheduled_augmentation(args, 9, state)
        self.assertEqual(phase, "light")
        self.assertTrue(augmentation["randaugment_enabled"])
        phase, augmentation = scheduled_augmentation(args, 10, state)
        self.assertEqual(phase, "clean_low")
        self.assertFalse(augmentation["randaugment_enabled"])
        for epoch, loss in ((10, 0.21), (11, 0.22), (12, 0.23)):
            update_overfitting_state(
                state, epoch, "clean_low",
                {"loss": loss, "accuracy": 94.5}, 94.94,
                accuracy_drop=0.2, restore_phase="light",
            )
        self.assertEqual(state["restored_phase"], "light")
        self.assertEqual(scheduled_augmentation(args, 20, state)[0], "light")
        update_overfitting_state(
            state, 20, "light", {"loss": 0.1, "accuracy": 95.0}, 95.0,
            accuracy_drop=0.2, restore_phase="light",
        )
        self.assertEqual(state["restored_phase"], "light")


if __name__ == "__main__":
    unittest.main()
