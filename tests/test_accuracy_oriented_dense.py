import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from evaluate_accuracy_oriented_dense import integrity, make_views
from models.accuracy_convnext import AccuracyConvNeXt, DenseConvNeXtBlock, architecture_metadata
from train_accuracy_oriented_dense import transfer_imagenet_weights


class AccuracyOrientedDenseTests(unittest.TestCase):
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

    def test_strict_round_trip_and_ten_views(self):
        model = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        clone = AccuracyConvNeXt(10, depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))
        self.assertFalse(any(integrity(clone, model.state_dict()).values()))
        clone.load_state_dict(model.state_dict(), strict=True)
        views = make_views(torch.rand(2, 3, 32, 32), "flip_shift")
        self.assertEqual(len(views), 10)
        self.assertTrue(all(tuple(view.shape) == (2, 3, 32, 32) for view in views))


if __name__ == "__main__":
    unittest.main()
