"""Strict single-view and 10-view TTA evaluation for fully-dense CIFAR models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.accuracy_convnext import AccuracyConvNeXt


NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), datasets.CIFAR10),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), datasets.CIFAR100),
}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate fully-dense CIFAR ConvNeXt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=tuple(NORMALIZATION), required=True)
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def make_views(images, mode):
    if mode == "none":
        return [images]
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    base = [
        padded[:, :, 1:33, 1:33], padded[:, :, 0:32, 1:33],
        padded[:, :, 2:34, 1:33], padded[:, :, 1:33, 0:32],
        padded[:, :, 1:33, 2:34],
    ]
    flipped = torch.flip(images, dims=(-1,))
    padded = F.pad(flipped, (1, 1, 1, 1), mode="reflect")
    return base + [
        padded[:, :, 1:33, 1:33], padded[:, :, 0:32, 1:33],
        padded[:, :, 2:34, 1:33], padded[:, :, 1:33, 0:32],
        padded[:, :, 1:33, 2:34],
    ]


def integrity(model, state):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(key for key in set(expected) & set(state) if expected[key].shape != state[key].shape)
    return {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatched}


@torch.inference_mode()
def evaluate(model, loader, device, mean, std, mode):
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    total = correct = 0
    loss_sum = 0.0
    started = time.time()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        views = make_views(images, mode)
        logits = None
        for view in views:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model((view - mean) / std)
            logits = output if logits is None else logits + output
        logits = logits / len(views)
        loss_sum += F.cross_entropy(logits.float(), labels).item() * labels.numel()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return {"mode": mode, "views_per_model": len(make_views(torch.zeros(1, 3, 32, 32), mode)),
            "forward_passes_per_sample": len(make_views(torch.zeros(1, 3, 32, 32), mode)),
            "samples": total, "correct": correct, "accuracy": 100.0 * correct / total,
            "loss": loss_sum / total, "seconds": time.time() - started}


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", {})
    if architecture.get("model_type") != "fully_dense_ann":
        raise RuntimeError("Checkpoint is not a fully-dense ANN model")
    expected_classes = 10 if args.dataset == "cifar10" else 100
    if int(architecture.get("num_classes", -1)) != expected_classes:
        raise RuntimeError("Checkpoint class count does not match --dataset")
    model = AccuracyConvNeXt(expected_classes, tuple(architecture["depths"]), tuple(architecture["dims"]),
                            int(architecture["kernel_size"]), drop_path_rate=0.0)
    state = checkpoint.get("ema")
    if not isinstance(state, dict):
        raise RuntimeError("Accuracy-oriented evaluation requires EMA weights")
    report_integrity = integrity(model, state)
    if any(report_integrity.values()):
        raise RuntimeError(f"Checkpoint integrity failure: {report_integrity}")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mean, std, dataset_class = NORMALIZATION[args.dataset]
    dataset = dataset_class(args.data_path, train=False, transform=transforms.ToTensor(), download=args.download)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    results = [evaluate(model, loader, device, mean, std, mode) for mode in ("none", "flip_shift")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset, "checkpoint": str(checkpoint_path.resolve()), "weights_source": "ema",
        "best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
        "best_epoch": checkpoint.get("best_epoch"), "architecture": architecture,
        "integrity": report_integrity, "results": results,
    }
    (output_dir / "tta_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "tta_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
