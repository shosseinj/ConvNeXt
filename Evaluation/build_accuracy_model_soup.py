"""Build a validation-selected single-model EMA soup for dense CIFAR ConvNeXt."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.accuracy_convnext import AccuracyConvNeXt


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    parser = argparse.ArgumentParser("Build CIFAR-10 validation-selected EMA model soup")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoint) != 3:
        parser.error("Exactly three --checkpoint arguments are required")
    return args


def candidate_weights(units=10):
    return [
        (first / units, second / units, third / units)
        for first in range(units + 1)
        for second in range(units - first + 1)
        for third in (units - first - second,)
    ]


def average_states(states, weights):
    if len(states) != len(weights):
        raise ValueError("State and weight counts differ")
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("Soup weights must be non-negative and sum to one")
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("Soup state dictionaries have different keys")
    averaged = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Soup tensor shape mismatch: {key}")
        if tensors[0].is_floating_point():
            value = torch.zeros_like(tensors[0])
            for weight, tensor in zip(weights, tensors):
                value.add_(tensor, alpha=float(weight))
            averaged[key] = value
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating soup tensors differ: {key}")
            averaged[key] = tensors[0].clone()
    return averaged


def load_sources(paths, expected_split_seed):
    checkpoints, states, reports = [], [], []
    architecture = None
    for name in paths:
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Soup checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        current_architecture = checkpoint.get("architecture")
        saved_args = checkpoint.get("args", {})
        if not isinstance(current_architecture, dict) or current_architecture.get("model_type") != "fully_dense_ann":
            raise RuntimeError(f"Not a fully-dense ANN checkpoint: {path}")
        if int(current_architecture.get("num_classes", -1)) != 10:
            raise RuntimeError(f"Not a CIFAR-10 checkpoint: {path}")
        if saved_args.get("dataset") != "cifar10":
            raise RuntimeError(f"Checkpoint dataset metadata is not cifar10: {path}")
        if int(saved_args.get("split_seed", -1)) != expected_split_seed:
            raise RuntimeError(f"Checkpoint split_seed mismatch: {path}")
        if architecture is None:
            architecture = current_architecture
        elif current_architecture != architecture:
            raise RuntimeError(f"Soup checkpoint architecture mismatch: {path}")
        state = checkpoint.get("ema")
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint has no EMA state: {path}")
        checkpoints.append(checkpoint)
        states.append(state)
        reports.append({
            "path": str(path.resolve()),
            "best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
            "best_epoch": checkpoint.get("best_epoch"),
            "weights_source": "ema",
        })
    # This also validates keys, shapes, and immutable buffers before the search.
    average_states(states, (1.0, 0.0, 0.0))
    return checkpoints, states, architecture, reports


def build_validation_loader(data_path, split_seed, batch_size, num_workers, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(data_path, train=True, transform=transform, download=download)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(split_seed)).tolist()
    validation = Subset(dataset, indices[:5000])
    return DataLoader(
        validation, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def evaluate_validation(model, loader, device):
    total = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        loss_sum += F.cross_entropy(logits.float(), labels).item() * labels.numel()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return {"accuracy": 100.0 * correct / total, "loss": loss_sum / total, "correct": correct, "samples": total}


def main():
    args = parse_args()
    checkpoints, states, architecture, sources = load_sources(args.checkpoint, args.split_seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = AccuracyConvNeXt(
        10, tuple(architecture["depths"]), tuple(architecture["dims"]),
        int(architecture["kernel_size"]), drop_path_rate=0.0,
    ).to(device).eval()
    loader = build_validation_loader(
        args.data_path, args.split_seed, args.batch_size,
        args.num_workers, args.download,
    )
    candidates = []
    best = None
    best_state = None
    for index, weights in enumerate(candidate_weights()):
        state = average_states(states, weights)
        model.load_state_dict(state, strict=True)
        metrics = evaluate_validation(model, loader, device)
        candidate = {"index": index, "weights": list(weights), **metrics}
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
        if best is None or (metrics["accuracy"], -metrics["loss"]) > (best["accuracy"], -best["loss"]):
            best = candidate
            best_state = {key: value.detach().cpu().clone() for key, value in state.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    soup = {
        "model": best_state,
        "ema": best_state,
        "architecture": architecture,
        "args": {"dataset": "cifar10", "split_seed": args.split_seed, "model_soup": True},
        "best_validation_accuracy": best["accuracy"],
        "best_epoch": None,
        "model_soup": {
            "selection_dataset": "CIFAR-10 training-set validation partition",
            "selection_samples": 5000,
            "test_data_used_for_selection": False,
            "grid_step": 0.1,
            "sources": sources,
            "selected": best,
        },
    }
    checkpoint_path = output_dir / "best_checkpoint.pth"
    torch.save(soup, checkpoint_path)
    report = {"checkpoint": str(checkpoint_path.resolve()), "sources": sources, "selected": best, "candidates": candidates}
    (output_dir / "model_soup_search.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "model_soup_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "weight_v2", "weight_v3", "weight_v4", "accuracy", "loss", "correct", "samples"])
        for row in candidates:
            writer.writerow([row["index"], *row["weights"], row["accuracy"], row["loss"], row["correct"], row["samples"]])
    print(json.dumps(report | {"candidates": f"{len(candidates)} validation-only candidates"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
