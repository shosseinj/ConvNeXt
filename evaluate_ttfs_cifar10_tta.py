#!/usr/bin/env python3
"""Evaluate continuous TTFS ConvNeXt on CIFAR-10/100 with EMA and TTA."""

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

from models.convnext import ConvNeXtSpiking


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate continuous TTFS ConvNeXt on CIFAR-10/100 with TTA"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint path; repeat this argument for a model ensemble.",
    )
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument(
        "--dataset",
        choices=("auto", "cifar10", "cifar100"),
        default="auto",
        help="Infer from checkpoint metadata by default.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument(
        "--save_confusion_matrix",
        type=str2bool,
        default=True,
        help="Save confusion-matrix CSV/PNG files and values in JSON.",
    )
    parser.add_argument(
        "--weights_source", choices=("auto", "ema", "model"), default="auto"
    )
    parser.add_argument(
        "--tta_modes",
        default="none,flip,flip_shift",
        help="Comma-separated subset of: none, flip, flip_shift",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    modes = tuple(item.strip() for item in args.tta_modes.split(",") if item.strip())
    allowed = {"none", "flip", "flip_shift"}
    if not modes or any(mode not in allowed for mode in modes):
        parser.error("--tta_modes must contain only none, flip, flip_shift")
    args.tta_modes = tuple(dict.fromkeys(modes))
    return args


def architecture_from_checkpoint(checkpoint):
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError("Checkpoint is missing architecture metadata")
    normalized = dict(architecture)
    residual_operator = normalized.get("residual_operator")
    if residual_operator not in {"min", "mean", "learnable_gate"}:
        raise RuntimeError(
            "Checkpoint architecture residual_operator must be one of "
            "'min', 'mean', or 'learnable_gate'"
        )
    normalized.setdefault("final_score_norm", False)
    normalized.setdefault("ttfs_grn", False)
    normalized.setdefault(
        "force_positive_pointwise_weights",
        bool((checkpoint.get("args") or {}).get(
            "force_positive_pointwise_weights", False
        )),
    )
    return normalized


def build_model(checkpoint):
    architecture = architecture_from_checkpoint(checkpoint)
    saved_args = checkpoint.get("args", {})
    dims = tuple(int(value) for value in architecture["dims"])
    depths = tuple(int(value) for value in architecture["depths"])
    num_classes = int(
        architecture.get("num_classes", saved_args.get("num_classes", 10))
    )
    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        dw_kernel_size=int(architecture["depthwise_kernel_size"]),
        cifar_stem=True,
        downsample_kernel_size=int(
            architecture.get("downsample_kernel_size", 3)
        ),
        drop_path_rate=0.0,
        t_min=float(saved_args.get("t_min", 0.0)),
        t_max=float(saved_args.get("t_max", 1.0)),
        head_dropout=0.0,
        spike_dropout=0.0,
        pw2_mode=architecture.get("pw2_mode", "ttfs"),
        ttfs_norm_mode=architecture.get("ttfs_norm_mode", "none"),
        final_score_norm=bool(architecture.get("final_score_norm", False)),
        dwconv_mode=architecture.get("dwconv_mode", "dense"),
        downsample_mode=architecture.get("downsample_mode", "dense"),
        residual_operator=architecture["residual_operator"],
        force_positive_weights=bool(
            saved_args.get("force_positive_weights", False)
        ),
        force_positive_pointwise_weights=bool(
            architecture["force_positive_pointwise_weights"]
        ),
        init_delay=float(saved_args.get("init_delay", 0.0)),
        stage_delays=tuple(float(value) for value in architecture["stage_delays"]),
    )
    architecture["num_classes"] = num_classes
    return model, architecture


def selected_state(checkpoint, requested_source):
    if requested_source == "ema":
        if checkpoint.get("ema") is None:
            raise RuntimeError("--weights_source ema requested, but checkpoint has no EMA")
        return checkpoint["ema"], "ema"
    if requested_source == "model":
        return checkpoint["model"], "model"
    if checkpoint.get("ema") is not None:
        return checkpoint["ema"], "ema"
    return checkpoint["model"], "model"


def checkpoint_integrity(model, state):
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatches = []
    for key in sorted(set(model_state) & set(state)):
        if tuple(model_state[key].shape) != tuple(state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "expected": list(model_state[key].shape),
                    "found": list(state[key].shape),
                }
            )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def encode(images, t_min, t_max):
    if images.min().item() < -1e-6 or images.max().item() > 1.0 + 1e-6:
        raise ValueError("TTFS evaluator expects raw images in [0,1]")
    return t_min + (1.0 - images) * (t_max - t_min)


def shifted_views(images):
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    return [
        padded[:, :, 1:33, 1:33],  # original
        padded[:, :, 0:32, 1:33],  # one pixel down in the resulting view
        padded[:, :, 2:34, 1:33],  # one pixel up
        padded[:, :, 1:33, 0:32],  # one pixel right
        padded[:, :, 1:33, 2:34],  # one pixel left
    ]


def make_views(images, mode):
    if mode == "none":
        return [images]
    if mode == "flip":
        return [images, torch.flip(images, dims=(-1,))]
    base_views = shifted_views(images)
    flipped_views = shifted_views(torch.flip(images, dims=(-1,)))
    return base_views + flipped_views


@torch.inference_mode()
def evaluate(
    models, loader, mode, device, amp, t_min, t_max, class_names,
    save_confusion_matrix_values,
):
    num_classes = len(class_names)
    total = 0
    correct = 0
    loss_sum = 0.0
    per_class_total = torch.zeros(num_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    confusion_matrix = (
        torch.zeros((num_classes, num_classes), dtype=torch.long)
        if save_confusion_matrix_values else None
    )
    started = time.time()
    view_count = None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        views = make_views(images, mode)
        view_count = len(views)
        logits_sum = None
        prediction_count = 0
        for view in views:
            encoded = encode(view, t_min, t_max)
            for model in models:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp and device.type == "cuda",
                ):
                    logits = model(encoded)
                logits_sum = logits if logits_sum is None else logits_sum + logits
                prediction_count += 1
        averaged_logits = logits_sum / prediction_count
        loss = F.cross_entropy(averaged_logits.float(), labels)
        predictions = averaged_logits.argmax(dim=1)
        matches = predictions.eq(labels)
        if confusion_matrix is not None:
            batch_confusion = torch.bincount(
                (labels * num_classes + predictions).detach().cpu(),
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            confusion_matrix += batch_confusion
        total += labels.numel()
        correct += matches.sum().item()
        loss_sum += loss.item() * labels.numel()
        for class_index in range(num_classes):
            class_mask = labels.eq(class_index)
            per_class_total[class_index] += class_mask.sum().cpu()
            per_class_correct[class_index] += (matches & class_mask).sum().cpu()

    result = {
        "mode": mode,
        "models": len(models),
        "views_per_model": view_count,
        "forward_passes_per_sample": len(models) * view_count,
        "loss": loss_sum / total,
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "samples": total,
        "seconds": time.time() - started,
        "per_class_accuracy": {
            class_names[index]: 100.0
            * per_class_correct[index].item()
            / max(per_class_total[index].item(), 1)
            for index in range(num_classes)
        },
    }
    if confusion_matrix is not None:
        result["confusion_matrix"] = confusion_matrix.tolist()
    return result


def save_confusion_matrix(output_dir, result, class_names, dataset_display_name):
    mode = result["mode"]
    matrix = result["confusion_matrix"]
    csv_path = output_dir / f"confusion_matrix_{mode}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])

    try:
        import matplotlib.pyplot as plt

        num_classes = len(class_names)
        figure_size = (24, 22) if num_classes > 20 else (10, 9)
        figure, axis = plt.subplots(figsize=figure_size)
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            xticks=range(num_classes),
            yticks=range(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="Actual class",
            title=(
                f"{dataset_display_name} confusion matrix: {mode} "
                f"({result['accuracy']:.2f}% accuracy)"
            ),
        )
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
        if num_classes <= 20:
            threshold = max(max(row) for row in matrix) / 2.0
            for row_index, row in enumerate(matrix):
                for column_index, value in enumerate(row):
                    axis.text(
                        column_index,
                        row_index,
                        str(value),
                        ha="center",
                        va="center",
                        color="white" if value > threshold else "black",
                        fontsize=8,
                    )
        else:
            axis.tick_params(axis="both", labelsize=5)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"confusion_matrix_{mode}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
    except ImportError:
        print(
            "matplotlib is unavailable; saved confusion-matrix values as CSV only",
            flush=True,
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    models = []
    checkpoint_reports = []
    reference_architecture = None
    t_min = None
    t_max = None
    for checkpoint_name in args.checkpoint:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model, architecture = build_model(checkpoint)
        comparable_architecture = {
            key: value for key, value in architecture.items()
            if key not in {"ttfs_grn", "deep_supervision"}
        }
        if reference_architecture is None:
            reference_architecture = comparable_architecture
        elif comparable_architecture != reference_architecture:
            raise RuntimeError(
                "All ensemble checkpoints must have identical inference architecture"
            )
        state, source = selected_state(checkpoint, args.weights_source)
        integrity = checkpoint_integrity(model, state)
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path.resolve()),
                    "weights_source": source,
                    "best_validation_accuracy": checkpoint.get(
                        "best_val_accuracy"
                    ),
                    "integrity": integrity,
                },
                indent=2,
            ),
            flush=True,
        )
        if any(integrity.values()):
            raise RuntimeError(
                f"Checkpoint integrity failed for {checkpoint_path}: {integrity}"
            )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        saved_args = checkpoint.get("args", {})
        checkpoint_t_min = float(saved_args.get("t_min", 0.0))
        checkpoint_t_max = float(saved_args.get("t_max", 1.0))
        if t_min is None:
            t_min, t_max = checkpoint_t_min, checkpoint_t_max
        elif (checkpoint_t_min, checkpoint_t_max) != (t_min, t_max):
            raise RuntimeError("Ensemble checkpoints use different TTFS time ranges")
        checkpoint_reports.append(
            {
                "path": str(checkpoint_path.resolve()),
                "weights_source": source,
                "best_validation_accuracy": checkpoint.get("best_val_accuracy"),
                "best_epoch": checkpoint.get("best_epoch"),
                "integrity": integrity,
            }
        )

    checkpoint_num_classes = int(reference_architecture["num_classes"])
    inferred_dataset = "cifar100" if checkpoint_num_classes == 100 else "cifar10"
    dataset_name = inferred_dataset if args.dataset == "auto" else args.dataset
    expected_classes = 100 if dataset_name == "cifar100" else 10
    if checkpoint_num_classes != expected_classes:
        raise RuntimeError(
            f"Checkpoint has {checkpoint_num_classes} output classes, but "
            f"--dataset {dataset_name} requires {expected_classes}"
        )
    dataset_class = datasets.CIFAR100 if dataset_name == "cifar100" else datasets.CIFAR10
    dataset_display_name = "CIFAR-100" if dataset_name == "cifar100" else "CIFAR-10"
    dataset = dataset_class(
        root=args.data_path,
        train=False,
        transform=transforms.ToTensor(),
        download=args.download,
    )
    class_names = tuple(dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    results = []
    for mode in args.tta_modes:
        metrics = evaluate(
            models,
            loader,
            mode,
            device,
            args.amp,
            t_min,
            t_max,
            class_names,
            args.save_confusion_matrix,
        )
        results.append(metrics)
        if args.save_confusion_matrix:
            save_confusion_matrix(
                output_dir, metrics, class_names, dataset_display_name
            )
        print(json.dumps(metrics, indent=2), flush=True)

    report = {
        "dataset": f"{dataset_display_name} test",
        "device": str(device),
        "amp": args.amp,
        "checkpoints": checkpoint_reports,
        "architecture": reference_architecture,
        "ttfs_time_range": [t_min, t_max],
        "results": results,
        "best_reported_mode": max(results, key=lambda item: item["accuracy"])[
            "mode"
        ],
        "note": (
            "Report all predeclared TTA modes. Selecting a TTA recipe after "
            "seeing test labels is not a training improvement."
        ),
    }
    (output_dir / "tta_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "tta_evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode",
                "models",
                "views_per_model",
                "forward_passes_per_sample",
                "loss",
                "accuracy",
                "correct",
                "samples",
                "seconds",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {key: result[key] for key in writer.fieldnames}
            )


if __name__ == "__main__":
    main()
