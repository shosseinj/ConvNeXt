#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.convnext import ConvNeXtSpiking

# The repository already has a root-level utils.py. Importing the requested
# tracker directory explicitly avoids shadowing that existing module.
TRACKER_UTILS_DIRECTORY = Path(__file__).resolve().parent / "utils"
if str(TRACKER_UTILS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TRACKER_UTILS_DIRECTORY))
from simple_experiment_tracker import SimpleExperimentTracker, local_timestamp


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


def four_int_tuple(value):
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected four comma-separated integers"
        ) from error
    if len(parsed) != 4 or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "Expected exactly four positive comma-separated integers"
        )
    return parsed


def args_parser():
    parser = argparse.ArgumentParser(
        "Continuous TTFS ConvNeXt on native CIFAR-10 32x32"
    )
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument(
        "--output_dir",
        default="results/cifar10_ttfs_native32_proposed_seed42",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--experiment_notes", default="")
    parser.add_argument("--dataset", default="CIFAR-10")
    parser.add_argument("--residual_operator", default="min")
    parser.add_argument("--pw1_mode", default="continuous TTFS")
    parser.add_argument("--pw2_mode", choices=("dense", "ttfs"), default="dense")
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--spike_dropout", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--dims", type=four_int_tuple, default="96,192,384,512")
    parser.add_argument("--depths", type=four_int_tuple, default="3,3,6,3")
    parser.add_argument("--dw_kernel_size", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=1.0)
    parser.add_argument("--force_positive_weights", type=str2bool, default=False)
    parser.add_argument("--init_delay", type=float, default=0.0)
    parser.add_argument("--stage_delays", default="0.4,0.0,0.0,0.0")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--print_freq", type=int, default=50)
    args = parser.parse_args()
    if args.drop_path != 0.0:
        parser.error("--drop_path must remain 0.0 for TTFS spike-time semantics")
    if args.mixup_alpha < 0.0:
        parser.error("--mixup_alpha must be non-negative")
    if not 0.0 <= args.head_dropout < 1.0:
        parser.error("--head_dropout must be in [0,1)")
    if not 0.0 <= args.spike_dropout <= 1.0:
        parser.error("--spike_dropout must be in [0,1]")
    if args.early_stopping_patience < 1:
        parser.error("--early_stopping_patience must be at least 1")
    if args.dw_kernel_size <= 0 or args.dw_kernel_size % 2 == 0:
        parser.error("--dw_kernel_size must be a positive odd integer")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(
        args.data_path, train=True, transform=train_transform, download=args.download
    )
    validation_dataset = datasets.CIFAR10(
        args.data_path, train=True, transform=eval_transform, download=False
    )
    test_dataset = datasets.CIFAR10(
        args.data_path, train=False, transform=eval_transform, download=args.download
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    validation_indices = indices[: args.val_size]
    train_indices = indices[args.val_size :]
    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    return (
        DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            **common,
        ),
        DataLoader(
            Subset(validation_dataset, validation_indices),
            batch_size=args.batch_size,
            shuffle=False,
            **common,
        ),
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **common,
        ),
    )


def make_model(args):
    delays = [float(value) for value in args.stage_delays.split(",")]
    if len(delays) != 4:
        raise ValueError("stage_delays must have 4 values")
    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=10,
        depths=args.depths,
        dims=args.dims,
        dw_kernel_size=args.dw_kernel_size,
        drop_path_rate=args.drop_path,
        t_min=args.t_min,
        t_max=args.t_max,
        head_dropout=args.head_dropout,
        spike_dropout=args.spike_dropout,
        pw2_mode=args.pw2_mode,
        force_positive_weights=args.force_positive_weights,
        init_delay=args.init_delay,
        stage_delays=delays,
    )
    stem = nn.Conv2d(
        3, args.dims[0], kernel_size=3, stride=1, padding=1, bias=True
    )
    nn.init.trunc_normal_(stem.weight, std=0.02)
    nn.init.zeros_(stem.bias)
    model.downsample_layers[0] = (
        nn.Sequential(stem)
        if isinstance(model.downsample_layers[0], nn.Sequential)
        else stem
    )
    return model


def encode(images, args):
    if images.min().item() < -1e-6 or images.max().item() > 1.0 + 1e-6:
        raise ValueError("Input to TTFS encode must be raw image data in [0,1]")
    return args.t_min + (1.0 - images) * (args.t_max - args.t_min)


def mixup_batch(images, labels, alpha):
    """Mix raw [0,1] images; TTFS encoding is intentionally done afterward."""
    if images.min().item() < -1e-6 or images.max().item() > 1.0 + 1e-6:
        raise ValueError("Mixup must receive raw images in [0,1]")
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    return mixed_images, labels, labels[permutation], lam


def lr_at(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
    progress = (epoch - args.warmup_epochs) / max(
        1, args.epochs - args.warmup_epochs - 1
    )
    progress = min(max(progress, 0.0), 1.0)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def architecture_metadata(args):
    return {
        "dims": list(args.dims),
        "depths": list(args.depths),
        "input_resolution": [32, 32],
        "depthwise_kernel_size": args.dw_kernel_size,
        "pw2_mode": args.pw2_mode,
        "stem": {
            "in_channels": 3,
            "out_channels": args.dims[0],
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
    }


def validate_resume_architecture(checkpoint, args):
    checkpoint_architecture = checkpoint.get("architecture")
    requested_architecture = architecture_metadata(args)
    if checkpoint_architecture != requested_architecture:
        raise ValueError(
            "Resume checkpoint architecture does not match this run. "
            f"Checkpoint={checkpoint_architecture}, requested={requested_architecture}. "
            "Do not resume the previous large-model checkpoint."
        )


def create_experiment_report(
    args,
    output_dir,
    train_sample_count,
    validation_sample_count,
    test_sample_count,
    parameter_count,
    previous_report=None,
):
    previous_report = previous_report or {}
    previous_experiment = previous_report.get("experiment", {})
    previous_results = previous_report.get("results", {})
    previous_optional = previous_report.get("optional_evaluation", {})
    stage_delays = [float(value) for value in args.stage_delays.split(",")]
    delay_enabled = args.init_delay != 0.0 or any(
        value != 0.0 for value in stage_delays
    )
    experiment_name = (
        args.experiment_name.strip()
        or previous_experiment.get("experiment_name")
        or output_dir.name
    )
    experiment_notes = (
        args.experiment_notes.strip()
        or previous_experiment.get("notes")
        or None
    )
    return {
        "experiment": {
            "experiment_name": experiment_name,
            "date_time": previous_experiment.get("date_time") or local_timestamp(),
            "output_directory": str(output_dir.resolve()),
            "notes": experiment_notes,
            "seed": args.seed,
            "status": "resumed" if args.resume else "running",
            "updated_at": local_timestamp(),
        },
        "dataset": {
            "dataset_name": args.dataset,
            "number_of_classes": 10,
            "input_resolution": [32, 32],
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "test_sample_count": test_sample_count,
            "preprocessing": (
                "ToTensor to raw [0,1], optional training Mixup, then "
                "continuous TTFS encoding"
            ),
            "augmentation": (
                "training: RandomCrop(32,padding=4), RandomHorizontalFlip, "
                f"Mixup(alpha={args.mixup_alpha}); validation/test: ToTensor only"
            ),
        },
        "architecture": {
            "dims": list(args.dims),
            "depths": list(args.depths),
            "parameter_count": parameter_count,
            "stem_kernel": 3,
            "stem_stride": 1,
            "stem_padding": 1,
            "depthwise_kernel_size": args.dw_kernel_size,
            "residual_operator": args.residual_operator,
            "pw1_mode": args.pw1_mode,
            "pw2_mode": args.pw2_mode,
            "spike_dropout": args.spike_dropout,
            "delay_enabled": delay_enabled,
            "stage_delays": stage_delays,
            "t_min": args.t_min,
            "t_max": args.t_max,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "head_dropout": args.head_dropout,
            "mixup_alpha": args.mixup_alpha,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "results": {
            "best_epoch": previous_results.get("best_epoch"),
            "best_validation_accuracy": previous_results.get(
                "best_validation_accuracy"
            ),
            "final_train_accuracy": previous_results.get("final_train_accuracy"),
            "final_validation_accuracy": previous_results.get(
                "final_validation_accuracy"
            ),
            "test_accuracy": previous_results.get("test_accuracy"),
            "test_loss": previous_results.get("test_loss"),
            "training_time_seconds": previous_results.get(
                "training_time_seconds", 0.0
            ),
            "checkpoint_path": previous_results.get("checkpoint_path"),
        },
        "optional_evaluation": {
            "activation_sparsity": previous_optional.get("activation_sparsity"),
            "dense_macs_per_sample": previous_optional.get(
                "dense_macs_per_sample"
            ),
            "theoretical_synops_per_sample": previous_optional.get(
                "theoretical_synops_per_sample"
            ),
        },
    }


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total = 0
    weighted_correct = 0.0
    loss_sum = 0.0
    start_time = time.time()

    for iteration, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            # Mixup is applied to raw images in [0,1], before TTFS encoding.
            images, labels_a, labels_b, lam = mixup_batch(
                images, labels, args.mixup_alpha
            )
            optimizer.zero_grad(set_to_none=True)
        else:
            # Validation and test follow this branch and never use Mixup.
            labels_a = labels_b = labels
            lam = 1.0

        images = encode(images, args)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            output = model(images)
            loss = lam * criterion(output, labels_a) + (1.0 - lam) * criterion(
                output, labels_b
            )

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        if training:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        batch_size = labels.size(0)
        predictions = output.argmax(dim=1)
        weighted_correct += lam * (predictions == labels_a).sum().item()
        weighted_correct += (1.0 - lam) * (predictions == labels_b).sum().item()
        total += batch_size
        loss_sum += loss.item() * batch_size
        if iteration % args.print_freq == 0:
            print(
                json.dumps(
                    {
                        "phase": "train" if training else "validation",
                        "iteration": iteration,
                        "loss": loss_sum / total,
                        "accuracy": 100.0 * weighted_correct / total,
                    }
                ),
                flush=True,
            )

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": 100.0 * weighted_correct / max(total, 1),
        "samples": total,
        "seconds": time.time() - start_time,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_validation_accuracy,
    best_epoch,
    epochs_without_improvement,
    args,
):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "architecture": architecture_metadata(args),
            "args": vars(args),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def main():
    args = args_parser()
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    train_loader, validation_loader, test_loader = build_loaders(args)
    model = make_model(args).to(device)

    # Verify native 32x32 input, unchanged stride-1 stem, and stage schedule.
    shapes = {}
    handles = []
    for index, layer in enumerate(model.downsample_layers):
        handles.append(
            layer.register_forward_hook(
                lambda module, inputs, output, index=index: shapes.__setitem__(
                    index, tuple(output.shape)
                )
            )
        )
    with torch.no_grad():
        model(encode(torch.rand(1, 3, 32, 32, device=device), args))
    for handle in handles:
        handle.remove()
    print("Runtime downsample shapes:", shapes)
    expected_shapes = {
        0: (1, args.dims[0], 32, 32),
        1: (1, args.dims[1], 16, 16),
        2: (1, args.dims[2], 8, 8),
        3: (1, args.dims[3], 4, 4),
    }
    assert shapes == expected_shapes, f"Expected {expected_shapes}, got {shapes}"
    stem = model.downsample_layers[0][0]
    assert stem.out_channels == args.dims[0]
    assert model.head.in_features == args.dims[-1]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Architecture: dims={args.dims}, depths={args.depths}, "
        f"parameters={parameter_count:,}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    start_epoch = 0
    best_validation_accuracy = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_architecture(checkpoint, args)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", -1))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        print(
            f"Resumed from epoch {checkpoint['epoch']} with "
            f"best validation accuracy {best_validation_accuracy:.2f}% and "
            f"{epochs_without_improvement} epochs without improvement"
        )

    config = {
        **vars(args),
        "dims": list(args.dims),
        "depths": list(args.depths),
        "input_resolution": [32, 32],
        "stem": {
            "in_channels": 3,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "out_channels": args.dims[0],
        },
        "spatial_schedule": [32, 32, 16, 8, 4],
        "depthwise_kernel_size": args.dw_kernel_size,
        "parameter_count": parameter_count,
        "temporal_formulation": "continuous analytic TTFS",
        "simulation_steps": None,
        "mixup_order": "raw images in [0,1], then TTFS encode",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    tracker = SimpleExperimentTracker(
        output_directory=output_dir,
        registry_path=Path(__file__).resolve().parent / "experiments_registry.csv",
    )
    experiment_report = create_experiment_report(
        args=args,
        output_dir=output_dir,
        train_sample_count=len(train_loader.dataset),
        validation_sample_count=len(validation_loader.dataset),
        test_sample_count=len(test_loader.dataset),
        parameter_count=parameter_count,
        previous_report=tracker.load_existing_report(),
    )
    previous_training_time = experiment_report["results"].get(
        "training_time_seconds"
    )
    if not isinstance(previous_training_time, (int, float)):
        previous_training_time = 0.0
    tracking_session_started = time.time()
    tracker.save(experiment_report)

    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        learning_rate = lr_at(epoch, args)
        print(f'\n\n\n\nEpoch == {epoch}')
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        train_metrics = run_epoch(
            model, train_loader, criterion, device, args, optimizer, scaler
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model, validation_loader, criterion, device, args
            )

        improved = validation_metrics["accuracy"] > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"val_{key}": value
                for key, value in validation_metrics.items()
            },
            "best_validation_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        print(json.dumps(log_row), flush=True)
        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row) + "\n")

        if improved:
            save_checkpoint(
                output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                scaler,
                epoch,
                best_validation_accuracy,
                best_epoch,
                epochs_without_improvement,
                args,
            )
        save_checkpoint(
            output_dir / "last_checkpoint.pth",
            model,
            optimizer,
            scaler,
            epoch,
            best_validation_accuracy,
            best_epoch,
            epochs_without_improvement,
            args,
        )
        last_epoch = epoch

        should_stop = (
            epochs_without_improvement >= args.early_stopping_patience
        )
        experiment_report["experiment"]["status"] = (
            "early_stopped" if should_stop else "running"
        )
        experiment_report["results"].update(
            {
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_validation_accuracy,
                "final_train_accuracy": train_metrics["accuracy"],
                "final_validation_accuracy": validation_metrics["accuracy"],
                "training_time_seconds": previous_training_time
                + (time.time() - tracking_session_started),
                "checkpoint_path": str(
                    (output_dir / "best_checkpoint.pth").resolve()
                ),
            }
        )
        tracker.save(experiment_report)

        if should_stop:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: validation accuracy did not "
                f"improve for {epochs_without_improvement} epochs."
            )
            break

    best_checkpoint_path = output_dir / "best_checkpoint.pth"
    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found after training/resume: {best_checkpoint_path}"
        )
    best_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"], strict=True)
    with torch.inference_mode():
        # run_epoch has no optimizer here, so the test path cannot apply Mixup.
        test_metrics = run_epoch(model, test_loader, criterion, device, args)

    summary = {
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "early_stopped": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(output_dir / "last_checkpoint.pth"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    experiment_report["experiment"]["status"] = (
        "early_stopped" if stopped_early else "completed"
    )
    experiment_report["results"].update(
        {
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "training_time_seconds": previous_training_time
            + (time.time() - tracking_session_started),
            "checkpoint_path": str(best_checkpoint_path.resolve()),
        }
    )
    tracker.save(experiment_report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
