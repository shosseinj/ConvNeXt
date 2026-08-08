#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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


def dataset_name(value):
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError("Dataset must be CIFAR-10 or CIFAR-100")
    return aliases[normalized]


def dataset_metadata(name):
    if name == "cifar10":
        return {
            "display_name": "CIFAR-10",
            "dataset_class": datasets.CIFAR10,
            "num_classes": 10,
        }
    if name == "cifar100":
        return {
            "display_name": "CIFAR-100",
            "dataset_class": datasets.CIFAR100,
            "num_classes": 100,
        }
    raise ValueError(f"Unsupported dataset: {name}")


def args_parser():
    parser = argparse.ArgumentParser(
        "Continuous TTFS ConvNeXt on native CIFAR 32x32 datasets"
    )
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument(
        "--output_dir",
        default="results/cifar10_ttfs_native32_k3_ttfs_stage_delay_seed42",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--experiment_notes", default="")
    parser.add_argument("--dataset", type=dataset_name, default="cifar10")
    parser.add_argument("--residual_operator", default="min")
    parser.add_argument("--pw1_mode", default="continuous TTFS")
    parser.add_argument("--pw2_mode", choices=("dense", "ttfs"), default="ttfs")
    parser.add_argument(
        "--ttfs_norm_mode",
        choices=("none", "score_layernorm"),
        default="none",
    )
    parser.add_argument("--final_score_norm", type=str2bool, default=False)
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_scheduler_patience", type=int, default=6)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.05)
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--spike_dropout", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--randaugment", type=str2bool, default=False)
    parser.add_argument("--randaugment_num_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--random_erasing", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--dims", type=four_int_tuple, default="96,192,384,768")
    parser.add_argument("--depths", type=four_int_tuple, default="2,2,6,2")
    parser.add_argument("--dw_kernel_size", type=int, choices=(3, 5, 7), default=3)
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
    selected_dataset = dataset_metadata(args.dataset)
    args.dataset_display_name = selected_dataset["display_name"]
    args.num_classes = selected_dataset["num_classes"]
    if args.drop_path != 0.0:
        parser.error("--drop_path must remain 0.0 for TTFS spike-time semantics")
    if args.mixup_alpha < 0.0:
        parser.error("--mixup_alpha must be non-negative")
    if args.cutmix_alpha < 0.0:
        parser.error("--cutmix_alpha must be non-negative")
    if args.randaugment_num_ops < 1:
        parser.error("--randaugment_num_ops must be at least 1")
    if args.randaugment_magnitude < 0:
        parser.error("--randaugment_magnitude must be non-negative")
    if not 0.0 <= args.random_erasing <= 1.0:
        parser.error("--random_erasing must be in [0,1]")
    if not 0.0 <= args.head_dropout < 1.0:
        parser.error("--head_dropout must be in [0,1)")
    if not 0.0 <= args.spike_dropout <= 1.0:
        parser.error("--spike_dropout must be in [0,1]")
    if args.early_stopping_patience < 1:
        parser.error("--early_stopping_patience must be at least 1")
    if args.lr_scheduler_patience < 1:
        parser.error("--lr_scheduler_patience must be at least 1")
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        parser.error("--lr_scheduler_factor must be in (0,1)")
    if args.early_stopping_min_delta < 0.0:
        parser.error("--early_stopping_min_delta must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema_decay must be in [0,1)")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_run_command(output_dir):
    def powershell_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    arguments = [
        str(Path(sys.argv[0]).resolve()),
        *sys.argv[1:],
    ]
    lines = [f"& {powershell_quote(sys.executable)} `"]
    for index, argument in enumerate(arguments):
        suffix = " `" if index < len(arguments) - 1 else ""
        lines.append(f"  {powershell_quote(argument)}{suffix}")
    (output_dir / "run_command.ps1").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


class ModelEMA:
    """Exponential moving average used for validation and final evaluation."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, averaged in self.module.state_dict().items():
            current = source[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                averaged.copy_(current)

    @torch.no_grad()
    def set(self, model):
        self.module.load_state_dict(model.state_dict(), strict=True)


def build_loaders(args):
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.randaugment:
        train_transforms.append(
            transforms.RandAugment(
                num_ops=args.randaugment_num_ops,
                magnitude=args.randaugment_magnitude,
            )
        )
    train_transforms.append(transforms.ToTensor())
    if args.random_erasing > 0.0:
        train_transforms.append(transforms.RandomErasing(p=args.random_erasing))
    train_transform = transforms.Compose(train_transforms)
    eval_transform = transforms.ToTensor()
    dataset_class = dataset_metadata(args.dataset)["dataset_class"]
    train_dataset = dataset_class(
        args.data_path, train=True, transform=train_transform, download=args.download
    )
    validation_dataset = dataset_class(
        args.data_path, train=True, transform=eval_transform, download=False
    )
    test_dataset = dataset_class(
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
        num_classes=args.num_classes,
        depths=args.depths,
        dims=args.dims,
        dw_kernel_size=args.dw_kernel_size,
        cifar_stem=True,
        downsample_kernel_size=3,
        drop_path_rate=args.drop_path,
        t_min=args.t_min,
        t_max=args.t_max,
        head_dropout=args.head_dropout,
        spike_dropout=args.spike_dropout,
        pw2_mode=args.pw2_mode,
        ttfs_norm_mode=args.ttfs_norm_mode,
        final_score_norm=args.final_score_norm,
        force_positive_weights=args.force_positive_weights,
        init_delay=args.init_delay,
        stage_delays=delays,
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


def cutmix_batch(images, labels, alpha):
    if images.min().item() < -1e-6 or images.max().item() > 1.0 + 1e-6:
        raise ValueError("CutMix must receive raw images in [0,1]")
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    height, width = images.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = random.randrange(width)
    center_y = random.randrange(height)
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, labels, labels[permutation], lam


def batch_mix(images, labels, args):
    if args.mixup_alpha > 0.0 and args.cutmix_alpha > 0.0:
        if random.getrandbits(1):
            return mixup_batch(images, labels, args.mixup_alpha)
        return cutmix_batch(images, labels, args.cutmix_alpha)
    if args.mixup_alpha > 0.0:
        return mixup_batch(images, labels, args.mixup_alpha)
    if args.cutmix_alpha > 0.0:
        return cutmix_batch(images, labels, args.cutmix_alpha)
    return images, labels, labels, 1.0


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
        "num_classes": args.num_classes,
        "dims": list(args.dims),
        "depths": list(args.depths),
        "input_resolution": [32, 32],
        "depthwise_kernel_size": args.dw_kernel_size,
        "pw2_mode": args.pw2_mode,
        "ttfs_norm_mode": args.ttfs_norm_mode,
        "final_score_norm": args.final_score_norm,
        "downsample_kernel_size": 3,
        "downsample_stride": 2,
        "downsample_padding": 1,
        "stage_delays": [float(value) for value in args.stage_delays.split(",")],
        "delay_parameterization": "max_delay * sigmoid(raw_delay)",
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
    if checkpoint_architecture is not None:
        checkpoint_architecture = dict(checkpoint_architecture)
        checkpoint_architecture.setdefault("final_score_norm", False)
        checkpoint_architecture.setdefault("num_classes", 10)
    if checkpoint_architecture != requested_architecture:
        raise ValueError(
            "Resume checkpoint architecture does not match this run. "
            f"Checkpoint={checkpoint_architecture}, requested={requested_architecture}. "
            "Do not resume the previous large-model checkpoint."
        )


def actual_stage_delays(model):
    values = []
    for stage_index, stage in enumerate(model.stages):
        block_values = [block.effective_delay_means() for block in stage]
        values.append(
            {
                "stage": stage_index,
                "mid": sum(item["mid"] for item in block_values) / len(block_values),
                "out": (
                    sum(item["out"] for item in block_values) / len(block_values)
                    if block_values[0]["out"] is not None else None
                ),
            }
        )
    return values


def delay_gradient_diagnostic(model, args, device):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    was_training = model.training
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        images = torch.rand(2, 3, 32, 32, device=device)
        logits = model(encode(images, args))
        expected_logits_shape = (images.size(0), args.num_classes)
        if logits.shape != expected_logits_shape:
            raise RuntimeError(
                f"Diagnostic expected logits shape {expected_logits_shape}, "
                f"got {tuple(logits.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Diagnostic produced non-finite logits")
        logits.square().mean().backward()
        stage_norms = []
        norm_gradient_norms = []
        for stage_index, stage in enumerate(model.stages):
            squared_mid = 0.0
            squared_out = 0.0
            for block in stage:
                if block.D_mid.grad is None:
                    raise RuntimeError(
                        f"Missing delay gradient in stage {stage_index}"
                    )
                squared_mid += float(block.D_mid.grad.float().square().sum().item())
                if block.D_out is not None:
                    if block.D_out.grad is None:
                        raise RuntimeError(
                            f"Missing output delay gradient in stage {stage_index}"
                        )
                    squared_out += float(
                        block.D_out.grad.float().square().sum().item()
                    )
            stage_norms.append(
                {
                    "stage": stage_index,
                    "D_mid": math.sqrt(squared_mid),
                    "D_out": math.sqrt(squared_out) if stage[0].D_out is not None else None,
                }
            )
            for block_index, block in enumerate(stage):
                if block.norm is None:
                    continue
                for parameter_name, parameter in block.norm.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(
                            "LayerNorm parameter received a missing or non-finite "
                            f"gradient: stage={stage_index}, block={block_index}, "
                            f"parameter={parameter_name}"
                        )
                    norm_gradient_norms.append(
                        {
                            "stage": stage_index,
                            "block": block_index,
                            "parameter": parameter_name,
                            "gradient_norm": float(
                                parameter.grad.float().norm().item()
                            ),
                        }
                    )
        final_norm_gradient_norms = []
        if args.final_score_norm:
            for parameter_name, parameter in model.final_norm.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(
                        "Final LayerNorm received a missing or non-finite gradient: "
                        f"{parameter_name}"
                    )
                final_norm_gradient_norms.append(
                    {
                        "parameter": parameter_name,
                        "gradient_norm": float(parameter.grad.float().norm().item()),
                    }
                )
        return {
            "logits_shape": list(logits.shape),
            "logits_finite": True,
            "delay_gradient_norms": stage_norms,
            "layernorm_gradient_norms": norm_gradient_norms,
            "final_layernorm_gradient_norms": final_norm_gradient_norms,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


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
            "dataset_name": args.dataset_display_name,
            "number_of_classes": args.num_classes,
            "input_resolution": [32, 32],
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "test_sample_count": test_sample_count,
            "preprocessing": (
                "augmentation, ToTensor/RandomErasing, optional Mixup/CutMix, "
                "then continuous TTFS encoding"
            ),
            "augmentation": (
                "training: RandomCrop(32,padding=4), RandomHorizontalFlip, "
                f"RandAugment(enabled={args.randaugment},ops="
                f"{args.randaugment_num_ops},magnitude={args.randaugment_magnitude}), "
                f"RandomErasing(p={args.random_erasing}), Mixup(alpha="
                f"{args.mixup_alpha}), CutMix(alpha={args.cutmix_alpha}); "
                "validation/test: ToTensor only"
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
            "downsample_kernel": 3,
            "downsample_stride": 2,
            "downsample_padding": 1,
            "residual_operator": args.residual_operator,
            "pw1_mode": args.pw1_mode,
            "pw2_mode": args.pw2_mode,
            "ttfs_norm_mode": args.ttfs_norm_mode,
            "final_score_norm": args.final_score_norm,
            "spike_dropout": args.spike_dropout,
            "delay_enabled": delay_enabled,
            "stage_delays": stage_delays,
            "delay_parameterization": "max_delay * sigmoid(raw_delay)",
            "t_min": args.t_min,
            "t_max": args.t_max,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "lr_scheduler": "ReduceLROnPlateau(mode=max)",
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "minimum_learning_rate": args.min_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "head_dropout": args.head_dropout,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "randaugment": args.randaugment,
            "randaugment_num_ops": args.randaugment_num_ops,
            "randaugment_magnitude": args.randaugment_magnitude,
            "random_erasing": args.random_erasing,
            "early_stopping_patience": args.early_stopping_patience,
            "ema_enabled": args.ema,
            "ema_decay": args.ema_decay if args.ema else None,
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


def run_epoch(
    model, loader, criterion, device, args, optimizer=None, scaler=None, ema=None
):
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
            # Batch mixing remains on raw [0,1] tensors before TTFS encoding.
            images, labels_a, labels_b, lam = batch_mix(images, labels, args)
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
            if ema is not None:
                ema.update(model)

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
    scheduler,
    scaler,
    ema,
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
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "ema_decay": ema.decay if ema is not None else None,
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
    save_run_command(output_dir)
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
    assert (
        stem.in_channels,
        stem.out_channels,
        stem.kernel_size,
        stem.stride,
        stem.padding,
    ) == (3, args.dims[0], (3, 3), (1, 1), (1, 1))
    for layer in model.downsample_layers[1:]:
        convolution = layer[0]
        assert convolution.kernel_size == (3, 3)
        assert convolution.stride == (2, 2)
        assert convolution.padding == (1, 1)
    assert model.head.in_features == args.dims[-1]
    assert model.head.out_features == args.num_classes
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Dataset: {args.dataset_display_name}, classes={args.num_classes}, "
        "input_resolution=32x32"
    )
    print(
        f"Architecture: dims={args.dims}, depths={args.depths}, "
        f"parameters={parameter_count:,}"
    )
    print(f"TTFS normalization mode: {args.ttfs_norm_mode}")
    print(f"Final score normalization enabled: {args.final_score_norm}")
    print(
        "Augmentation settings:",
        {
            "randaugment": args.randaugment,
            "randaugment_num_ops": args.randaugment_num_ops,
            "randaugment_magnitude": args.randaugment_magnitude,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "random_erasing": args.random_erasing,
        },
    )
    measured_delays = actual_stage_delays(model)
    print("Actual stage delays:", measured_delays)
    smoke_test = delay_gradient_diagnostic(model, args, device)
    delay_gradient_norms = smoke_test["delay_gradient_norms"]
    print("Delay gradient norms after test backward:", delay_gradient_norms)
    print("TTFS normalization smoke test:", smoke_test)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        threshold=args.early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    start_epoch = 0
    best_validation_accuracy = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_architecture(checkpoint, args)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Legacy cosine checkpoints have no plateau-scheduler state. Start
            # the requested LR schedule from the selected best model.
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            scheduler.best = checkpoint.get("best_val_accuracy", -float("inf"))
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None:
            if checkpoint.get("ema") is not None:
                ema.module.load_state_dict(checkpoint["ema"], strict=True)
            else:
                ema.set(model)
                print("Checkpoint has no EMA state; initialized EMA from model weights")
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
        "dataset_configuration": {
            "name": args.dataset_display_name,
            "canonical_name": args.dataset,
            "number_of_classes": args.num_classes,
            "input_resolution": [32, 32],
        },
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
        "downsampling": {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
        },
        "actual_stage_delays": measured_delays,
        "delay_parameterization": "max_delay * sigmoid(raw_delay)",
        "delay_gradient_norms_test_backward": delay_gradient_norms,
        "ttfs_normalization_mode": args.ttfs_norm_mode,
        "ttfs_normalization_smoke_test": smoke_test,
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
        print(f'\n\n\n\nEpoch == {epoch}')
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / max(1, args.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        learning_rate = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(
            model, train_loader, criterion, device, args, optimizer, scaler, ema
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                ema.module if ema is not None else model,
                validation_loader,
                criterion,
                device,
                args,
            )

        checkpoint_improved = (
            validation_metrics["accuracy"] > best_validation_accuracy
        )
        meaningfully_improved = validation_metrics["accuracy"] > (
            best_validation_accuracy + args.early_stopping_min_delta
        )
        if checkpoint_improved:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
        if meaningfully_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= args.warmup_epochs:
            scheduler.step(validation_metrics["accuracy"])
        next_learning_rate = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
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

        if checkpoint_improved:
            save_checkpoint(
                output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                ema,
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
            scheduler,
            scaler,
            ema,
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
    evaluation_state = best_checkpoint.get("ema")
    if evaluation_state is None:
        evaluation_state = best_checkpoint["model"]
    model.load_state_dict(evaluation_state, strict=True)
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
