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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

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
        "tinyimagenet": "tinyimagenet",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "Dataset must be CIFAR-10, CIFAR-100, or Tiny ImageNet"
        )
    return aliases[normalized]


def dataset_metadata(name):
    if name == "cifar10":
        return {
            "display_name": "CIFAR-10",
            "dataset_class": datasets.CIFAR10,
            "num_classes": 10,
            "input_resolution": 32,
            "crop_padding": 4,
            "default_val_size": 5000,
        }
    if name == "cifar100":
        return {
            "display_name": "CIFAR-100",
            "dataset_class": datasets.CIFAR100,
            "num_classes": 100,
            "input_resolution": 32,
            "crop_padding": 4,
            "default_val_size": 5000,
        }
    if name == "tinyimagenet":
        return {
            "display_name": "Tiny ImageNet",
            "dataset_class": None,
            "num_classes": 200,
            "input_resolution": 64,
            "crop_padding": 8,
            "default_val_size": 10000,
        }
    raise ValueError(f"Unsupported dataset: {name}")


def resolve_tinyimagenet_root(data_path):
    root = Path(data_path)
    candidates = (root, root / "tiny-imagenet-200")
    for candidate in candidates:
        if (
            (candidate / "train").is_dir()
            and (candidate / "val" / "images").is_dir()
            and (candidate / "val" / "val_annotations.txt").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Tiny ImageNet root must contain train/, val/images/, and "
        f"val/val_annotations.txt. Checked: {', '.join(map(str, candidates))}"
    )


class TinyImageNetValidationDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.root = Path(root)
        self.transform = transform
        annotations_path = self.root / "val_annotations.txt"
        images_directory = self.root / "images"
        samples = []
        for line in annotations_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed Tiny ImageNet annotation: {line!r}")
            filename, class_id = fields[:2]
            if class_id not in class_to_idx:
                raise ValueError(f"Unknown Tiny ImageNet class in val annotations: {class_id}")
            image_path = images_directory / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"Tiny ImageNet validation image missing: {image_path}")
            samples.append((image_path, class_to_idx[class_id]))
        if not samples:
            raise ValueError(f"No Tiny ImageNet validation samples found in {annotations_path}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def resolve_experiment_identity(dataset, experiment_name, seed, output_dir):
    base_name = str(experiment_name).strip()
    if (
        not base_name
        or base_name in {".", ".."}
        or "/" in base_name
        or "\\" in base_name
        or Path(base_name).is_absolute()
    ):
        raise ValueError(
            "--experiment_name must be a non-empty base name without path separators"
        )

    full_name = f"{dataset}_{base_name}_seed{seed}"
    requested_output_dir = str(output_dir).strip()
    resolved_output_dir = requested_output_dir or (
        Path("results") / dataset / base_name / f"seed_{seed}"
    ).as_posix()
    return full_name, resolved_output_dir


def args_parser():
    parser = argparse.ArgumentParser(
        "Continuous TTFS ConvNeXt on native CIFAR and Tiny ImageNet datasets"
    )
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument("--output_dir", default="")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default="")
    checkpoint_group.add_argument("--pretrained_checkpoint", default="")
    checkpoint_group.add_argument("--ann_pretrained_checkpoint", default="")
    checkpoint_group.add_argument("--constrained_finetune_checkpoint", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--experiment_notes", default="")
    parser.add_argument("--dataset", type=dataset_name, default="cifar10")
    parser.add_argument(
        "--residual_operator",
        choices=("min", "mean", "learnable_gate"),
        default="min",
    )
    parser.add_argument(
        "--allow_pretrained_residual_operator_change",
        type=str2bool,
        default=False,
        help=(
            "Allow the residual-fusion ablation to initialize a mean or "
            "learnable-gate Fully-TTFS target from a matched dense checkpoint "
            "whose residual operator is min. All other architecture fields "
            "remain strict."
        ),
    )
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
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--conv_delay_lr", type=float, default=None)
    parser.add_argument("--delay_regularization_weight", type=float, default=0.0)
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
    parser.add_argument(
        "--dwconv_mode", choices=("dense", "ttfs"), default="ttfs"
    )
    parser.add_argument(
        "--downsample_mode", choices=("dense", "ttfs"), default="ttfs"
    )
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=1.0)
    parser.add_argument("--force_positive_weights", type=str2bool, default=False)
    parser.add_argument(
        "--force_positive_pointwise_weights",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--init_delay", type=float, default=0.0)
    parser.add_argument("--stage_delays", default="0.4,0.0,0.0,0.0")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_size", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    args = parser.parse_args()
    selected_dataset = dataset_metadata(args.dataset)
    args.dataset_display_name = selected_dataset["display_name"]
    args.num_classes = selected_dataset["num_classes"]
    args.input_resolution = selected_dataset["input_resolution"]
    args.crop_padding = selected_dataset["crop_padding"]
    if args.val_size is None:
        args.val_size = selected_dataset["default_val_size"]
    if args.split_seed is None:
        args.split_seed = args.seed
    try:
        args.experiment_name, args.output_dir = resolve_experiment_identity(
            dataset=args.dataset,
            experiment_name=args.experiment_name,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        parser.error(str(error))
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
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.conv_delay_lr is not None and args.conv_delay_lr <= 0.0:
        parser.error("--conv_delay_lr must be positive")
    if args.delay_regularization_weight < 0.0:
        parser.error("--delay_regularization_weight must be non-negative")
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


def build_optimizer(model, args):
    convolution_delay_lr = getattr(args, "conv_delay_lr", None)
    if convolution_delay_lr is None:
        return torch.optim.AdamW(
            [
                {
                    "params": list(model.parameters()),
                    "name": "all",
                    "lr": args.lr,
                    "target_lr": args.lr,
                }
            ],
            weight_decay=args.weight_decay,
        )

    transferred = []
    convolution_delays = []
    for name, parameter in model.named_parameters():
        if name.endswith("D_conv"):
            convolution_delays.append(parameter)
        else:
            transferred.append(parameter)
    if not convolution_delays:
        raise ValueError(
            "--conv_delay_lr requires TTFS convolution D_conv parameters"
        )
    return torch.optim.AdamW(
        [
            {
                "params": transferred,
                "name": "transferred",
                "lr": args.lr,
                "target_lr": args.lr,
            },
            {
                "params": convolution_delays,
                "name": "conv_delays",
                "lr": convolution_delay_lr,
                "target_lr": convolution_delay_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )


def apply_warmup_learning_rates(optimizer, epoch, warmup_epochs):
    scale = (epoch + 1) / max(1, warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = group.get("target_lr", group["lr"]) * scale


def build_loaders(args):
    metadata = dataset_metadata(args.dataset)
    input_resolution = metadata["input_resolution"]
    train_transforms = [
        transforms.RandomCrop(input_resolution, padding=metadata["crop_padding"]),
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
    if args.dataset == "tinyimagenet":
        root = resolve_tinyimagenet_root(args.data_path)
        train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
        validation_dataset = datasets.ImageFolder(
            root / "train", transform=eval_transform
        )
        if len(train_dataset.classes) != metadata["num_classes"]:
            raise ValueError(
                f"Expected {metadata['num_classes']} Tiny ImageNet train classes, "
                f"found {len(train_dataset.classes)}"
            )
        test_dataset = TinyImageNetValidationDataset(
            root / "val",
            train_dataset.class_to_idx,
            transform=eval_transform,
        )
    else:
        dataset_class = metadata["dataset_class"]
        train_dataset = dataset_class(
            args.data_path,
            train=True,
            transform=train_transform,
            download=args.download,
        )
        validation_dataset = dataset_class(
            args.data_path,
            train=True,
            transform=eval_transform,
            download=False,
        )
        test_dataset = dataset_class(
            args.data_path,
            train=False,
            transform=eval_transform,
            download=args.download,
        )
    if not 0 < args.val_size < len(train_dataset):
        raise ValueError(
            f"val_size must be between 1 and {len(train_dataset) - 1}, "
            f"got {args.val_size}"
        )
    generator = torch.Generator().manual_seed(args.split_seed)
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
        dwconv_mode=args.dwconv_mode,
        downsample_mode=args.downsample_mode,
        residual_operator=getattr(args, "residual_operator", "min"),
        force_positive_weights=args.force_positive_weights,
        force_positive_pointwise_weights=(
            getattr(args, "force_positive_pointwise_weights", False)
        ),
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
        "input_resolution": [args.input_resolution, args.input_resolution],
        "depthwise_kernel_size": args.dw_kernel_size,
        "dwconv_mode": args.dwconv_mode,
        "downsample_mode": args.downsample_mode,
        "residual_operator": getattr(args, "residual_operator", "min"),
        "force_positive_weights": args.force_positive_weights,
        "force_positive_pointwise_weights": (
            getattr(args, "force_positive_pointwise_weights", False)
        ),
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
        checkpoint_architecture.setdefault("dwconv_mode", "dense")
        checkpoint_architecture.setdefault("downsample_mode", "dense")
        checkpoint_architecture.setdefault("residual_operator", "min")
        checkpoint_architecture.setdefault("force_positive_weights", False)
        checkpoint_architecture.setdefault(
            "force_positive_pointwise_weights", False
        )
    if checkpoint_architecture != requested_architecture:
        raise ValueError(
            "Resume checkpoint architecture does not match this run. "
            f"Checkpoint={checkpoint_architecture}, requested={requested_architecture}. "
            "Do not resume the previous large-model checkpoint."
        )


def validate_resume_training_configuration(checkpoint, args):
    checkpoint_args = checkpoint.get("args") or {}
    checkpoint_weight = float(
        checkpoint_args.get("delay_regularization_weight", 0.0)
    )
    requested_weight = float(args.delay_regularization_weight)
    if checkpoint_weight != requested_weight:
        raise ValueError(
            "Resume checkpoint delay regularization weight does not match this "
            f"run. Checkpoint={checkpoint_weight}, requested={requested_weight}."
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


def effective_delay_regularization(model):
    delay_sum = None
    delay_count = 0
    for stage in model.stages:
        for block in stage:
            raw_delays = (block.D_mid, block.D_out)
            for raw_delay in raw_delays:
                if raw_delay is None:
                    continue
                effective = block._bounded_delay(
                    raw_delay, raw_delay.device, raw_delay.dtype
                )
                delay_sum = effective.sum() if delay_sum is None else delay_sum + effective.sum()
                delay_count += effective.numel()
    if delay_sum is None or delay_count == 0:
        raise ValueError("Delay regularization requires TTFS D_mid/D_out parameters")
    return delay_sum / delay_count


def effective_delay_statistics(model):
    with torch.no_grad():
        mean = float(effective_delay_regularization(model).item())
    return {
        "definition": "mean effective bounded D_mid/D_out delay",
        "overall_mean": mean,
        "per_stage": actual_stage_delays(model),
    }


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Pretrained checkpoint must be a dictionary")
    state_name = "ema" if checkpoint.get("ema") is not None else "model"
    state = checkpoint.get(state_name)
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "Pretrained checkpoint must contain a non-empty EMA or model state"
        )
    return state_name, state


def _legacy_convolution_mode(state, field):
    if field == "dwconv_mode":
        ttfs = any(".dwconv.D_conv" in key for key in state)
        dense = any(
            key.startswith("stages.")
            and ".dwconv.weight" in key
            for key in state
        )
    else:
        ttfs = any(
            key.startswith("downsample_layers.") and key.endswith(".D_conv")
            for key in state
        )
        dense = any(
            key.startswith("downsample_layers.")
            and key.endswith(".0.weight")
            for key in state
        )
    if ttfs == dense:
        raise ValueError(
            f"Cannot determine pretrained checkpoint {field} from state dict"
        )
    return "ttfs" if ttfs else "dense"


def _canonical_pw1_mode(value):
    normalized = str(value).strip().lower().replace("_", " ")
    return "ttfs" if "ttfs" in normalized else normalized


def validate_pretrained_architecture(checkpoint, args):
    _, state = _checkpoint_state(checkpoint)
    architecture = checkpoint.get("architecture") or {}
    if not isinstance(architecture, dict):
        raise ValueError("Pretrained checkpoint architecture must be a dictionary")
    source_args = checkpoint.get("args") or {}
    if not isinstance(source_args, dict):
        raise ValueError("Pretrained checkpoint args must be a dictionary")

    dwconv_mode = str(
        architecture.get(
            "dwconv_mode",
            source_args.get("dwconv_mode") or _legacy_convolution_mode(state, "dwconv_mode"),
        )
    ).lower()
    downsample_mode = str(
        architecture.get(
            "downsample_mode",
            source_args.get("downsample_mode")
            or _legacy_convolution_mode(state, "downsample_mode"),
        )
    ).lower()
    if dwconv_mode != "dense":
        raise ValueError("Pretrained checkpoint must use dense depthwise convolution")
    if downsample_mode != "dense":
        raise ValueError("Pretrained checkpoint must use dense downsampling convolution")
    if args.dwconv_mode != "ttfs" or args.downsample_mode != "ttfs":
        raise ValueError(
            "Pretrained dense conversion requires TTFS depthwise and downsampling modes"
        )

    source_num_classes = architecture.get("num_classes")
    if source_num_classes is None:
        source_num_classes = source_args.get("num_classes")
    if source_num_classes is None and "head.weight" in state:
        source_num_classes = state["head.weight"].shape[0]

    source_residual = str(
        architecture.get(
            "residual_operator",
            source_args.get("residual_operator", "min"),
        )
    ).strip().lower()
    target_residual = str(
        getattr(args, "residual_operator", "min")
    ).strip().lower()
    allow_residual_change = bool(
        getattr(args, "allow_pretrained_residual_operator_change", False)
    )
    if allow_residual_change:
        if source_residual != "min":
            raise ValueError(
                "Residual ablation initialization requires a min source "
                f"checkpoint, got {source_residual!r}"
            )
        if target_residual not in {"mean", "learnable_gate"}:
            raise ValueError(
                "Residual ablation target must be 'mean' or "
                "'learnable_gate'"
            )

    comparisons = {
        "num_classes": (source_num_classes, args.num_classes),
        "dims": (architecture.get("dims", source_args.get("dims")), list(args.dims)),
        "depths": (architecture.get("depths", source_args.get("depths")), list(args.depths)),
        "depthwise_kernel_size": (
            architecture.get("depthwise_kernel_size", source_args.get("dw_kernel_size")),
            args.dw_kernel_size,
        ),
        "pw2_mode": (
            architecture.get("pw2_mode", source_args.get("pw2_mode")),
            args.pw2_mode,
        ),
        "ttfs_norm_mode": (
            architecture.get("ttfs_norm_mode", source_args.get("ttfs_norm_mode")),
            args.ttfs_norm_mode,
        ),
        "final_score_norm": (
            architecture.get("final_score_norm", source_args.get("final_score_norm", False)),
            args.final_score_norm,
        ),
        "input_resolution": (
            architecture.get(
                "input_resolution",
                [args.input_resolution, args.input_resolution],
            ),
            [args.input_resolution, args.input_resolution],
        ),
        "stage_delays": (
            architecture.get("stage_delays", source_args.get("stage_delays")),
            [float(value) for value in args.stage_delays.split(",")],
        ),
    }
    if not allow_residual_change:
        comparisons["residual_operator"] = (
            source_residual,
            target_residual,
        )
    mismatches = []
    for field, (source, target) in comparisons.items():
        if field in {"dims", "depths", "input_resolution"} and source is not None:
            source = list(source)
        if field == "stage_delays" and isinstance(source, str):
            source = [float(value) for value in source.split(",")]
        if source != target:
            mismatches.append(f"{field}: source={source!r}, target={target!r}")

    source_pw1 = _canonical_pw1_mode(source_args.get("pw1_mode", "ttfs"))
    target_pw1 = _canonical_pw1_mode(args.pw1_mode)
    if source_pw1 != target_pw1:
        mismatches.append(
            f"pw1_mode: source={source_pw1!r}, target={target_pw1!r}"
        )

    source_dataset = source_args.get("dataset")
    if source_dataset is not None:
        try:
            source_dataset = dataset_name(source_dataset)
        except argparse.ArgumentTypeError:
            display_aliases = {
                "cifar-10": "cifar10",
                "cifar-100": "cifar100",
                "tiny imagenet": "tinyimagenet",
            }
            source_dataset = display_aliases.get(str(source_dataset).strip().lower())
        if source_dataset != args.dataset:
            mismatches.append(
                f"dataset: source={source_dataset!r}, target={args.dataset!r}"
            )

    if mismatches:
        raise ValueError(
            "Pretrained checkpoint architecture does not match target: "
            + "; ".join(mismatches)
        )


def _dense_to_ttfs_key(source_key):
    fields = source_key.split(".")
    if (
        len(fields) >= 5
        and fields[0] == "stages"
        and fields[3] == "dwconv"
        and fields[4] in {"weight", "bias"}
    ):
        return ".".join(fields[:4] + ["conv"] + fields[4:])
    if (
        len(fields) >= 4
        and fields[0] == "downsample_layers"
        and fields[1] in {"1", "2", "3"}
        and fields[2] == "0"
        and fields[3] in {"weight", "bias"}
    ):
        return ".".join(fields[:3] + ["conv"] + fields[3:])
    return source_key


def convert_dense_checkpoint_to_ttfs(model, checkpoint, args):
    validate_pretrained_architecture(checkpoint, args)
    source_name, source_state = _checkpoint_state(checkpoint)
    target_state = model.state_dict()
    delay_keys = sorted(key for key in target_state if key.endswith(".D_conv"))
    gate_keys = sorted(
        key for key in target_state if key.endswith(".raw_residual_gate")
    )
    converted = {
        key: tensor.clone()
        for key, tensor in target_state.items()
        if key in delay_keys or key in gate_keys
    }
    unused = []
    for source_key, source_tensor in source_state.items():
        target_key = _dense_to_ttfs_key(source_key)
        if target_key not in target_state:
            unused.append(source_key)
            continue
        if source_tensor.shape != target_state[target_key].shape:
            raise ValueError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_state[target_key].shape)}"
            )
        if target_key in converted:
            raise ValueError(f"Source checkpoint unexpectedly supplies {target_key}")
        converted[target_key] = source_tensor

    if unused:
        raise ValueError(
            "Pretrained checkpoint contains unused source parameters: "
            + ", ".join(sorted(unused))
        )
    missing_transfers = sorted(set(target_state) - set(converted))
    if missing_transfers:
        raise ValueError(
            "Pretrained conversion did not initialize target parameters: "
            + ", ".join(missing_transfers)
        )

    incompatible = model.load_state_dict(converted, strict=True)
    return {
        "source_state": source_name,
        "source_parameter_keys": len(source_state),
        "transferred_parameter_keys": len(source_state),
        "initialized_delay_keys": delay_keys,
        "initialized_gate_keys": gate_keys,
        "source_residual_operator": str(
            (checkpoint.get("architecture") or {}).get(
                "residual_operator",
                (checkpoint.get("args") or {}).get("residual_operator", "min"),
            )
        ).strip().lower(),
        "target_residual_operator": str(
            getattr(args, "residual_operator", "min")
        ).strip().lower(),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def validate_ann_pretrained_architecture(checkpoint, args):
    architecture = checkpoint.get("architecture") or {}
    source_args = checkpoint.get("args") or {}
    if not isinstance(architecture, dict) or not isinstance(source_args, dict):
        raise ValueError("ANN checkpoint architecture and args must be dictionaries")
    expected = {
        "model_type": (architecture.get("model_type"), "fully_dense_ann"),
        "num_classes": (architecture.get("num_classes"), args.num_classes),
        "depths": (architecture.get("depths"), list(args.depths)),
        "dims": (architecture.get("dims"), list(args.dims)),
        "kernel_size": (architecture.get("kernel_size"), args.dw_kernel_size),
        "pw1_mode": (architecture.get("pw1_mode"), "dense"),
        "pw2_mode": (architecture.get("pw2_mode"), "dense"),
        "dwconv_mode": (architecture.get("dwconv_mode"), "dense"),
        "downsample_mode": (architecture.get("downsample_mode"), "dense"),
        "residual_operator": (architecture.get("residual_operator"), "sum"),
        "normalization": (architecture.get("normalization"), "layernorm"),
        "activation": (architecture.get("activation"), "gelu"),
        "dataset": (source_args.get("dataset"), args.dataset),
        "split_seed": (source_args.get("split_seed"), args.split_seed),
    }
    mismatches = []
    for field, (source, target) in expected.items():
        if field in {"depths", "dims"} and source is not None:
            source = list(source)
        if source != target:
            mismatches.append(f"{field}: source={source!r}, target={target!r}")
    target_requirements = {
        "pw1_mode": _canonical_pw1_mode(args.pw1_mode),
        "pw2_mode": args.pw2_mode,
        "dwconv_mode": args.dwconv_mode,
        "downsample_mode": args.downsample_mode,
        "residual_operator": args.residual_operator,
        "ttfs_norm_mode": args.ttfs_norm_mode,
        "final_score_norm": args.final_score_norm,
    }
    required_target = {
        "pw1_mode": "ttfs",
        "pw2_mode": "ttfs",
        "dwconv_mode": "ttfs",
        "downsample_mode": "ttfs",
        "residual_operator": "min",
        "ttfs_norm_mode": "score_layernorm",
        "final_score_norm": True,
    }
    for field, value in target_requirements.items():
        if value != required_target[field]:
            mismatches.append(
                f"target {field}: actual={value!r}, required={required_target[field]!r}"
            )
    if mismatches:
        raise ValueError(
            "ANN checkpoint is incompatible with Fully-TTFS target: "
            + "; ".join(mismatches)
        )


def _ann_to_ttfs_key(source_key):
    fields = source_key.split(".")
    if source_key == "norm.weight":
        return "final_norm.weight"
    if source_key == "norm.bias":
        return "final_norm.bias"
    if len(fields) >= 5 and fields[0] == "stages":
        if fields[3] == "dwconv" and fields[4] in {"weight", "bias"}:
            return ".".join(fields[:4] + ["conv"] + fields[4:])
        if fields[3] == "pwconv1":
            return ".".join(fields[:3] + ["pw1"] + fields[4:])
        if fields[3] == "pwconv2":
            return ".".join(fields[:3] + ["pw2"] + fields[4:])
    if (
        len(fields) == 4
        and fields[0] == "downsample_layers"
        and fields[1] in {"1", "2", "3"}
        and fields[2] == "1"
        and fields[3] in {"weight", "bias"}
    ):
        return ".".join(fields[:2] + ["0", "conv", fields[3]])
    return source_key


def convert_ann_checkpoint_to_ttfs(model, checkpoint, args):
    validate_ann_pretrained_architecture(checkpoint, args)
    source_name, source_state = _checkpoint_state(checkpoint)
    if source_name != "ema":
        raise ValueError("ANN conversion requires authoritative EMA weights")
    target_state = model.state_dict()
    delay_keys = sorted(
        key for key in target_state
        if key.endswith(".D_conv") or key.endswith(".D_mid") or key.endswith(".D_out")
    )
    if len(delay_keys) != 39:
        raise ValueError(f"Fully-TTFS target must contain 39 delay tensors, got {len(delay_keys)}")
    converted = {key: target_state[key].clone() for key in delay_keys}
    excluded = []
    transferred = []
    for source_key, source_tensor in source_state.items():
        fields = source_key.split(".")
        ann_only_norm = (
            len(fields) == 4
            and fields[0] == "downsample_layers"
            and (
                (fields[1] == "0" and fields[2] == "1")
                or (fields[1] in {"1", "2", "3"} and fields[2] == "0")
            )
            and fields[3] in {"weight", "bias"}
        )
        if ann_only_norm:
            excluded.append(source_key)
            continue
        target_key = _ann_to_ttfs_key(source_key)
        if target_key not in target_state:
            raise ValueError(
                f"ANN checkpoint parameter has no Fully-TTFS mapping: {source_key}"
            )
        if source_tensor.shape != target_state[target_key].shape:
            raise ValueError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_state[target_key].shape)}"
            )
        if target_key in converted:
            raise ValueError(f"Duplicate Fully-TTFS target parameter: {target_key}")
        converted[target_key] = source_tensor.detach().clone()
        transferred.append({"source": source_key, "target": target_key})
    expected_excluded = 2 * (1 + 3)
    if len(excluded) != expected_excluded:
        raise ValueError(
            f"Expected exactly {expected_excluded} ANN-only normalization tensors, "
            f"got {len(excluded)}"
        )
    missing = sorted(set(target_state) - set(converted))
    if missing:
        raise ValueError(
            "ANN conversion did not initialize target parameters: " + ", ".join(missing)
        )
    incompatible = model.load_state_dict(converted, strict=True)
    return {
        "initialization_type": "fully_dense_ann_to_fully_ttfs",
        "source_state": source_name,
        "source_parameter_keys": len(source_state),
        "transferred_parameter_keys": len(transferred),
        "parameter_mapping": transferred,
        "excluded_source_keys": sorted(excluded),
        "initialized_delay_keys": delay_keys,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_best_validation_accuracy": checkpoint.get(
            "best_validation_accuracy"
        ),
    }


def validate_constrained_finetune_architecture(checkpoint, args):
    _, state = _checkpoint_state(checkpoint)
    architecture = checkpoint.get("architecture") or {}
    source_args = checkpoint.get("args") or {}
    if not isinstance(architecture, dict) or not isinstance(source_args, dict):
        raise ValueError("Constrained fine-tune checkpoint metadata is invalid")

    source_dwconv = str(
        architecture.get(
            "dwconv_mode",
            source_args.get("dwconv_mode")
            or _legacy_convolution_mode(state, "dwconv_mode"),
        )
    ).lower()
    source_downsample = str(
        architecture.get(
            "downsample_mode",
            source_args.get("downsample_mode")
            or _legacy_convolution_mode(state, "downsample_mode"),
        )
    ).lower()
    source_pw1 = _canonical_pw1_mode(source_args.get("pw1_mode", "ttfs"))
    source_pw2 = str(
        architecture.get("pw2_mode", source_args.get("pw2_mode", "ttfs"))
    ).lower()
    if (source_dwconv, source_downsample, source_pw1, source_pw2) != (
        "ttfs", "ttfs", "ttfs", "ttfs"
    ):
        raise ValueError(
            "Constrained fine-tune source must be fully TTFS for "
            "PW1, PW2, depthwise convolution, and downsampling"
        )
    if (args.dwconv_mode, args.downsample_mode, _canonical_pw1_mode(args.pw1_mode), args.pw2_mode) != (
        "ttfs", "ttfs", "ttfs", "ttfs"
    ):
        raise ValueError("Constrained fine-tune target must be fully TTFS")

    source_all_positive = bool(
        architecture.get(
            "force_positive_weights",
            source_args.get("force_positive_weights", False),
        )
    )
    source_pointwise_positive = bool(
        architecture.get(
            "force_positive_pointwise_weights",
            source_args.get("force_positive_pointwise_weights", False),
        )
    )
    if source_all_positive or source_pointwise_positive:
        raise ValueError(
            "Constrained fine-tune source must be unconstrained"
        )
    if args.force_positive_weights or not getattr(
        args, "force_positive_pointwise_weights", False
    ):
        raise ValueError(
            "Target must enable pointwise-only non-negative weights"
        )

    source_dataset = source_args.get("dataset")
    if source_dataset is not None:
        source_dataset = dataset_name(source_dataset)
    if source_dataset != args.dataset:
        raise ValueError(
            f"Constrained fine-tune dataset mismatch: "
            f"source={source_dataset!r}, target={args.dataset!r}"
        )

    requested = architecture_metadata(args)
    comparisons = {
        "num_classes": architecture.get(
            "num_classes", source_args.get("num_classes")
        ),
        "dims": architecture.get("dims", source_args.get("dims")),
        "depths": architecture.get("depths", source_args.get("depths")),
        "depthwise_kernel_size": architecture.get(
            "depthwise_kernel_size", source_args.get("dw_kernel_size")
        ),
        "residual_operator": architecture.get(
            "residual_operator", source_args.get("residual_operator", "min")
        ),
        "ttfs_norm_mode": architecture.get(
            "ttfs_norm_mode", source_args.get("ttfs_norm_mode")
        ),
        "final_score_norm": architecture.get(
            "final_score_norm", source_args.get("final_score_norm", False)
        ),
        "input_resolution": architecture.get(
            "input_resolution",
            [args.input_resolution, args.input_resolution],
        ),
        "stage_delays": architecture.get(
            "stage_delays", source_args.get("stage_delays")
        ),
    }
    mismatches = []
    for field, source in comparisons.items():
        target = requested[field]
        if field in {"dims", "depths", "input_resolution"} and source is not None:
            source = list(source)
        if field == "stage_delays" and isinstance(source, str):
            source = [float(value) for value in source.split(",")]
        if source != target:
            mismatches.append(f"{field}: source={source!r}, target={target!r}")
    if mismatches:
        raise ValueError(
            "Constrained fine-tune checkpoint architecture mismatch: "
            + "; ".join(mismatches)
        )


def initialize_constrained_finetune(model, checkpoint, args):
    validate_constrained_finetune_architecture(checkpoint, args)
    source_name, source_state = _checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(source_state, strict=True)
    return {
        "kind": "fully_ttfs_to_nonnegative_pointwise",
        "source_state": source_name,
        "transferred_parameter_keys": len(source_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def apply_pretrained_lineage(args, checkpoint):
    initialization = checkpoint.get("pretrained_initialization")
    if not isinstance(initialization, dict):
        return None
    initialization = copy.deepcopy(initialization)
    args.pretrained_initialization = initialization
    source_checkpoint = initialization.get("source_checkpoint")
    if source_checkpoint:
        if initialization.get("initialization_type") == "fully_dense_ann_to_fully_ttfs":
            args.ann_pretrained_checkpoint = str(source_checkpoint)
        else:
            args.pretrained_checkpoint = str(source_checkpoint)
    return initialization


def apply_constrained_finetune_lineage(args, checkpoint):
    initialization = checkpoint.get("constrained_finetune_initialization")
    if not isinstance(initialization, dict):
        return None
    initialization = copy.deepcopy(initialization)
    args.constrained_finetune_initialization = initialization
    source_checkpoint = initialization.get("source_checkpoint")
    if source_checkpoint:
        args.constrained_finetune_checkpoint = str(source_checkpoint)
    return initialization


def delay_gradient_diagnostic(model, args, device):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    was_training = model.training
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        images = torch.rand(
            2,
            3,
            args.input_resolution,
            args.input_resolution,
            device=device,
        )
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
        convolution_delay_gradient_norms = []
        for module_name, module in model.named_modules():
            if module.__class__.__name__ != "ContinuousTTFSConv2d":
                continue
            if module.D_conv.grad is None:
                raise RuntimeError(
                    f"Missing TTFS convolution delay gradient: {module_name}"
                )
            if not torch.isfinite(module.D_conv.grad).all():
                raise RuntimeError(
                    f"Non-finite TTFS convolution delay gradient: {module_name}"
                )
            convolution_delay_gradient_norms.append(
                {
                    "module": module_name,
                    "gradient_norm": float(module.D_conv.grad.float().norm().item()),
                    "effective_delay": module.effective_delay_mean(),
                    "groups": module.groups,
                }
            )
        return {
            "logits_shape": list(logits.shape),
            "logits_finite": True,
            "delay_gradient_norms": stage_norms,
            "layernorm_gradient_norms": norm_gradient_norms,
            "final_layernorm_gradient_norms": final_norm_gradient_norms,
            "convolution_delay_gradient_norms": convolution_delay_gradient_norms,
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
            "status": (
                "resumed"
                if args.resume
                else "fine_tuning"
                if (args.pretrained_checkpoint or args.ann_pretrained_checkpoint)
                else "running"
            ),
            "updated_at": local_timestamp(),
        },
        "dataset": {
            "dataset_name": args.dataset_display_name,
            "number_of_classes": args.num_classes,
            "input_resolution": [args.input_resolution, args.input_resolution],
            "train_sample_count": train_sample_count,
            "validation_sample_count": validation_sample_count,
            "test_sample_count": test_sample_count,
            "preprocessing": (
                "augmentation, ToTensor/RandomErasing, optional Mixup/CutMix, "
                "then continuous TTFS encoding"
            ),
            "augmentation": (
                f"training: RandomCrop({args.input_resolution},padding="
                f"{args.crop_padding}), RandomHorizontalFlip, "
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
            "depthwise_mode": args.dwconv_mode,
            "downsample_kernel": 3,
            "downsample_stride": 2,
            "downsample_padding": 1,
            "downsample_mode": args.downsample_mode,
            "residual_operator": getattr(args, "residual_operator", "min"),
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
            "convolution_delay_learning_rate": args.conv_delay_lr,
            "delay_regularization_weight": args.delay_regularization_weight,
            "delay_regularization_definition": (
                "mean effective bounded D_mid/D_out delay"
            ),
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
            "pretrained_checkpoint": args.pretrained_checkpoint or None,
            "ann_pretrained_checkpoint": args.ann_pretrained_checkpoint or None,
            "pretrained_initialization": getattr(
                args, "pretrained_initialization", None
            ),
            "constrained_finetune_initialization": getattr(
                args, "constrained_finetune_initialization", None
            ),
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
    delay_regularization_weight = float(
        getattr(args, "delay_regularization_weight", 0.0)
    )
    model.train(training)
    total = 0
    weighted_correct = 0.0
    loss_sum = 0.0
    classification_loss_sum = 0.0
    delay_regularization_sum = 0.0
    weighted_delay_regularization_sum = 0.0
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
            classification_loss = lam * criterion(output, labels_a) + (1.0 - lam) * criterion(
                output, labels_b
            )
            if training:
                delay_regularization = effective_delay_regularization(model)
                weighted_delay_regularization = (
                    delay_regularization_weight * delay_regularization
                )
            else:
                delay_regularization = classification_loss.new_zeros(())
                weighted_delay_regularization = classification_loss.new_zeros(())
            loss = classification_loss + weighted_delay_regularization

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
        if training:
            metric_values = torch.stack(
                (
                    loss.detach(),
                    classification_loss.detach(),
                    delay_regularization.detach(),
                    weighted_delay_regularization.detach(),
                )
            ).float().cpu().tolist()
        else:
            loss_value = float(loss.detach().item())
            metric_values = (loss_value, loss_value, 0.0, 0.0)
        loss_sum += metric_values[0] * batch_size
        classification_loss_sum += metric_values[1] * batch_size
        delay_regularization_sum += metric_values[2] * batch_size
        weighted_delay_regularization_sum += metric_values[3] * batch_size
        if iteration % args.print_freq == 0:
            print(
                json.dumps(
                    {
                        "phase": "train" if training else "validation",
                        "iteration": iteration,
                        "loss": loss_sum / total,
                        "classification_loss": classification_loss_sum / total,
                        "delay_regularization": delay_regularization_sum / total,
                        "weighted_delay_regularization": (
                            weighted_delay_regularization_sum / total
                        ),
                        "total_loss": loss_sum / total,
                        "accuracy": 100.0 * weighted_correct / total,
                    }
                ),
                flush=True,
            )

    return {
        "loss": loss_sum / max(total, 1),
        "classification_loss": classification_loss_sum / max(total, 1),
        "delay_regularization": delay_regularization_sum / max(total, 1),
        "weighted_delay_regularization": (
            weighted_delay_regularization_sum / max(total, 1)
        ),
        "total_loss": loss_sum / max(total, 1),
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
            "pretrained_initialization": getattr(
                args, "pretrained_initialization", None
            ),
            "constrained_finetune_initialization": getattr(
                args, "constrained_finetune_initialization", None
            ),
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
    pretrained_initialization = None
    constrained_finetune_initialization = None
    if args.pretrained_checkpoint:
        pretrained_path = Path(args.pretrained_checkpoint)
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {pretrained_path}"
            )
        pretrained_checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        pretrained_initialization = convert_dense_checkpoint_to_ttfs(
            model,
            pretrained_checkpoint,
            args,
        )
        pretrained_initialization["source_checkpoint"] = str(
            pretrained_path.resolve()
        )
        args.pretrained_initialization = pretrained_initialization
        print(
            "Initialized fully TTFS model from dense checkpoint: "
            f"{pretrained_initialization['source_checkpoint']}"
        )
        print(
            "Transferred parameter keys: "
            f"{pretrained_initialization['transferred_parameter_keys']}; "
            "initialized convolution delay tensors: "
            f"{len(pretrained_initialization['initialized_delay_keys'])}"
        )
    if args.ann_pretrained_checkpoint:
        ann_pretrained_path = Path(args.ann_pretrained_checkpoint)
        if not ann_pretrained_path.is_file():
            raise FileNotFoundError(
                f"ANN pretrained checkpoint does not exist: {ann_pretrained_path}"
            )
        ann_pretrained_checkpoint = torch.load(
            ann_pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        pretrained_initialization = convert_ann_checkpoint_to_ttfs(
            model,
            ann_pretrained_checkpoint,
            args,
        )
        pretrained_initialization["source_checkpoint"] = str(
            ann_pretrained_path.resolve()
        )
        args.pretrained_initialization = pretrained_initialization
        print(
            "Initialized Fully-TTFS model from fully dense ANN checkpoint: "
            f"{pretrained_initialization['source_checkpoint']}"
        )
        print(
            "Transferred parameter keys: "
            f"{pretrained_initialization['transferred_parameter_keys']}; "
            "excluded ANN-only normalization tensors: "
            f"{len(pretrained_initialization['excluded_source_keys'])}; "
            "initialized TTFS delay tensors: "
            f"{len(pretrained_initialization['initialized_delay_keys'])}"
        )
    if args.constrained_finetune_checkpoint:
        constrained_path = Path(args.constrained_finetune_checkpoint)
        if not constrained_path.is_file():
            raise FileNotFoundError(
                "Constrained fine-tune checkpoint does not exist: "
                f"{constrained_path}"
            )
        constrained_checkpoint = torch.load(
            constrained_path,
            map_location="cpu",
            weights_only=False,
        )
        constrained_finetune_initialization = initialize_constrained_finetune(
            model,
            constrained_checkpoint,
            args,
        )
        constrained_finetune_initialization["source_checkpoint"] = str(
            constrained_path.resolve()
        )
        args.constrained_finetune_initialization = (
            constrained_finetune_initialization
        )
        print(
            "Initialized pointwise-constrained fully TTFS model from: "
            f"{constrained_finetune_initialization['source_checkpoint']}"
        )

    # Verify native input resolution, unchanged stride-1 stem, and stage schedule.
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
        model(
            encode(
                torch.rand(
                    1,
                    3,
                    args.input_resolution,
                    args.input_resolution,
                    device=device,
                ),
                args,
            )
        )
    for handle in handles:
        handle.remove()
    print("Runtime downsample shapes:", shapes)
    expected_shapes = {
        index: (
            1,
            args.dims[index],
            args.input_resolution // (2 ** index),
            args.input_resolution // (2 ** index),
        )
        for index in range(4)
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
        f"input_resolution={args.input_resolution}x{args.input_resolution}"
    )
    print(
        f"Architecture: dims={args.dims}, depths={args.depths}, "
        f"parameters={parameter_count:,}"
    )
    print(f"TTFS normalization mode: {args.ttfs_norm_mode}")
    print(f"Depthwise convolution mode: {args.dwconv_mode}")
    print(f"Downsampling convolution mode: {args.downsample_mode}")
    print(
        "Non-negative effective pointwise weights: "
        f"{args.force_positive_pointwise_weights}"
    )
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
    optimizer = build_optimizer(model, args)
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

    if args.constrained_finetune_checkpoint:
        initial_directory = output_dir / "initial_constraint"
        initial_directory.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            initial_directory / "initial_constrained_checkpoint.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            ema,
            -1,
            -1.0,
            -1,
            0,
            args,
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        restored_initialization = apply_pretrained_lineage(args, checkpoint)
        if restored_initialization is not None:
            pretrained_initialization = restored_initialization
        restored_constraint = apply_constrained_finetune_lineage(args, checkpoint)
        if restored_constraint is not None:
            constrained_finetune_initialization = restored_constraint
        validate_resume_architecture(checkpoint, args)
        validate_resume_training_configuration(checkpoint, args)
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
            "input_resolution": [args.input_resolution, args.input_resolution],
        },
        "dims": list(args.dims),
        "depths": list(args.depths),
        "input_resolution": [args.input_resolution, args.input_resolution],
        "stem": {
            "in_channels": 3,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "out_channels": args.dims[0],
        },
        "spatial_schedule": [
            args.input_resolution,
            args.input_resolution,
            args.input_resolution // 2,
            args.input_resolution // 4,
            args.input_resolution // 8,
        ],
        "depthwise_kernel_size": args.dw_kernel_size,
        "depthwise_mode": args.dwconv_mode,
        "downsampling": {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "mode": args.downsample_mode,
        },
        "actual_stage_delays": measured_delays,
        "delay_parameterization": "max_delay * sigmoid(raw_delay)",
        "delay_regularization_weight": args.delay_regularization_weight,
        "delay_regularization_definition": (
            "mean effective bounded D_mid/D_out delay"
        ),
        "delay_gradient_norms_test_backward": delay_gradient_norms,
        "ttfs_normalization_mode": args.ttfs_norm_mode,
        "ttfs_normalization_smoke_test": smoke_test,
        "parameter_count": parameter_count,
        "temporal_formulation": "continuous analytic TTFS",
        "simulation_steps": None,
        "mixup_order": "raw images in [0,1], then TTFS encode",
        "pretrained_initialization": pretrained_initialization,
        "constrained_finetune_initialization": (
            constrained_finetune_initialization
        ),
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
            apply_warmup_learning_rates(
                optimizer,
                epoch,
                args.warmup_epochs,
            )
        learning_rate = optimizer.param_groups[0]["lr"]
        learning_rates = {
            group.get("name", f"group_{index}"): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }

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
        next_learning_rates = {
            group.get("name", f"group_{index}"): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }

        log_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "learning_rates": learning_rates,
            "next_learning_rates": next_learning_rates,
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
    final_delay_statistics = effective_delay_statistics(model)
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
        "delay_regularization_weight": args.delay_regularization_weight,
        "delay_regularization_definition": (
            "mean effective bounded D_mid/D_out delay"
        ),
        "final_effective_delays": final_delay_statistics,
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
            "delay_regularization_weight": args.delay_regularization_weight,
            "final_effective_delays": final_delay_statistics,
            "training_time_seconds": previous_training_time
            + (time.time() - tracking_session_started),
            "checkpoint_path": str(best_checkpoint_path.resolve()),
        }
    )
    tracker.save(experiment_report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
