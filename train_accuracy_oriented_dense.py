"""Train the compact fully-dense accuracy-oriented ConvNeXt on CIFAR."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.accuracy_convnext import AccuracyConvNeXt, architecture_metadata


IMAGENET_CONVNEXT_TINY = (
    "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"
)
DATASETS = {
    "cifar10": (datasets.CIFAR10, 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (datasets.CIFAR100, 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


def parse_args():
    parser = argparse.ArgumentParser("Fully-dense accuracy-oriented ConvNeXt")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--data_path", default="../cifar_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--resume", default="")
    parser.add_argument("--imagenet_checkpoint", default="official")
    parser.add_argument("--refinement_checkpoint", default="")
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_transferred", type=float, default=2e-5)
    parser.add_argument("--lr_new", type=float, default=2e-4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_classifier", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--randaugment_num_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--random_erasing", type=float, default=0.1)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument(
        "--augmentation_schedule",
        choices=("static", "refinement60", "lowaug30"),
        default="static",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.resume and args.refinement_checkpoint:
        parser.error("--resume and --refinement_checkpoint are mutually exclusive")
    if args.refinement_checkpoint and args.imagenet_checkpoint not in {"", "official"}:
        parser.error("--refinement_checkpoint cannot be combined with an explicit --imagenet_checkpoint")
    if args.resume and args.imagenet_checkpoint not in {"", "official"}:
        parser.error("--resume cannot be combined with an explicit --imagenet_checkpoint")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args, augmentation=None):
    dataset_class, _, mean, std = DATASETS[args.dataset]
    augmentation = augmentation or {
        "randaugment_enabled": True,
        "randaugment_magnitude": args.randaugment_magnitude,
        "random_erasing": args.random_erasing,
    }
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation.get("randaugment_enabled", True):
        train_transforms.append(transforms.RandAugment(
            args.randaugment_num_ops, int(augmentation["randaugment_magnitude"])
        ))
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=float(augmentation["random_erasing"])),
    ])
    train_transform = transforms.Compose(train_transforms)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_augmented = dataset_class(args.data_path, train=True, transform=train_transform, download=args.download)
    train_clean = dataset_class(args.data_path, train=True, transform=eval_transform, download=args.download)
    test_dataset = dataset_class(args.data_path, train=False, transform=eval_transform, download=args.download)
    indices = torch.randperm(len(train_augmented), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    validation_indices, training_indices = indices[:5000], indices[5000:]
    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(Subset(train_augmented, training_indices), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **common)
    validation_loader = DataLoader(Subset(train_clean, validation_indices), batch_size=args.batch_size,
                                   shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


REFINEMENT60_PHASES = {
    "strong": {
        "randaugment_enabled": True,
        "mixup_alpha": 0.10,
        "cutmix_alpha": 0.50,
        "randaugment_magnitude": 7,
        "random_erasing": 0.05,
        "label_smoothing": 0.05,
        "drop_path": 0.05,
    },
    "middle": {
        "randaugment_enabled": True,
        "mixup_alpha": 0.05,
        "cutmix_alpha": 0.25,
        "randaugment_magnitude": 4,
        "random_erasing": 0.02,
        "label_smoothing": 0.025,
        "drop_path": 0.025,
    },
    "clean": {
        "randaugment_enabled": True,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "randaugment_magnitude": 2,
        "random_erasing": 0.0,
        "label_smoothing": 0.01,
        "drop_path": 0.0,
    },
}

LOWAUG30_PHASES = {
    "light": {
        "mixup_alpha": 0.02,
        "cutmix_alpha": 0.10,
        "randaugment_enabled": True,
        "randaugment_magnitude": 2,
        "random_erasing": 0.0,
        "label_smoothing": 0.02,
        "drop_path": 0.01,
    },
    "clean_low": {
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "randaugment_enabled": False,
        "randaugment_magnitude": 0,
        "random_erasing": 0.0,
        "label_smoothing": 0.01,
        "drop_path": 0.0,
    },
}


def scheduled_augmentation(args, epoch, schedule_state):
    if args.augmentation_schedule == "static":
        return "static", {
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "randaugment_enabled": True,
            "randaugment_magnitude": args.randaugment_magnitude,
            "random_erasing": args.random_erasing,
            "label_smoothing": args.label_smoothing,
            "drop_path": args.drop_path,
        }
    restored_phase = schedule_state.get("restored_phase")
    if restored_phase:
        phases = (
            LOWAUG30_PHASES
            if args.augmentation_schedule == "lowaug30"
            else REFINEMENT60_PHASES
        )
        return restored_phase, dict(phases[restored_phase])
    if args.augmentation_schedule == "lowaug30":
        phase = "light" if epoch < 10 else "clean_low"
        return phase, dict(LOWAUG30_PHASES[phase])
    if epoch < 10:
        phase = "strong"
    elif epoch < 45:
        phase = "middle"
    else:
        phase = "clean"
    return phase, dict(REFINEMENT60_PHASES[phase])


def set_drop_path_rate(model, maximum_rate):
    blocks = [block for stage in model.stages for block in stage]
    rates = torch.linspace(0, float(maximum_rate), len(blocks)).tolist()
    for block, rate in zip(blocks, rates):
        if hasattr(block.drop_path, "drop_prob"):
            block.drop_path.drop_prob = rate


def update_overfitting_state(
    schedule_state, epoch, phase, validation, best_accuracy,
    accuracy_drop=0.3, restore_phase=None,
):
    previous_loss = schedule_state.get("previous_validation_loss")
    if epoch >= 10 and schedule_state.get("restored_phase") is None:
        if previous_loss is not None and validation["loss"] > previous_loss:
            schedule_state["overfit_counter"] += 1
        else:
            schedule_state["overfit_counter"] = 0
        if (
            schedule_state["overfit_counter"] >= 3
            and validation["accuracy"] <= best_accuracy - accuracy_drop
        ):
            schedule_state["restored_phase"] = restore_phase or (
                "strong" if phase == "middle" else "middle"
            )
    schedule_state["previous_validation_loss"] = validation["loss"]
    return schedule_state


def unwrap_source(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "model_ema", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key], key
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, "root"
    raise RuntimeError("Unsupported ImageNet checkpoint format")


def load_imagenet_source(specification):
    if specification == "official":
        checkpoint = torch.hub.load_state_dict_from_url(IMAGENET_CONVNEXT_TINY, map_location="cpu", check_hash=False)
        source = IMAGENET_CONVNEXT_TINY
    else:
        path = Path(specification)
        if not path.is_file():
            raise FileNotFoundError(f"ImageNet checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = str(path.resolve())
    state, state_key = unwrap_source(checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}, source, state_key


def transfer_imagenet_weights(model, specification):
    source_state, source, state_key = load_imagenet_source(specification)
    target_state = model.state_dict()
    transferred, skipped_source = [], []
    def allowed(key):
        if key in {"norm.weight", "norm.bias"}:
            return True
        if key.startswith("stages."):
            return any(token in key for token in (".norm.", ".pwconv1.", ".pwconv2.", ".gamma"))
        parts = key.split(".")
        return (
            len(parts) == 4 and parts[0] == "downsample_layers"
            and ((parts[1] == "0" and parts[2] == "1") or (parts[1] != "0" and parts[2] == "0"))
        )

    for key, value in source_state.items():
        if allowed(key) and key in target_state and target_state[key].shape == value.shape:
            # The compact model intentionally takes only blocks that exist in its state dict.
            target_state[key] = value.detach().clone()
            transferred.append(key)
        else:
            skipped_source.append(key)
    model.load_state_dict(target_state, strict=True)
    transferred_set = set(transferred)
    newly_initialized = sorted(set(target_state) - transferred_set)
    if not transferred or not any("pwconv1.weight" in key for key in transferred):
        raise RuntimeError("ImageNet transfer did not consume compatible pointwise weights")
    return {
        "source": source,
        "source_state_key": state_key,
        "transferred_keys": sorted(transferred),
        "newly_initialized_keys": newly_initialized,
        "skipped_source_keys": sorted(skipped_source),
        "transferred_count": len(transferred),
        "newly_initialized_count": len(newly_initialized),
    }


def initialize_refinement(model, checkpoint_name, dataset):
    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Refinement checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_architecture = checkpoint.get("architecture")
    target_architecture = architecture_metadata(model)
    if source_architecture != target_architecture:
        raise RuntimeError(
            "Refinement checkpoint architecture mismatch: "
            f"source={source_architecture}, target={target_architecture}"
        )
    source_args = checkpoint.get("args", {})
    if source_args.get("dataset") not in {None, dataset}:
        raise RuntimeError(
            f"Refinement checkpoint dataset mismatch: {source_args.get('dataset')} != {dataset}"
        )
    state = checkpoint.get("ema")
    if not isinstance(state, dict):
        raise RuntimeError("Refinement checkpoint is missing authoritative EMA weights")
    model.load_state_dict(state, strict=True)
    return {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_weights": "ema",
        "source_best_epoch": checkpoint.get("best_epoch"),
        "source_best_validation_accuracy": checkpoint.get("best_validation_accuracy"),
        "fresh_training_state": True,
    }


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.mul_(self.decay).add_(source[key].detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(source[key])


def mixed_batch(images, labels, args):
    mixup_alpha = float(args.mixup_alpha)
    cutmix_alpha = float(args.cutmix_alpha)
    if images.size(0) < 2 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return images, labels, labels, 1.0
    permutation = torch.randperm(images.size(0), device=images.device)
    use_mixup = mixup_alpha > 0.0 and (
        cutmix_alpha <= 0.0 or random.getrandbits(1)
    )
    if use_mixup:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        return lam * images + (1 - lam) * images[permutation], labels, labels[permutation], lam
    lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
    height, width = images.shape[-2:]
    ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    cx, cy = random.randrange(width), random.randrange(height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, labels, labels[permutation], lam


def run_epoch(model, loader, device, criterion, amp, optimizer=None, scaler=None, ema=None, args=None):
    training = optimizer is not None
    model.train(training)
    total = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        target_a, target_b, lam = labels, labels, 1.0
        if training:
            images, target_a, target_b, lam = mixed_batch(images, labels, args)
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
        total += labels.numel()
        correct += logits.argmax(1).eq(labels).sum().item()
        loss_sum += loss.item() * labels.numel()
    return {"loss": loss_sum / total, "accuracy": 100.0 * correct / total, "samples": total}


def lr_lambda(epoch, warmup, epochs, minimum_ratio):
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    _, num_classes, _, _ = DATASETS[args.dataset]
    model = AccuracyConvNeXt(num_classes=num_classes, drop_path_rate=args.drop_path)
    start_epoch, best_epoch, best_accuracy, stale_epochs = 0, -1, float("-inf"), 0
    schedule_state = {
        "active_phase": None,
        "restored_phase": None,
        "overfit_counter": 0,
        "previous_validation_loss": None,
    }
    transfer = None
    refinement = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture", {}).get("model_type") != "fully_dense_ann":
            raise RuntimeError("Resume checkpoint is not a fully-dense ANN checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        transfer = checkpoint["pretrained_transfer"]
        refinement = checkpoint.get("refinement_initialization")
    elif args.refinement_checkpoint:
        refinement = initialize_refinement(model, args.refinement_checkpoint, args.dataset)
        source_checkpoint = torch.load(
            args.refinement_checkpoint, map_location="cpu", weights_only=False
        )
        transfer = source_checkpoint.get("pretrained_transfer")
        if not isinstance(transfer, dict):
            raise RuntimeError("Refinement source is missing pretrained-transfer lineage")
    else:
        transfer = transfer_imagenet_weights(model, args.imagenet_checkpoint)
    if refinement is not None:
        backbone_params = [parameter for name, parameter in model.named_parameters() if not name.startswith("head.")]
        classifier_params = [parameter for name, parameter in model.named_parameters() if name.startswith("head.")]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_backbone, "initial_lr": args.lr_backbone, "name": "backbone"},
            {"params": classifier_params, "lr": args.lr_classifier, "initial_lr": args.lr_classifier, "name": "classifier"},
        ], weight_decay=args.weight_decay)
        scheduler_lrs = (args.lr_backbone, args.lr_classifier)
    else:
        transferred_names = set(transfer["transferred_keys"])
        transferred_params, new_params = [], []
        for name, parameter in model.named_parameters():
            (transferred_params if name in transferred_names else new_params).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": transferred_params, "lr": args.lr_transferred, "initial_lr": args.lr_transferred, "name": "transferred"},
            {"params": new_params, "lr": args.lr_new, "initial_lr": args.lr_new, "name": "new"},
        ], weight_decay=args.weight_decay)
        scheduler_lrs = (args.lr_transferred, args.lr_new)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        lambda epoch, target=scheduler_lrs[0]: lr_lambda(epoch, args.warmup_epochs, args.epochs, args.min_lr / target),
        lambda epoch, target=scheduler_lrs[1]: lr_lambda(epoch, args.warmup_epochs, args.epochs, args.min_lr / target),
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.module.load_state_dict(checkpoint["ema"], strict=True)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_accuracy = float(checkpoint["best_validation_accuracy"])
        stale_epochs = int(checkpoint["stale_epochs"])
        schedule_state.update(checkpoint.get("augmentation_schedule_state", {}))
    model.to(device)
    ema.module.to(device)
    initial_phase, initial_augmentation = scheduled_augmentation(
        args, start_epoch, schedule_state
    )
    train_loader, validation_loader, test_loader = build_loaders(
        args, initial_augmentation
    )
    loaded_phase = initial_phase
    validation_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    config = vars(args) | {
        "architecture": architecture_metadata(model),
        "pretrained_transfer": transfer,
        "refinement_initialization": refinement,
        "augmentation_schedule": {
            "static": None,
            "refinement60": REFINEMENT60_PHASES,
            "lowaug30": LOWAUG30_PHASES,
        }[args.augmentation_schedule],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        phase, augmentation = scheduled_augmentation(args, epoch, schedule_state)
        if phase != loaded_phase:
            train_loader, validation_loader, test_loader = build_loaders(
                args, augmentation
            )
            loaded_phase = phase
        schedule_state["active_phase"] = phase
        set_drop_path_rate(model, augmentation["drop_path"])
        epoch_args = copy.copy(args)
        epoch_args.mixup_alpha = augmentation["mixup_alpha"]
        epoch_args.cutmix_alpha = augmentation["cutmix_alpha"]
        training_criterion = nn.CrossEntropyLoss(
            label_smoothing=augmentation["label_smoothing"]
        )
        train_metrics = run_epoch(
            model, train_loader, device, training_criterion, args.amp,
            optimizer, scaler, ema, epoch_args
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                ema.module, validation_loader, device,
                validation_criterion, args.amp
            )
        improved = validation_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch, stale_epochs = validation_metrics["accuracy"], epoch, 0
        else:
            stale_epochs += 1
        if args.augmentation_schedule != "static":
            update_overfitting_state(
                schedule_state, epoch, phase,
                validation_metrics, best_accuracy,
                accuracy_drop=(
                    0.2 if args.augmentation_schedule == "lowaug30" else 0.3
                ),
                restore_phase=(
                    "light" if args.augmentation_schedule == "lowaug30" else None
                ),
            )
        else:
            schedule_state["previous_validation_loss"] = validation_metrics["loss"]
        scheduler.step()
        state = {
            "model": model.state_dict(), "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy, "stale_epochs": stale_epochs,
            "architecture": architecture_metadata(model), "args": vars(args),
            "pretrained_transfer": transfer,
            "refinement_initialization": refinement,
            "augmentation_schedule_state": dict(schedule_state),
        }
        torch.save(state, output_dir / "last_checkpoint.pth")
        if improved:
            torch.save(state, output_dir / "best_checkpoint.pth")
        row = {"epoch": epoch, "learning_rates": [g["lr"] for g in optimizer.param_groups],
               "train": train_metrics, "validation": validation_metrics,
               "augmentation_phase": phase,
               "augmentation": augmentation,
               "augmentation_schedule_state": dict(schedule_state),
               "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch}
        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if stale_epochs >= args.early_stopping_patience:
            stopped_early = True
            break
    best = torch.load(output_dir / "best_checkpoint.pth", map_location="cpu", weights_only=False)
    ema.module.load_state_dict(best["ema"], strict=True)
    ema.module.to(device)
    with torch.inference_mode():
        test_metrics = run_epoch(
            ema.module, test_loader, device, validation_criterion, args.amp
        )
    summary = {
        "dataset": args.dataset, "seed": args.seed, "split_seed": args.split_seed,
        "training_stage": "checkpoint_refinement" if refinement is not None else "imagenet_initialization",
        "refinement_initialization": refinement,
        "augmentation_schedule_state": schedule_state,
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
        "test_metrics": test_metrics, "last_epoch": epoch,
        "early_stopped": stopped_early, "training_time_seconds": time.time() - started,
        "best_checkpoint": str((output_dir / "best_checkpoint.pth").resolve()),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
