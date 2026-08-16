
# evaluate_sparsity.py

import argparse
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Go back from Evaluation/ to project root ConvNeXt/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from models.convnext import (
    ConvNeXtSpiking,
    SpikingBlock,
    ContinuousTTFSConv2d,
)
class _TerminalAndFileWriter:
    def __init__(self, terminal, report_file):
        self.terminal = terminal
        self.report_file = report_file

    def write(self, text):
        self.terminal.write(text)
        return self.report_file.write(text)

    def flush(self):
        self.terminal.flush()
        self.report_file.flush()


def report_path_for_checkpoint(checkpoint_path):
    return Path(checkpoint_path).parent / "activation_sparsity.md"


@contextmanager
def markdown_output(report_path, terminal=None):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    terminal = terminal or sys.stdout

    with report_path.open("w", encoding="utf-8", newline="\n") as report_file:
        report_file.write("# Activation Sparsity Evaluation\n\n```text\n")
        writer = _TerminalAndFileWriter(terminal, report_file)
        try:
            with redirect_stdout(writer):
                yield
        finally:
            report_file.write("```\n")


# ============================================================
# Dataset information
# ============================================================

def get_dataset_info(dataset_name):
    name = dataset_name.lower()

    if name == "cifar10":
        return {
            "num_classes": 10,
            "input_size": 32,
        }

    if name == "cifar100":
        return {
            "num_classes": 100,
            "input_size": 32,
        }

    if name in {"tinyimagenet", "tiny_imagenet", "tiny-imagenet"}:
        return {
            "num_classes": 200,
            "input_size": 64,
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ============================================================
# Checkpoint utilities
# ============================================================

def find_checkpoint(path):
    path = Path(path)

    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist:\n{path}"
        )

    candidates = []

    # Prefer best checkpoints first
    for pattern in [
        "*best*.pth",
        "*best*.pt",
        "*.pth",
        "*.pt",
    ]:
        candidates.extend(path.rglob(pattern))

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        raise FileNotFoundError(
            f"No .pth or .pt checkpoint found inside:\n{path}"
        )

    best_candidates = [
        p for p in candidates
        if "best" in p.name.lower()
    ]

    if best_candidates:
        best_candidates.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        checkpoint = best_candidates[0]
    else:
        candidates.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        checkpoint = candidates[0]

    return checkpoint


def extract_state_dict(checkpoint):
    """
    Supports common checkpoint formats:
      model.state_dict()
      {"model": state_dict}
      {"state_dict": state_dict}
      {"model_state_dict": state_dict}
      {"model_ema": state_dict}
      {"ema": state_dict}
    """

    if not isinstance(checkpoint, dict):
        return checkpoint

    possible_keys = [
        "model_ema",
        "ema_state_dict",
        "ema",
        "model",
        "model_state_dict",
        "state_dict",
    ]

    for key in possible_keys:
        if key not in checkpoint:
            continue

        value = checkpoint[key]

        if not isinstance(value, dict):
            continue

        # Nested formats
        for inner_key in [
            "module",
            "model",
            "state_dict",
        ]:
            if (
                inner_key in value
                and isinstance(value[inner_key], dict)
            ):
                nested = value[inner_key]

                if any(
                    isinstance(v, torch.Tensor)
                    for v in nested.values()
                ):
                    return nested

        # Direct state dict
        if any(
            isinstance(v, torch.Tensor)
            for v in value.values()
        ):
            return value

    # Checkpoint itself may already be state_dict
    if any(
        isinstance(v, torch.Tensor)
        for v in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not find a model state_dict in checkpoint."
    )


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def _infer_convolution_mode(field, state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Cannot infer missing {field}: checkpoint state dict is unavailable."
        )

    keys = tuple(state_dict)
    if field == "dwconv_mode":
        ttfs_marker = any(key.endswith(".dwconv.D_conv") for key in keys)
        dense_marker = any(key.endswith(".dwconv.weight") for key in keys)
    elif field == "downsample_mode":
        ttfs_marker = any(
            "downsample_layers." in key and key.endswith(".D_conv")
            for key in keys
        )
        dense_marker = any(
            "downsample_layers." in key and key.endswith(".0.weight")
            for key in keys
        )
    else:
        raise ValueError(f"Unsupported convolution mode field: {field}")

    if ttfs_marker == dense_marker:
        marker_state = "conflicting" if ttfs_marker else "unrecognized"
        raise ValueError(
            f"Cannot infer missing {field}: state-dict markers are {marker_state}."
        )

    return "ttfs" if ttfs_marker else "dense"


def get_checkpoint_convolution_modes(checkpoint, state_dict=None):
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must be a dictionary to determine convolution modes."
        )

    architecture = checkpoint.get("architecture")
    if architecture is None:
        architecture = {}
    elif not isinstance(architecture, dict):
        raise ValueError("Checkpoint architecture metadata must be a dictionary.")

    modes = {}
    for field in ("dwconv_mode", "downsample_mode"):
        if field not in architecture:
            modes[field] = _infer_convolution_mode(field, state_dict)
            continue

        value = str(architecture[field]).strip().lower()
        if value not in {"dense", "ttfs"}:
            raise ValueError(
                f"Checkpoint architecture field {field} must be "
                f"'dense' or 'ttfs', got {architecture[field]!r}."
            )
        modes[field] = value

    return modes


def get_checkpoint_residual_operator(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must be a dictionary to determine the residual operator."
        )

    architecture = checkpoint.get("architecture")
    if architecture is None:
        architecture = {}
    elif not isinstance(architecture, dict):
        raise ValueError("Checkpoint architecture metadata must be a dictionary.")

    value = str(architecture.get("residual_operator", "min")).strip().lower()
    if value not in {"min", "mean", "learnable_gate"}:
        raise ValueError(
            "Checkpoint architecture field residual_operator must be "
            f"'min', 'mean', or 'learnable_gate', got {value!r}."
        )
    return value


def get_checkpoint_pointwise_constraint(checkpoint):
    architecture = checkpoint.get("architecture") or {}
    saved_args = checkpoint.get("args") or {}
    if not isinstance(architecture, dict) or not isinstance(saved_args, dict):
        raise ValueError("Checkpoint constraint metadata must be a dictionary")
    value = architecture.get(
        "force_positive_pointwise_weights",
        saved_args.get("force_positive_pointwise_weights", False),
    )
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(
        "Checkpoint force_positive_pointwise_weights must be boolean"
    )


# ============================================================
# Dataset
# ============================================================

def build_loader(
    dataset_name,
    data_path,
    batch_size,
    workers,
):
    name = dataset_name.lower()

    transform = transforms.ToTensor()

    if name == "cifar10":

        dataset = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=False,
            transform=transform,
        )

    elif name == "cifar100":

        dataset = datasets.CIFAR100(
            root=data_path,
            train=False,
            download=False,
            transform=transform,
        )

    elif name in {
        "tinyimagenet",
        "tiny_imagenet",
        "tiny-imagenet",
    }:
        from train_continuous_ttfs_cifar10_32x32_stem1 import (
            TinyImageNetValidationDataset,
            resolve_tinyimagenet_root,
        )

        root = resolve_tinyimagenet_root(data_path)
        train_dataset = datasets.ImageFolder(root / "train")
        if len(train_dataset.classes) != 200:
            raise ValueError(
                "Tiny ImageNet training directory must contain exactly 200 classes, "
                f"found {len(train_dataset.classes)}"
            )
        dataset = TinyImageNetValidationDataset(
            root / "val",
            train_dataset.class_to_idx,
            transform=transform,
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    return dataset, loader


# ============================================================
# Input encoding
# ============================================================

def image_to_spike_time(
    images,
    t_min=0.0,
    t_max=1.0,
):
    """
    Basic TTFS intensity encoding:

        intensity = 1 -> t_min
        intensity = 0 -> t_max

    Earlier spike = stronger input.
    """

    return (
        t_min
        + (1.0 - images)
        * (t_max - t_min)
    )


# ============================================================
# Model construction
# ============================================================

def build_model(
    args,
    convolution_modes,
    residual_operator="min",
    force_positive_pointwise_weights=False,
):
    info = get_dataset_info(args.dataset)

    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=info["num_classes"],

        # Training architecture
        dims=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),

        dw_kernel_size=args.dw_kernel_size,
        downsample_kernel_size=3,

        # Your training script is stem1 / CIFAR-style stem
        cifar_stem=args.cifar_stem,

        dwconv_mode=convolution_modes["dwconv_mode"],
        downsample_mode=convolution_modes["downsample_mode"],
        residual_operator=residual_operator,
        force_positive_pointwise_weights=(
            force_positive_pointwise_weights
        ),
        pw2_mode="ttfs",

        stage_delays=(
            0.05,
            0.02,
            0.01,
            0.01,
        ),

        ttfs_norm_mode="score_layernorm",
        final_score_norm=True,

        head_dropout=0.1,
        spike_dropout=0.0,

        drop_path_rate=0.0,

        t_min=0.0,
        t_max=1.0,
    )

    return model


# ============================================================
# Sparsity counter
# ============================================================

class SparsityCounter:
    def __init__(self):
        self.data = {}

    def add(
        self,
        name,
        tensor,
        t_max,
    ):
        if tensor is None:
            return

        tensor = tensor.detach()

        silent = (
            tensor >= (t_max - 1e-6)
        ).sum().item()

        total = tensor.numel()

        if name not in self.data:
            self.data[name] = {
                "silent": 0,
                "total": 0,
            }

        self.data[name]["silent"] += silent
        self.data[name]["total"] += total

    def sparsity(self, name):
        d = self.data[name]

        if d["total"] == 0:
            return 0.0

        return (
            100.0
            * d["silent"]
            / d["total"]
        )

    def global_sparsity(self):
        total_silent = sum(
            d["silent"]
            for d in self.data.values()
        )

        total_elements = sum(
            d["total"]
            for d in self.data.values()
        )

        if total_elements == 0:
            return 0.0

        return (
            100.0
            * total_silent
            / total_elements
        )


# ============================================================
# Model inspection
# ============================================================

def inspect_model(model):
    spiking_blocks = []

    ttfs_convs = []

    for name, module in model.named_modules():

        if isinstance(module, SpikingBlock):
            spiking_blocks.append(
                (name, module)
            )

        if isinstance(
            module,
            ContinuousTTFSConv2d,
        ):
            ttfs_convs.append(
                (name, module)
            )

    expected_total = (
        len(ttfs_convs)
        + 2 * len(spiking_blocks)
    )

    print()
    print("=" * 90)
    print("MODEL SPARSITY STRUCTURE")
    print("=" * 90)

    print(
        f"Spiking blocks:              "
        f"{len(spiking_blocks)}"
    )

    print(
        f"ContinuousTTFSConv2d:        "
        f"{len(ttfs_convs)}"
    )

    print(
        f"PW1 TTFS outputs:            "
        f"{len(spiking_blocks)}"
    )

    print(
        f"PW2 TTFS outputs:            "
        f"{len(spiking_blocks)}"
    )

    print(
        f"Expected total TTFS points:  "
        f"{expected_total}"
    )

    print("=" * 90)

    print("\nTTFS Conv modules:")

    for name, _ in ttfs_convs:
        print(f"  {name}")

    return spiking_blocks, ttfs_convs, expected_total


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_sparsity(
    model,
    loader,
    device,
    t_min,
    t_max,
):
    model.eval()

    counter = SparsityCounter()

    spiking_blocks, ttfs_convs, expected_total = inspect_model(
        model
    )

    # --------------------------------------------------------
    # Hooks for all ContinuousTTFSConv2d
    #
    # Expected:
    #   12 TTFS depthwise conv
    #    3 TTFS downsample
    #   ----------------------
    #   15 modules
    # --------------------------------------------------------

    handles = []

    def make_hook(layer_name):

        def hook(module, inputs, output):

            counter.add(
                layer_name,
                output,
                t_max,
            )

        return hook

    for name, module in ttfs_convs:

        handle = module.register_forward_hook(
            make_hook(name)
        )

        handles.append(handle)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    correct = 0
    total_samples = 0

    for batch_index, (
        images,
        targets,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # -----------------------------------------------
        # Encode images as TTFS spike times
        # -----------------------------------------------

        spike_times = image_to_spike_time(
            images,
            t_min=t_min,
            t_max=t_max,
        )

        # -----------------------------------------------
        # Forward
        # -----------------------------------------------

        logits = model(spike_times)

        # -----------------------------------------------
        # PW1 and PW2
        #
        # Requires these attributes in SpikingBlock:
        #
        # self.t_mid_spike
        # self.t_out_spike
        # -----------------------------------------------

        for block_name, block in spiking_blocks:

            t_mid = getattr(
                block,
                "t_mid_spike",
                None,
            )

            t_out = getattr(
                block,
                "t_out_spike",
                None,
            )

            if t_mid is not None:

                counter.add(
                    f"{block_name}.pw1_ttfs",
                    t_mid,
                    t_max,
                )

            if t_out is not None:

                counter.add(
                    f"{block_name}.pw2_ttfs",
                    t_out,
                    t_max,
                )

        # -----------------------------------------------
        # Accuracy
        # -----------------------------------------------

        prediction = logits.argmax(dim=1)

        correct += (
            prediction == targets
        ).sum().item()

        total_samples += targets.size(0)

        if (
            batch_index == 0
            or (batch_index + 1) % 20 == 0
            or (batch_index + 1) == len(loader)
        ):

            running_accuracy = (
                100.0
                * correct
                / total_samples
            )

            print(
                f"Batch "
                f"{batch_index + 1:4d}/"
                f"{len(loader):4d} | "
                f"samples={total_samples:6d} | "
                f"accuracy="
                f"{running_accuracy:6.2f}%"
            )

    # Remove hooks
    for handle in handles:
        handle.remove()

    accuracy = (
        100.0
        * correct
        / total_samples
    )

    return counter, accuracy, expected_total


# ============================================================
# Layer classification
# ============================================================

def get_layer_type(name):

    if "downsample_layers" in name:
        return "downsample"

    if ".dwconv" in name:
        return "dwconv"

    if "pw1_ttfs" in name:
        return "pw1"

    if "pw2_ttfs" in name:
        return "pw2"

    return "other"


def get_stage(name):

    if "stages.0." in name:
        return 0

    if "stages.1." in name:
        return 1

    if "stages.2." in name:
        return 2

    if "stages.3." in name:
        return 3

    # Downsampling i feeds Stage i
    if "downsample_layers.1." in name:
        return 1

    if "downsample_layers.2." in name:
        return 2

    if "downsample_layers.3." in name:
        return 3

    return None


# ============================================================
# Weighted summary helper
# ============================================================

def weighted_summary(
    counter,
    names,
):
    silent = 0
    total = 0

    for name in names:

        if name not in counter.data:
            continue

        silent += (
            counter.data[name]["silent"]
        )

        total += (
            counter.data[name]["total"]
        )

    if total == 0:
        return 0.0, silent, total

    sparsity = (
        100.0
        * silent
        / total
    )

    return sparsity, silent, total


# ============================================================
# Report
# ============================================================

def print_report(
    counter,
    accuracy,
    dataset_name,
    expected_points,
):
    print()
    print()
    print("=" * 115)
    print(
        f"TTFS ACTIVATION SPARSITY REPORT "
        f"- {dataset_name.upper()}"
    )
    print("=" * 115)

    print(
        f"{'Layer':65s}"
        f"{'Type':12s}"
        f"{'Silent':>14s}"
        f"{'Total':>14s}"
        f"{'Sparsity':>10s}"
    )

    print("-" * 115)

    for name in sorted(counter.data.keys()):

        d = counter.data[name]

        sparsity = counter.sparsity(
            name
        )

        layer_type = get_layer_type(
            name
        )

        print(
            f"{name:65s}"
            f"{layer_type:12s}"
            f"{d['silent']:14,d}"
            f"{d['total']:14,d}"
            f"{sparsity:9.2f}%"
        )

    print("-" * 115)

    overall = counter.global_sparsity()

    print(
        f"{'GLOBAL WEIGHTED SPARSITY':77s}"
        f"{'':14s}"
        f"{'':14s}"
        f"{overall:9.2f}%"
    )

    print("=" * 115)

    print()
    print(
        f"Classification accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Measured TTFS points:     "
        f"{len(counter.data)}"
    )

    print(
        f"Expected TTFS points:     "
        f"{expected_points}"
    )

    if len(counter.data) != expected_points:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected {expected_points} TTFS outputs but "
            f"measured {len(counter.data)}."
        )

        print(
            "Most likely t_mid_spike and/or "
            "t_out_spike are not being stored "
            "inside SpikingBlock.forward()."
        )

    # ========================================================
    # Layer-type summary
    # ========================================================

    print()
    print("=" * 80)
    print("SPARSITY BY TTFS OPERATION TYPE")
    print("=" * 80)

    for layer_type in [
        "dwconv",
        "pw1",
        "pw2",
        "downsample",
    ]:

        names = [
            name
            for name in counter.data
            if get_layer_type(name)
            == layer_type
        ]

        sparsity, silent, total = (
            weighted_summary(
                counter,
                names,
            )
        )

        print(
            f"{layer_type:15s}"
            f"{len(names):3d} layers | "
            f"sparsity={sparsity:7.2f}% | "
            f"silent={silent:,} | "
            f"total={total:,}"
        )

    print("=" * 80)

    # ========================================================
    # Stage summary
    # ========================================================

    print()
    print("=" * 80)
    print("STAGE-WISE WEIGHTED SPARSITY")
    print("=" * 80)

    for stage in range(4):

        names = [
            name
            for name in counter.data
            if get_stage(name) == stage
        ]

        sparsity, silent, total = (
            weighted_summary(
                counter,
                names,
            )
        )

        print(
            f"Stage {stage}: "
            f"{sparsity:7.2f}% | "
            f"TTFS points={len(names):2d} | "
            f"silent={silent:,} | "
            f"total={total:,}"
        )

    print("=" * 80)

    # ========================================================
    # Paper-ready output
    # ========================================================

    print()
    print("=" * 80)
    print("PAPER-READY SUMMARY")
    print("=" * 80)

    print(
        f"Dataset:              "
        f"{dataset_name}"
    )

    print(
        f"Accuracy:             "
        f"{accuracy:.2f}%"
    )

    print(
        f"Activation sparsity:  "
        f"{overall:.2f}%"
    )

    print(
        f"TTFS layers/points:   "
        f"{len(counter.data)}"
    )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TTFS activation sparsity "
            "for CIFAR-10, CIFAR-100, or "
            "Tiny ImageNet."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='cifar10',
        choices=[
            "cifar10",
            "cifar100",
            "tinyimagenet",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "./results/cifar10/downsample_dense_dwconv_dense/"
            "seed_42/best_checkpoint.pth"
        ),
        help=(
            "Checkpoint file or folder "
            "containing checkpoint."
        ),
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default='../cifar_data/'
    )

    parser.add_argument(
        "--dw_kernel_size",
        type=int,
        default=3,
        help=(
            "Must match training configuration."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--cifar_stem",
        type=lambda x: str(x).lower()
        in {"1", "true", "yes"},
        default=True,
        help=(
            "Must match training. "
            "Use true for your stem1 setup."
        ),
    )

    return parser.parse_args()


def run_evaluation(args, checkpoint_path):

    # ========================================================
    # Device
    # ========================================================

    if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        device = torch.device(
            args.device
        )
    else:
        device = torch.device("cpu")

    print()
    print(
        f"Device: {device}"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset, loader = build_loader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    print(
        f"Evaluation samples: "
        f"{len(dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    convolution_modes = get_checkpoint_convolution_modes(
        checkpoint,
        state_dict,
    )
    residual_operator = get_checkpoint_residual_operator(checkpoint)
    pointwise_constraint = get_checkpoint_pointwise_constraint(checkpoint)
    architecture = checkpoint.get("architecture") or {}
    mode_sources = {
        field: (
            "metadata"
            if field in architecture
            else "legacy state-dict inference"
        )
        for field in ("dwconv_mode", "downsample_mode")
    }

    print()
    print(
        "Detected depthwise convolution mode: "
        f"{convolution_modes['dwconv_mode']} "
        f"({mode_sources['dwconv_mode']})"
    )
    print(
        "Detected downsampling convolution mode: "
        f"{convolution_modes['downsample_mode']} "
        f"({mode_sources['downsample_mode']})"
    )
    residual_source = (
        "metadata"
        if "residual_operator" in architecture
        else "legacy default"
    )
    print(
        "Detected residual operator: "
        f"{residual_operator} ({residual_source})"
    )
    pointwise_source = (
        "metadata"
        if "force_positive_pointwise_weights" in architecture
        else "legacy default"
    )
    print(
        "Detected non-negative effective pointwise weights: "
        f"{pointwise_constraint} ({pointwise_source})"
    )

    model = build_model(
        args,
        convolution_modes,
        residual_operator,
        pointwise_constraint,
    )

    incompatible_keys = model.load_state_dict(
        state_dict,
        strict=True,
    )
    missing = incompatible_keys.missing_keys
    unexpected = incompatible_keys.unexpected_keys

    print()
    print(
        f"Missing keys:    "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(unexpected)}"
    )

    model = model.to(
        device
    )

    # ========================================================
    # Evaluate
    # ========================================================

    counter, accuracy, expected_points = (
        evaluate_sparsity(
            model=model,
            loader=loader,
            device=device,
            t_min=0.0,
            t_max=1.0,
        )
    )

    # ========================================================
    # Report
    # ========================================================

    print_report(
        counter=counter,
        accuracy=accuracy,
        dataset_name=args.dataset,
        expected_points=expected_points,
    )


def main():
    args = parse_args()
    checkpoint_path = find_checkpoint(args.checkpoint)
    report_path = report_path_for_checkpoint(checkpoint_path)

    with markdown_output(report_path):
        print(f"Using checkpoint: {checkpoint_path}")
        run_evaluation(args, checkpoint_path)
        print()
        print(f"Markdown report saved to: {report_path}")


if __name__ == "__main__":
    main()

