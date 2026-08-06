#!/usr/bin/env python3
"""Standalone CIFAR-10 accuracy, TTFS sparsity, SynOps, and dense-MAC evaluator.

Dense convolutions/linears are reported as MACs. Only calls made through
``call_spiking_torch`` inside an actual ``SpikingBlock`` are reported as
theoretical event-driven SynOps. The model and checkpoint are not modified.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models.convnext as convnext_module
from models.convnext import ConvNeXtSpiking, SpikingBlock


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 Spiking ConvNeXt sparsity and operations"
    )
    parser.add_argument("--data_path", default="../cifar_data/", type=str)
    parser.add_argument("--eval_data_path", default=None, type=str)
    parser.add_argument("--data_set", default="CIFAR", choices=["CIFAR"])
    parser.add_argument("--nb_classes", default=10, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--batch_size", default=150, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", default=False, type=str2bool)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)

    # Arguments consumed by datasets.py/timm transforms.
    parser.add_argument("--imagenet_default_mean_and_std", default=True, type=str2bool)
    parser.add_argument("--crop_pct", default=None, type=float)
    parser.add_argument("--color_jitter", default=0.4, type=float)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--train_interpolation", default="bicubic", type=str)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)
    parser.add_argument("--resplit", default=False, type=str2bool)
    parser.add_argument("--disable_eval", default=False, type=str2bool)

    parser.add_argument("--drop_path", default=0.0, type=float)
    parser.add_argument("--layer_scale_init_value", default=1e-6, type=float)
    parser.add_argument("--head_init_scale", default=1.0, type=float)
    parser.add_argument("--ttfs_tmin", default=0.0, type=float)
    parser.add_argument("--ttfs_tmax", default=1.0, type=float)
    parser.add_argument("--ttfs_force_pos_weights", default=False, type=str2bool)
    parser.add_argument("--ttfs_init_delay", default=0.0, type=float)
    parser.add_argument("--ttfs_stage_delays", default="0.4,0.0,0.0,0.0", type=str)

    parser.add_argument(
        "--load_weights",
        default="./weights/ckpt_residual_Di_96.09/checkpoint_residual_Di_96.09.pth",
        type=str,
    )
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--output_dir", default="./sparsity_synops_results", type=str)
    parser.add_argument("--json_name", default="sparsity_synops_results.json", type=str)
    parser.add_argument("--csv_name", default="sparsity_synops_layers.csv", type=str)
    return parser


def resolve_checkpoint(path):
    resolved = os.path.normpath(path)
    if os.path.isdir(resolved):
        candidates = [
            os.path.join(resolved, name)
            for name in os.listdir(resolved)
            if name.endswith((".pth", ".pt"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No .pth or .pt checkpoint in {resolved}")
        resolved = max(candidates, key=os.path.getmtime)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def load_checkpoint_with_integrity(model, checkpoint_path, model_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in model_key.split("|"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")

    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model_state = model.state_dict()
    unexpected = sorted(key for key in state_dict if key not in model_state)
    shape_mismatches = []
    compatible = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape) if hasattr(value, "shape") else None,
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        compatible[key] = value

    missing = sorted(key for key in model_state if key not in compatible)
    model.load_state_dict(compatible, strict=False)
    integrity = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "loaded_key_count": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    print(f"Checkpoint: {integrity['checkpoint']}")
    print(f"Loaded compatible keys: {len(compatible)}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print("Shape mismatches:", shape_mismatches)
    return integrity


class OperationCollector:
    """Collect actual dense MACs and TTFS event-driven operation statistics."""

    def __init__(self, model, t_max, epsilon=1e-6):
        self.model = model
        self.t_max = float(t_max)
        self.epsilon = float(epsilon)
        self.handles = []
        self.original_call_spiking = None
        self.current_block = None
        self.current_call_index = 0
        self.activations = defaultdict(lambda: defaultdict(int))
        self.operations = defaultdict(lambda: defaultdict(int))
        self.block_names = {
            module: name
            for name, module in model.named_modules()
            if isinstance(module, SpikingBlock)
        }

    def _activation(self, block_name, point, tensor):
        stats = self.activations[(block_name, point)]
        stats["num_total"] += tensor.numel()
        stats["num_silent"] += (tensor >= self.t_max - self.epsilon).sum().item()

    def _block_pre_hook(self, module, inputs):
        self.current_block = module
        self.current_call_index = 0

    def _block_post_hook(self, module, inputs, output):
        self._activation(self.block_names[module], "final", output.detach())
        self.current_block = None
        self.current_call_index = 0

    def _wrapped_call_spiking(self, tj, weight, delays, t_min_prev, t_min, t_max):
        output = self.original_call_spiking(tj, weight, delays, t_min_prev, t_min, t_max)
        block = self.current_block
        if block is None:
            return output

        point = "t_mid_spike" if self.current_call_index == 0 else "t_out_spike"
        operation = "pw1" if self.current_call_index == 0 else "pw2"
        self.current_call_index += 1
        block_name = self.block_names[block]
        self._activation(block_name, point, output.detach())

        # tj is [events, input_features], weight is [input_features, outputs].
        # Each non-silent input event incurs one accumulation for each nonzero
        # outgoing connection. This avoids dense output-neuron/fan-out heuristics.
        input_events_by_feature = (tj < self.t_max - self.epsilon).sum(dim=0).to(torch.int64)
        nonzero_outgoing = (weight != 0).sum(dim=1).to(torch.int64)
        synops = (input_events_by_feature * nonzero_outgoing).sum().item()
        stats = self.operations[(block_name, operation)]
        stats["theoretical_synops"] += int(synops)
        stats["input_events"] += int(input_events_by_feature.sum().item())
        stats["input_neurons"] += int(tj.numel())
        stats["nonzero_weights"] = int((weight != 0).sum().item())
        stats["weight_count"] = int(weight.numel())
        return output

    def _dense_hook(self, name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if isinstance(module, nn.Conv2d):
                kernel_h, kernel_w = module.kernel_size
                macs_per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
                macs = output.numel() * macs_per_output
                operation_type = "depthwise_conv" if module.groups == module.in_channels else "dense_conv"
            else:
                macs = output.numel() * module.in_features
                operation_type = "dense_linear"
            stats = self.operations[(name, operation_type)]
            stats["dense_macs"] += int(macs)
            stats["output_elements"] += int(output.numel())
            if name.endswith(".pw2"):
                block_name = name[:-4]
                t_out = torch.clamp(-output.detach(), 0.0, self.t_max)
                self._activation(block_name, "t_out_spike", t_out)
            if isinstance(module, nn.Conv2d):
                stats["groups"] = int(module.groups)
                stats["kernel_h"] = int(module.kernel_size[0])
                stats["kernel_w"] = int(module.kernel_size[1])
        return hook

    def install(self):
        # Hooks are deliberately limited to the exact active SpikingBlock type.
        for module in self.block_names:
            self.handles.append(module.register_forward_pre_hook(self._block_pre_hook))
            self.handles.append(module.register_forward_hook(self._block_post_hook))

        dense_modules = {}
        for index, sequential in enumerate(self.model.downsample_layers):
            for child_name, child in sequential.named_modules():
                if isinstance(child, nn.Conv2d):
                    name = f"downsample_layers.{index}" + (f".{child_name}" if child_name else "")
                    dense_modules[child] = name
        for block, block_name in self.block_names.items():
            dense_modules[block.dwconv] = f"{block_name}.dwconv"
            # These hooks remain idle for the current active implementation,
            # which accesses pw1/pw2 weights directly through TTFS. If a block
            # actually invokes either Linear module, its cost is dense MACs.
            dense_modules[block.pw1] = f"{block_name}.pw1"
            dense_modules[block.pw2] = f"{block_name}.pw2"
        dense_modules[self.model.head] = "head"
        for module, name in dense_modules.items():
            self.handles.append(module.register_forward_hook(self._dense_hook(name)))

        self.original_call_spiking = convnext_module.call_spiking_torch
        convnext_module.call_spiking_torch = self._wrapped_call_spiking
        print(f"Registered activation hooks on {len(self.block_names)} actual SpikingBlock modules")

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.original_call_spiking is not None:
            convnext_module.call_spiking_torch = self.original_call_spiking
            self.original_call_spiking = None

    def results(self, sample_count):
        activation_rows = []
        for (layer, point), stats in sorted(self.activations.items()):
            total = stats["num_total"]
            silent = stats["num_silent"]
            activation_rows.append(
                {
                    "layer": layer,
                    "measurement": point,
                    "num_total": total,
                    "num_silent": silent,
                    "num_spiking": total - silent,
                    "sparsity_percent": 100.0 * silent / total if total else 0.0,
                }
            )

        operation_rows = []
        for (layer, operation), stats in sorted(self.operations.items()):
            row = {"layer": layer, "operation": operation}
            row.update(stats)
            dense_total = int(row.get("dense_macs", 0))
            synops_total = int(row.get("theoretical_synops", 0))
            row["dense_macs_per_sample"] = dense_total / sample_count
            row["dense_macs_total_dataset"] = dense_total
            row["theoretical_synops_per_sample"] = synops_total / sample_count
            row["theoretical_synops_total_dataset"] = synops_total
            operation_rows.append(row)
        return activation_rows, operation_rows


def evaluate(model, loader, device, t_max):
    collector = OperationCollector(model, t_max)
    collector.install()
    correct = 0
    sample_count = 0
    try:
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                sample_count += labels.size(0)
                if batch_index % 20 == 0 or batch_index == len(loader):
                    print(
                        f"Batch {batch_index}/{len(loader)} | samples={sample_count} "
                        f"| accuracy={100.0 * correct / sample_count:.2f}%"
                    )
    finally:
        collector.remove()
    activations, operations = collector.results(sample_count)
    return 100.0 * correct / sample_count, sample_count, activations, operations


def save_outputs(output_dir, json_name, csv_name, report):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    csv_path = output_dir / csv_name
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    fields = [
        "record_type", "layer", "measurement", "operation", "num_total",
        "num_silent", "num_spiking", "sparsity_percent", "input_events",
        "input_neurons", "nonzero_weights", "weight_count", "groups",
        "kernel_h", "kernel_w", "dense_macs_per_sample",
        "dense_macs_total_dataset", "theoretical_synops_per_sample",
        "theoretical_synops_total_dataset",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["activations"]:
            writer.writerow({"record_type": "activation", **row})
        for row in report["operations"]:
            writer.writerow({"record_type": "operation", **row})
    return json_path, csv_path


def main(args):
    from datasets import build_dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.disable_eval = False
    dataset, args.nb_classes = build_dataset(is_train=False, args=args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    print(f"CIFAR-10 test samples: {len(dataset)}")

    stage_delays = [float(item.strip()) for item in args.ttfs_stage_delays.split(",")]
    if len(stage_delays) != 4:
        raise ValueError("--ttfs_stage_delays must contain exactly four values")
    model = ConvNeXtSpiking(
        in_chans=3,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
        t_min=args.ttfs_tmin,
        t_max=args.ttfs_tmax,
        force_positive_weights=args.ttfs_force_pos_weights,
        init_delay=args.ttfs_init_delay,
        stage_delays=stage_delays,
    )
    checkpoint_path = resolve_checkpoint(args.load_weights)
    integrity = load_checkpoint_with_integrity(model, checkpoint_path, args.model_key)
    model.to(device).eval()

    accuracy, sample_count, activations, operations = evaluate(
        model, loader, device, args.ttfs_tmax
    )
    dense_total = sum(row["dense_macs_total_dataset"] for row in operations)
    synops_total = sum(row["theoretical_synops_total_dataset"] for row in operations)
    summary = {
        "accuracy_percent": accuracy,
        "sample_count": sample_count,
        "dense_macs_per_sample": dense_total / sample_count,
        "dense_macs_total_dataset": dense_total,
        "theoretical_synops_per_sample": synops_total / sample_count,
        "theoretical_synops_total_dataset": synops_total,
    }
    report = {
        "semantics": {
            "dense_macs": "stem, downsampling, depthwise convolutions, dense pointwise layers if present, and head",
            "theoretical_synops": "non-silent TTFS input events multiplied by their nonzero outgoing pointwise connections",
            "pw2_mode": "spiking: active SpikingBlock.forward calls call_spiking_torch for pw2",
        },
        "checkpoint_integrity": integrity,
        "summary": summary,
        "activations": activations,
        "operations": operations,
    }
    json_path, csv_path = save_outputs(args.output_dir, args.json_name, args.csv_name, report)

    print("\nOperation summary (cost classes are intentionally separate)")
    print(f"Accuracy: {accuracy:.2f}% ({sample_count} samples)")
    print(f"Dense MACs per sample: {summary['dense_macs_per_sample']:,.0f}")
    print(f"Dense MACs total dataset: {dense_total:,}")
    print(f"Theoretical SynOps per sample: {summary['theoretical_synops_per_sample']:,.0f}")
    print(f"Theoretical SynOps total dataset: {synops_total:,}")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
