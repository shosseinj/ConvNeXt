#!/usr/bin/env python3
"""Evaluate spike sparsity of the spiking ConvNeXt model on CIFAR-10.

Sparsity is measured as the percentage of neurons whose spike time equals t_max
(i.e., neurons that never spike / fire after the time window).

Usage (from repo root):
python scripts/evaluate_sparsity.py --load_weights model_ckpt_test/checkpoint-best.pth --data_set CIFAR --eval_data_path ../CIFAR-10-images/test --spiking true --ttfs_tmin 0.0 --ttfs_tmax 1.0

Output:
- Per-batch sparsity (% silent neurons)
- Per-layer average sparsity
- Overall model sparsity
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import get_args_parser
from datasets import build_dataset
from timm.models import create_model
import utils


class SparsityHook:
    """Hook to record spike times and compute sparsity per layer.
    
    Sparsity measures the percentage of neurons that fire LATE or NOT AT ALL
    (spike_time close to or equal to t_max).
    
    IMPORTANT: Due to torch.minimum in SpikingBlock, spike times can be negative
    (indicating very early firing). These should be counted as ACTIVE, not sparse.
    """
    def __init__(self, layer_name, t_max=1.0):
        self.layer_name = layer_name
        self.t_max = t_max
        self.spike_times = []
        self.sparsity = 0.0
        self.num_silent = 0
        self.num_total = 0

    def __call__(self, module, input, output):
        # output is spike_times tensor (batch, channels, height, width) for spiking blocks
        if isinstance(output, torch.Tensor):
            spike_times = output.detach()
            
            # Due to torch.minimum in SpikingBlock, spike times can range from negative to t_max
            # A neuron is "silent" (sparse) if it fires very LATE: spike_time >= t_max - epsilon
            # Negative spike times = EARLY firing = ACTIVE neurons
            threshold = self.t_max - 1e-6
            silent = (spike_times >= threshold).float()
            
            # Calculate sparsity as percentage of silent/late-firing neurons
            self.num_silent = silent.sum().item()
            self.num_total = spike_times.numel()
            
            if self.num_total > 0:
                self.sparsity = self.num_silent / self.num_total
            else:
                self.sparsity = 0.0
            
            self.spike_times.append(spike_times)

import torch
import numpy as np

class SparsityHook:
    def __init__(self, layer_name, t_max=1.0):
        self.layer_name = layer_name
        self.t_max = t_max
        self.sparsity = 0.0
        self.num_silent = 0
        self.num_total = 0

    def __call__(self, module, input, output):
        if not isinstance(output, torch.Tensor):
            return

        spike_times = output.detach()
        threshold = self.t_max - 1e-6

        silent = (spike_times >= threshold)
        self.num_silent = silent.sum().item()
        self.num_total = spike_times.numel()

        self.sparsity = (
            self.num_silent / self.num_total
            if self.num_total > 0 else 0.0
        )


class WeightSparsityHook:
    """Hook to measure weight sparsity for regular Conv2d layers (non-spiking downsampling)."""
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.sparsity = 0.0
        self.num_zeros = 0
        self.num_total = 0
        
    def __call__(self, module, input, output):
        # For Conv2d layers, measure weight sparsity
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            weights = module.weight.data
            self.num_zeros = (weights == 0).sum().item()
            self.num_total = weights.numel()
            self.sparsity = (
                self.num_zeros / self.num_total
                if self.num_total > 0 else 0.0
            )




def evaluate_snn_sparsity(
    model,
    loader,
    device,
    hooks,
    args,
):
    model.eval()

    correct = 0
    total = 0

    # store per-layer sparsity history (for averaging)
    spike_sparsities = {name: [] for name in hooks.keys()}

    # batch-wise global sparsity (optional, for logging)
    batch_sparsities = []

    # TRUE global accumulators (across ALL batches)
    global_silent_all = 0
    global_total_all = 0

    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(loader):
            if args.nb_batches and batch_idx >= args.nb_batches:
                break

            img = img.to(device)
            label = label.to(device)

            # --------------------
            # Forward
            # --------------------
            logits = model(img)
            pred = logits.argmax(dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

            # --------------------
            # Collect per-layer stats
            # --------------------
            batch_total_silent = 0
            batch_total_neurons = 0

            for layer_name, hook in hooks.items():
                # Handle SparsityHook (activation sparsity for SpikingBlocks)
                if hasattr(hook, "num_silent"):
                    spike_sparsities[layer_name].append(hook.sparsity)
                    batch_total_silent += hook.num_silent
                    batch_total_neurons += hook.num_total
                    global_silent_all += hook.num_silent
                    global_total_all += hook.num_total
                # Handle WeightSparsityHook (weight sparsity for downsampling Conv2d)
                elif hasattr(hook, "num_zeros"):
                    spike_sparsities[layer_name].append(hook.sparsity)
                    batch_total_silent += hook.num_zeros
                    batch_total_neurons += hook.num_total
                    global_silent_all += hook.num_zeros
                    global_total_all += hook.num_total

            # --------------------
            # Batch-wise global sparsity (optional)
            # --------------------
            if batch_total_neurons > 0:
                batch_global_sparsity = batch_total_silent / batch_total_neurons
            else:
                batch_global_sparsity = 0.0

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                batch_sparsities.append(batch_global_sparsity)

    # ==========================================================
    # FINAL REPORT
    # ==========================================================
    acc = 100.0 * correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("SNN EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Processed samples: {total}")

    if hooks:
        print("\nPer-Layer Sparsity Analysis:")
        print("=" * 60)
        
        # Separate hooks by type
        spiking_hooks = {k: v for k, v in hooks.items() if hasattr(v, "num_silent")}
        weight_hooks = {k: v for k, v in hooks.items() if hasattr(v, "num_zeros")}
        
        # Print activation sparsity (SpikingBlocks)
        if spiking_hooks:
            print("\nACTIVATION SPARSITY (Spiking Blocks):")
            print("(% of neurons silent during the time window)\n")
            for layer_name in sorted(spiking_hooks.keys()):
                avg_layer_sparsity = np.mean(spike_sparsities[layer_name])
                hook = spiking_hooks[layer_name]
                print(
                    f"{layer_name:<30} "
                    f"{100.0 * avg_layer_sparsity:6.2f}% "
                    f"(silent: {int(hook.num_silent)}/{int(hook.num_total)})"
                )
        
        # Print weight sparsity (Downsampling Conv2d)
        if weight_hooks:
            print("\nWEIGHT SPARSITY (Downsampling Layers):")
            print("(% of zero weights in convolution layers)\n")
            for layer_name in sorted(weight_hooks.keys()):
                avg_layer_sparsity = np.mean(spike_sparsities[layer_name])
                hook = weight_hooks[layer_name]
                print(
                    f"{layer_name:<30} "
                    f"{100.0 * avg_layer_sparsity:6.2f}% "
                    f"(zeros: {int(hook.num_zeros)}/{int(hook.num_total)})"
                )

        # --------------------
        # CORRECT GLOBAL SPARSITY
        # --------------------
        correct_global_sparsity = (
            global_silent_all / global_total_all
            if global_total_all > 0 else 0.0
        )

        print("\n" + "-" * 60)
        print(f"Correct Global SNN Sparsity: {100.0 * correct_global_sparsity:.2f}%")
        print(
            f"(Total silent neurons: {int(global_silent_all)} / "
            f"Total neurons: {int(global_total_all)})"
        )
        print("-" * 60)

    return {
        "accuracy": acc,
        "global_sparsity": correct_global_sparsity,
        "batch_sparsities": batch_sparsities,
        "layer_sparsities": spike_sparsities,
    }




def main():
    parent = get_args_parser()
    parser = argparse.ArgumentParser(parents=[parent])
    args = parser.parse_args()

    # ensure spiking mode is enabled
    args.spiking = True
    args.ttfs_convert = True
    device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

    print(f"Loading dataset: {args.data_set}")
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    args.nb_classes = nb_classes

    loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem
    )

    print(f"Instantiating spiking ConvNeXt model with {nb_classes} classes...")
    from models.convnext import ConvNeXtSpiking
    model = ConvNeXtSpiking(in_chans=3, num_classes=args.nb_classes,
                            drop_path_rate=args.drop_path,
                            layer_scale_init_value=args.layer_scale_init_value,
                            head_init_scale=args.head_init_scale,
                            t_min=args.ttfs_tmin, t_max=args.ttfs_tmax,
                            force_positive_weights=args.ttfs_force_pos_weights)

    # load checkpoint
    if args.load_weights:
        load_path = os.path.normpath(args.load_weights)
        print(f"Loading checkpoint from: {load_path}")
        if os.path.isdir(load_path):
            cand = [os.path.join(load_path, f) for f in os.listdir(load_path)
                    if f.lower().endswith(('.pth', '.pt'))]
            if len(cand) == 0:
                raise FileNotFoundError(f"No checkpoint files (.pth/.pt) found in directory: {load_path}")
            load_path = sorted(cand, key=os.path.getmtime)[-1]
            print(f"Using checkpoint: {load_path}")

        checkpoint = torch.load(load_path, map_location='cpu')
        checkpoint_model = checkpoint.get('model', checkpoint)
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        print("Checkpoint loaded successfully.")

    model.to(device)
    model.eval()

       # -------------------------------------------------
    # Register sparsity hooks
    # -------------------------------------------------
    hooks = {}
    hook_handles = []

    for name, module in model.named_modules():
        # Register SparsityHook for spiking blocks (measures activation sparsity)
        if hasattr(module, "t_max"):
            hook = SparsityHook(layer_name=name, t_max=args.ttfs_tmax)
            handle = module.register_forward_hook(hook)
            hooks[name] = hook
            hook_handles.append(handle)
        # Register WeightSparsityHook for Conv2d layers in downsampling (non-spiking)
        elif isinstance(module, nn.Conv2d) and 'downsample_layers' in name:
            hook = WeightSparsityHook(layer_name=name)
            handle = module.register_forward_hook(hook)
            hooks[name] = hook
            hook_handles.append(handle)

    print(f"Registered {len(hooks)} sparsity hooks (SpikingBlocks + Downsampling Conv2d layers).")

    # -------------------------------------------------
    # Evaluate model sparsity
    # -------------------------------------------------
    results = evaluate_snn_sparsity(
        model=model,
        loader=loader,
        device=device,
        hooks=hooks,
        args=args,
    )

    print("\nFinal Results Dictionary:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {type(v)}")

    # -------------------------------------------------
    # IMPORTANT: remove hooks to avoid memory leaks
    # -------------------------------------------------
    for h in hook_handles:
        h.remove()


if __name__ == '__main__':
    main()
