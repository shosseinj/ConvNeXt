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
    """Hook to record spike times and compute sparsity per layer."""
    def __init__(self, layer_name, t_max=1.0):
        self.layer_name = layer_name
        self.t_max = t_max
        self.spike_times = []
        self.sparsity = 0.0

    def __call__(self, module, input, output):
        # output is spike_times tensor (batch, ...)
        if isinstance(output, torch.Tensor):
            spike_times = output.detach()
            # count neurons where spike_time == t_max (never spiked)
            silent = (spike_times >= self.t_max - 1e-6).float()
            sparsity = silent.mean().item()
            self.sparsity = sparsity
            self.spike_times.append(spike_times)


def main():
    parent = get_args_parser()
    parser = argparse.ArgumentParser(parents=[parent])
    parser.add_argument('--nb_batches', type=int, default=None, help='Number of batches to evaluate (None=all)')
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
                            t_min=args.ttfs_tmin, t_max=args.ttfs_tmax)

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

    # register hooks to capture spike times at each spiking block
    hooks = {}
    layer_sparsities = defaultdict(list)

    def register_hooks(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if hasattr(child, 'forward'):  # any module with forward
                hook = SparsityHook(full_name, t_max=args.ttfs_tmax)
                child.register_forward_hook(hook)
                hooks[full_name] = hook

    register_hooks(model)

    # evaluate on validation set
    print(f"\nEvaluating sparsity on {len(dataset_val)} samples...")
    print(f"Time window: t_min={args.ttfs_tmin}, t_max={args.ttfs_tmax}")
    print(f"Model: {args.model}, Spiking={args.spiking}, TTFS_convert={args.ttfs_convert}\n")

    correct = 0
    total = 0
    batch_sparsities = []

    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(loader):
            if args.nb_batches and batch_idx >= args.nb_batches:
                break

            img = img.to(device)
            label = label.to(device)

            logits = model(img)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            # collect per-layer sparsities from this batch
            for layer_name, hook in hooks.items():
                layer_sparsities[layer_name].append(hook.sparsity)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_sparsity = np.mean([h.sparsity for h in hooks.values()]) if hooks else 0.0
                batch_sparsities.append(avg_sparsity)
                print(f"Batch [{batch_idx + 1}/{len(loader) if not args.nb_batches else min(args.nb_batches, len(loader))}] "
                      f"Accuracy: {100.0 * correct / total:.2f}% | "
                      f"Avg Layer Sparsity: {100.0 * avg_sparsity:.2f}%")

    # final stats
    final_accuracy = 100.0 * correct / total
    overall_sparsity = np.mean(batch_sparsities) if batch_sparsities else 0.0

    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {final_accuracy:.2f}%")
    print(f"Overall Average Sparsity: {overall_sparsity:.2f}%")
    print(f"Total samples evaluated: {total}")
    print(f"\nPer-Layer Average Sparsities:")
    for layer_name in sorted(hooks.keys()):
        avg_sparsity = np.mean(layer_sparsities[layer_name])
        print(f"  {layer_name}: {100.0 * avg_sparsity:.2f}%")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
