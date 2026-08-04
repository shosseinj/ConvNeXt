#!/usr/bin/env python3
"""
Debugged paper analysis – prints numbers, no JSON error, verifies spike time range.
Run this and paste the output here.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import DataLoader

from datasets import build_dataset
from models.convnext import ConvNeXtSpiking


class DebugSpikeHook:
    """Hook that prints the shape and range of the captured output."""
    def __init__(self, layer_name, t_max):
        self.layer_name = layer_name
        self.t_max = t_max
        self.spike_times = None
        self.sparsity = 0.0
        self.mean_time = 0.0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            spike_times = output[0]
        else:
            spike_times = output

        if not isinstance(spike_times, torch.Tensor):
            return

        # Debug: print first few values
        if self.layer_name == list(self.__dict__.keys())[0]:  # crude: only once
            print(f"Debug {self.layer_name}: output shape {spike_times.shape}, "
                  f"min {spike_times.min().item():.3f}, max {spike_times.max().item():.3f}")

        # Clamp to [0, t_max] to avoid negative times (just for safety)
        spike_times = spike_times.clamp(0, self.t_max)
        self.spike_times = spike_times.detach().cpu()

        silent = (spike_times >= self.t_max - 1e-6)
        self.sparsity = silent.float().mean().item()
        spiked = spike_times[~silent]
        if spiked.numel() > 0:
            self.mean_time = spiked.mean().item()
        else:
            self.mean_time = self.t_max


def collect_delays(model):
    delays = []
    for name, param in model.named_parameters():
        if 'delay' in name.lower():
            delays.extend(param.detach().cpu().numpy().flatten())
    return np.array(delays)


def evaluate(model, dataloader, device, args):
    model.eval()
    hooks = {}
    for name, module in model.named_modules():
        if hasattr(module, 't_max'):
            hook = DebugSpikeHook(name, args.ttfs_tmax)
            module.register_forward_hook(hook)
            hooks[name] = hook

    print(f"Registered {len(hooks)} hooks.")

    correct = 0
    total = 0
    layer_silent = defaultdict(float)
    layer_total = defaultdict(float)
    layer_spike_sum = defaultdict(float)
    layer_spike_cnt = defaultdict(int)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for name, hook in hooks.items():
                st = hook.spike_times
                if st is None:
                    continue
                silent = (st >= args.ttfs_tmax - 1e-6)
                layer_silent[name] += silent.sum().item()
                layer_total[name] += st.numel()
                spiked = st[~silent]
                if spiked.numel() > 0:
                    layer_spike_sum[name] += spiked.sum().item()
                    layer_spike_cnt[name] += spiked.numel()

            if (batch_idx+1) % 20 == 0:
                print(f"Batch {batch_idx+1} | Acc = {100.0*correct/total:.2f}%")

    # Compute per layer
    layer_sparsity = {}
    layer_mean_time = {}
    for name in hooks:
        if layer_total[name] > 0:
            sp = 100.0 * layer_silent[name] / layer_total[name]
            layer_sparsity[name] = sp
        if layer_spike_cnt[name] > 0:
            mt = layer_spike_sum[name] / layer_spike_cnt[name]
            layer_mean_time[name] = mt
        else:
            layer_mean_time[name] = args.ttfs_tmax

    global_sparsity = 100.0 * sum(layer_silent.values()) / sum(layer_total.values())
    active_ratio = 1.0 - global_sparsity / 100.0
    delays = collect_delays(model)

    return {
        "accuracy": 100.0 * correct / total,
        "global_sparsity": global_sparsity,
        "active_ratio": active_ratio,
        "layer_sparsity": layer_sparsity,
        "layer_mean_time": layer_mean_time,
        "delays": delays,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./cifar_data/')
    parser.add_argument('--load_weights', default='./best_ckpt/checkpoint_residual_Di_96.09.pth')
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ttfs_tmax', type=float, default=1.0)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--output_dir', default='./paper_results')
    # Dummy args
    parser.add_argument('--nb_classes', type=int, default=10)
    parser.add_argument('--imagenet_default_mean_and_std', type=bool, default=True)
    parser.add_argument('--data_set', default='CIFAR')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_pct', default=None)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=bool, default=False)
    parser.add_argument('--dist_eval', type=bool, default=True)
    parser.add_argument('--disable_eval', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    dataset_val, _ = build_dataset(is_train=False, args=args)
    loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
    print(f"Test samples: {len(dataset_val)}")

    # Model
    stage_delays = [0.4, 0.0, 0.0, 0.0]
    model = ConvNeXtSpiking(in_chans=3, num_classes=args.nb_classes,
                            drop_path_rate=0.0, layer_scale_init_value=1e-6,
                            head_init_scale=1.0, t_min=0.0, t_max=args.ttfs_tmax,
                            force_positive_weights=False, init_delay=0.0,
                            stage_delays=stage_delays)
    if os.path.exists(args.load_weights):
        ckpt = torch.load(args.load_weights, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        print("Checkpoint loaded.")
    model = model.to(device).eval()

    results = evaluate(model, loader, device, args)

    # Print final results (human readable)
    print("\n" + "="*60)
    print("FINAL NUMBERS FOR PAPER")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Global sparsity: {results['global_sparsity']:.2f}%")
    print(f"Active ratio (ρ): {results['active_ratio']:.3f}")
    print(f"FLOPs reduction: {(1 - results['active_ratio'])*100:.1f}%")
    print(f"Number of delay parameters: {len(results['delays'])}")
    if len(results['delays']) > 0:
        print(f"Delay mean: {np.mean(results['delays']):.4f} ± {np.std(results['delays']):.4f}")
    else:
        print("No delay parameters found (they were not trained).")

    print("\nPer‑layer sparsity (first 10 layers):")
    for i, (name, sp) in enumerate(list(results['layer_sparsity'].items())[:10]):
        print(f"  {name:<40} {sp:6.2f}%")

    print("\nPer‑layer mean spike time (first 10 layers):")
    for i, (name, mt) in enumerate(list(results['layer_mean_time'].items())[:10]):
        print(f"  {name:<40} {mt:6.3f}")

    # Also save a simple JSON without numpy errors
    import json
    json_out = {
        "accuracy": results['accuracy'],
        "global_sparsity": results['global_sparsity'],
        "active_ratio": results['active_ratio'],
        "flops_reduction_percent": (1 - results['active_ratio'])*100,
        "num_delays": len(results['delays']),
        "delay_mean": float(np.mean(results['delays'])) if len(results['delays'])>0 else None,
        "delay_std": float(np.std(results['delays'])) if len(results['delays'])>0 else None,
        "layer_sparsity": {k: float(v) for k, v in list(results['layer_sparsity'].items())[:20]},
        "layer_mean_time": {k: float(v) for k, v in list(results['layer_mean_time'].items())[:20]},
    }
    with open(os.path.join(args.output_dir, 'paper_numbers.json'), 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved simplified JSON to {os.path.join(args.output_dir, 'paper_numbers.json')}")

    # Critical check: if any mean spike time is negative, warn user
    if any(mt < 0 for mt in results['layer_mean_time'].values()):
        print("\n⚠️  WARNING: Negative mean spike times detected! The hook is not capturing the correct spike time tensor.")
        print("   You need to inspect the forward method of your spiking blocks.")
        print("   The output tensor should contain spike times in [0, t_max].")
        print("   In the meantime, you can still use the global sparsity (which is correct).")

if __name__ == '__main__':
    main()