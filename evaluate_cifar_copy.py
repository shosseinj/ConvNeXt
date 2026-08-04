#!/usr/bin/env python3
"""
Minimal evaluation script for CIFAR-10 Spiking ConvNeXt.
Uses the same dataset builder and evaluation function as main.py.
"""

import argparse
import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from engine import evaluate
from models.convnext import ConvNeXtSpiking
import utils


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt evaluation', add_help=False)

    # Dataset & data loading
    parser.add_argument('--data_path', default='./cifar_data/', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='CIFAR', choices=['CIFAR', 'IMNET', 'image_folder'])
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--num_workers', default=0, type=int)          # avoid shared memory issues
    parser.add_argument('--pin_mem', type=str2bool, default=False)
    parser.add_argument('--crop_pct', default=None, type=float)       # needed by build_transform

    # Augmentation parameters (needed by build_transform)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=str2bool, default=False)

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float)
    parser.add_argument('--head_init_scale', default=1.0, type=float)
    parser.add_argument('--spiking', type=str2bool, default=True)
    parser.add_argument('--ttfs_tmin', type=float, default=0.0)
    parser.add_argument('--ttfs_tmax', type=float, default=1.0)
    parser.add_argument('--ttfs_force_pos_weights', type=str2bool, default=False)
    parser.add_argument('--ttfs_init_delay', type=float, default=0.0)
    parser.add_argument('--ttfs_stage_delays', type=str, default="0.4,0.0,0.00,0.0")

    # Checkpoint loading
    parser.add_argument('--load_weights', default='./best_ckpt/checkpoint_residual_Di_96.09.pth', type=str)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Evaluation only
    parser.add_argument('--eval', type=str2bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # Dummy arguments to keep build_dataset happy
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', default=0, type=int)

    return parser


def main(args):
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build validation dataset (same as in main.py)
    args.disable_eval = False
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    print(f"Validation samples: {len(dataset_val)}")

    # DataLoader
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Create spiking model
    stage_delays = None
    if args.ttfs_stage_delays:
        try:
            stage_delays = [float(d.strip()) for d in args.ttfs_stage_delays.split(',')]
            if len(stage_delays) != 4:
                stage_delays = None
        except Exception:
            stage_delays = None

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
        stage_delays=stage_delays
    )

    # Load checkpoint
    if args.load_weights:
        load_path = os.path.normpath(args.load_weights)
        if os.path.isdir(load_path):
            candidates = [f for f in os.listdir(load_path) if f.lower().endswith(('.pth', '.pt'))]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint files in {load_path}")
            load_path = os.path.join(load_path, sorted(candidates, key=os.path.getmtime)[-1])
        print(f"Loading weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location='cpu')

        # Extract state_dict using model_key
        state_dict = None
        for key in args.model_key.split('|'):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint

        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded.")

    model = model.to(device)
    model.eval()

    # Evaluate using the same evaluate() function from engine.py
    print("Running evaluation...")
    test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    print(f"Accuracy on {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
    return test_stats['acc1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)