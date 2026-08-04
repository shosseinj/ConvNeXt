#!/usr/bin/env python3
"""
Evaluate accuracy + sparsity/efficiency for CIFAR-10 Spiking ConvNeXt.
Sparsity = fraction of neurons that never spike (output == t_max).
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from collections import defaultdict

from datasets import build_dataset
from engine import evaluate          # for accuracy only (optional)
from models.convnext import ConvNeXtSpiking
import utils


# ---------- sparsity hooks (same as in main.py) ----------
class SparsityHook:
    """Measure activation sparsity: fraction of outputs == t_max (no spike)."""
    def __init__(self, layer_name, t_max=1.0):
        self.layer_name = layer_name
        self.t_max = t_max
        self.sparsity = 0.0
        self.num_silent = 0
        self.num_total = 0

    def __call__(self, module, input, output):
        if not isinstance(output, torch.Tensor):
            return
        # output are spike times (t_min..t_max)
        silent = (output >= self.t_max - 1e-6)   # neurons that never spiked
        self.num_silent = silent.sum().item()
        self.num_total = output.numel()
        self.sparsity = self.num_silent / self.num_total if self.num_total > 0 else 0.0


class WeightSparsityHook:
    """Measure weight sparsity for downsampling Conv2d layers."""
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.sparsity = 0.0
        self.num_zero = 0
        self.num_total = 0

    def __call__(self, module, input, output):
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            w = module.weight.data
            self.num_zero = (w == 0).sum().item()
            self.num_total = w.numel()
            self.sparsity = self.num_zero / self.num_total if self.num_total > 0 else 0.0


def register_sparsity_hooks(model, t_max):
    """Register hooks for all spiking blocks and downsampling convs."""
    hooks = {}
    for name, module in model.named_modules():
        if hasattr(module, "t_max"):          # spiking block
            hook = SparsityHook(name, t_max)
            module.register_forward_hook(hook)
            hooks[name] = hook
        elif isinstance(module, nn.Conv2d) and 'downsample_layers' in name:
            hook = WeightSparsityHook(name)
            module.register_forward_hook(hook)
            hooks[name] = hook
    print(f"Registered {len(hooks)} sparsity hooks")
    return hooks


def evaluate_sparsity(model, dataloader, device, args):
    """Run evaluation and collect sparsity stats per layer."""
    model.eval()
    hooks = register_sparsity_hooks(model, args.ttfs_tmax)

    # accumulators
    correct = 0
    total = 0
    layer_sparsities = defaultdict(list)   # list of sparsity per batch for each layer
    global_silent = 0
    global_neurons = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # aggregate sparsity from hooks
            batch_silent = 0
            batch_total = 0
            for name, hook in hooks.items():
                if hasattr(hook, "sparsity"):
                    layer_sparsities[name].append(hook.sparsity)
                    if hasattr(hook, "num_silent"):      # activation sparsity
                        batch_silent += hook.num_silent
                        batch_total += hook.num_total
                        global_silent += hook.num_silent
                        global_neurons += hook.num_total
                    elif hasattr(hook, "num_zero"):      # weight sparsity
                        batch_silent += hook.num_zero
                        batch_total += hook.num_total
                        global_silent += hook.num_zero
                        global_neurons += hook.num_total

            # optional: print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                acc_sofar = 100.0 * correct / total
                print(f"Batch {batch_idx+1}/{len(dataloader)} | Acc={acc_sofar:.2f}%")

    # final accuracy
    accuracy = 100.0 * correct / total
    global_sparsity = (global_silent / global_neurons * 100) if global_neurons > 0 else 0.0

    # per-layer average sparsity
    avg_layer_sparsity = {}
    for name, sp_list in layer_sparsities.items():
        avg_layer_sparsity[name] = np.mean(sp_list) * 100   # percent

    return accuracy, global_sparsity, avg_layer_sparsity


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt evaluation (accuracy + sparsity)', add_help=False)

    # Dataset & data loading
    parser.add_argument('--data_path', default='./cifar_data/', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=10, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='CIFAR', choices=['CIFAR', 'IMNET', 'image_folder'])
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=False)
    parser.add_argument('--crop_pct', default=None, type=float)

    # Augmentation (still needed for dataset builder)
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
    parser.add_argument('--ttfs_stage_delays', type=str, default="0.3,0.1,0.05,0.0") 

    # Checkpoint
    parser.add_argument('--load_weights', default='./hossein_output/checkpoint-best.pth', type=str)
    # parser.add_argument('--load_weights', default='./best_ckpt/checkpoint_residual_Di_96.09.pth', type=str)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Evaluation
    parser.add_argument('--eval', type=str2bool, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # Dummy arguments
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--output_dir', default='./hossein_output')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', default=0, type=int)

    return parser


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build validation dataset
    args.disable_eval = False
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    print(f"Validation samples: {len(dataset_val)}")

    loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Create model
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
            candidates = [f for f in os.listdir(load_path) if f.endswith(('.pth', '.pt'))]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint in {load_path}")
            load_path = os.path.join(load_path, sorted(candidates, key=os.path.getmtime)[-1])
        print(f"Loading weights: {load_path}")
        ckpt = torch.load(load_path, map_location='cpu',  weights_only=False)
        state_dict = None
        for key in args.model_key.split('|'):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt
        # remove 'module.' prefix
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
        print("Checkpoint loaded.")

    model = model.to(device)



    
    # ================================================================
    # TRAINING SETUP
    # ================================================================
    from engine import train_one_epoch
    from utils import NativeScalerWithGradNormCount as NativeScaler
    from optim_factory import create_optimizer
    import time
    import datetime
    
    # Build training dataset
    dataset_train, _ = build_dataset(is_train=True, args=args)
    print(f"Training samples: {len(dataset_train)}")
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    # Training hyperparameters
    TRAIN_EPOCHS = 100
    TRAIN_LR = 4e-3
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 20
    WEIGHT_DECAY = 0.05
    LAMBDA_SPIKE = 0.0       # Set >0 for sparsity regularization
    LAMBDA_DELAY = 0.1       # Delay regularization
    CLIP_GRAD = None
    UPDATE_FREQ = 1
    SAVE_CKPT_FREQ = 10
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    for pg in optimizer.param_groups:
        pg['lr_scale'] = 1.0
    
    loss_scaler = NativeScaler()
    criterion = nn.CrossEntropyLoss()
    
    # Schedulers
    steps_per_epoch = len(train_loader) // UPDATE_FREQ
    lr_schedule = utils.cosine_scheduler(
        TRAIN_LR, MIN_LR, TRAIN_EPOCHS, steps_per_epoch,
        warmup_epochs=WARMUP_EPOCHS)
    wd_schedule = utils.cosine_scheduler(
        WEIGHT_DECAY, WEIGHT_DECAY, TRAIN_EPOCHS, steps_per_epoch)
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else './training_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Tracking
    best_acc = 0.0
    best_sparsity = 0.0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {TRAIN_EPOCHS}")
    print(f"Learning rate: {TRAIN_LR}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"lambda_spike: {LAMBDA_SPIKE}")
    print(f"lambda_delay: {LAMBDA_DELAY}")
    print(f"{'='*60}\n")
    
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    for epoch in range(TRAIN_EPOCHS):
        # Train one epoch
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, CLIP_GRAD, None, None,
            log_writer=None, wandb_logger=None,
            start_steps=epoch * steps_per_epoch,
            lr_schedule_values=lr_schedule,
            wd_schedule_values=wd_schedule,
            num_training_steps_per_epoch=steps_per_epoch,
            update_freq=UPDATE_FREQ,
            use_amp=False,
            lambda_delay=LAMBDA_DELAY,
            lambda_spike=LAMBDA_SPIKE)
        
        # Evaluate
        acc, sparsity, layer_sparsities = evaluate_sparsity(model, loader, device, args)
        
        # Print progress
        train_loss = train_stats.get('loss', 0)
        print(f"\nEpoch {epoch+1}/{TRAIN_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {acc:.2f}%")
        print(f"  Val Sparsity: {sparsity:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_sparsity = sparsity
            
            save_path = os.path.join(output_dir, 'checkpoint-best.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': acc,
                'sparsity': sparsity,
            }, save_path)
            print(f"  ★ New best model saved! (Acc: {acc:.2f}%, Sparsity: {sparsity:.2f}%)")
        
        # Periodic checkpoint
        if (epoch + 1) % SAVE_CKPT_FREQ == 0:
            save_path = os.path.join(output_dir, f'checkpoint-epoch{epoch+1}.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': acc,
                'sparsity': sparsity,
            }, save_path)
            print(f"  Checkpoint saved: checkpoint-epoch{epoch+1}.pth")
    
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE ({total_time})")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best Sparsity: {best_sparsity:.2f}%")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")



    # model.eval()

    # # Evaluate accuracy + sparsity
    # print("\n==== Running sparsity-aware evaluation ====")
    # acc, global_sparsity, layer_sparsities = evaluate_sparsity(model, loader, device, args)

    # # Report results
    # print("\n" + "="*60)
    # print(f"Accuracy: {acc:.2f}%")
    # print(f"Global sparsity (silent neurons): {global_sparsity:.2f}%")
    # print(f"  -> Efficiency: {100 - global_sparsity:.2f}% of neurons spike at least once")
    # print("="*60)

    # # Per-layer sparsity (top 10 most silent layers)
    # sorted_layers = sorted(layer_sparsities.items(), key=lambda x: -x[1])
    # print("\nPer-layer sparsity (activation + weight):")
    # for name, sp in sorted_layers[:10]:
    #     # mark if it's a spiking layer (ttfs) or weight sparsity
    #     layer_type = "Spiking" if "spiking_blocks" in name or any(x in name.lower() for x in ["conv", "stage"]) else "Weight"
    #     print(f"  {name:<45} {sp:6.2f}%   ({layer_type})")
    # if len(sorted_layers) > 10:
    #     print(f"  ... and {len(sorted_layers)-10} more layers")

    # # Save results to file
    # if args.output_dir:
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     out_file = os.path.join(args.output_dir, "sparsity_results.txt")
    #     with open(out_file, 'w') as f:
    #         f.write(f"Checkpoint: {args.load_weights}\n")
    #         f.write(f"Accuracy: {acc:.2f}%\n")
    #         f.write(f"Global sparsity: {global_sparsity:.2f}%\n")
    #         f.write("Per-layer sparsity:\n")
    #         for name, sp in sorted_layers:
    #             f.write(f"  {name}: {sp:.2f}%\n")
    #     print(f"\nResults saved to {out_file}")

    # return acc, global_sparsity


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt evaluation with sparsity', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)