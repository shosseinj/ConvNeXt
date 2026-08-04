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
from models.convnext import ConvNeXtSpiking, register_sparsity_hooks , calculate_model_sparsity,  compute_energy_correctly
import utils


# # ---------- sparsity hooks (same as in main.py) ----------
# class SparsityHook:
#     """Measure activation sparsity: fraction of outputs == t_max (no spike)."""
#     def __init__(self, layer_name, t_max=1.0):
#         self.layer_name = layer_name
#         self.t_max = t_max
#         self.sparsity = 0.0
#         self.num_silent = 0
#         self.num_total = 0

#     def __call__(self, module, input, output):
#         if not isinstance(output, torch.Tensor):
#             return
#         # output are spike times (t_min..t_max)
#         silent = (output >= self.t_max - 1e-6)   # neurons that never spiked
#         self.num_silent = silent.sum().item()
#         self.num_total = output.numel()
#         self.sparsity = self.num_silent / self.num_total if self.num_total > 0 else 0.0




# def register_sparsity_hooks(model, t_max):
#     """Register hooks for all spiking blocks and downsampling convs."""
#     hooks = {}
#     for name, module in model.named_modules():
#         if not name or "downsample" in name or "head" in name:
#              continue
#         if hasattr(module, "t_max"):          # spiking block
#             hook = SparsityHook(name, t_max)
#             module.register_forward_hook(hook)
#             hooks[name] = hook
   
#     print(f"Registered {len(hooks)} sparsity hooks")
#     return hooks







def print_sparsity_both_scenarios(hooks):

    """Print BOTH scenarios: final output sparsity AND per-spiking-op sparsity."""
    
    print("\n" + "="*100)
    print("SCENARIO 1: BLOCK OUTPUT SPARSITY (Final output after torch.minimum)")
    print("="*100)
    print(f"{'Layer':<25} {'Sparsity %':>12}")
    print("-"*40)
    
    final_sparsities = []
    for name, hook in hooks.items():
        sp = hook.sparsity_final * 100
        final_sparsities.append(sp)
        print(f"{name:<25} {sp:11.2f}%")
    
    avg_final = sum(final_sparsities) / len(final_sparsities)
    print("-"*40)
    print(f"{'AVERAGE':<25} {avg_final:11.2f}%")
    print(f"\n  Total layers: {len(final_sparsities)}")
    print(f"  Min: {min(final_sparsities):.2f}% | Max: {max(final_sparsities):.2f}%")
    
    print("\n" + "="*100)
    print("SCENARIO 2: PER-SPIKING-OP SPARSITY (All call_spiking_torch outputs)")
    print("="*100)
    print(f"{'Layer':<25} {'After pw1':>12} {'After pw2':>12}")
    print("-"*55)
    
    all_mid = []
    all_out = []
    
    for name, hook in hooks.items():
        mid_sp = hook.sparsity_mid * 100
        out_sp = hook.sparsity_out * 100
        
        print(f"{name:<25} {mid_sp:11.2f}% {out_sp:11.2f}%")
        
        all_mid.append(mid_sp)
        all_out.append(out_sp)
    
    # Averages
    avg_mid = sum(all_mid) / len(all_mid)
    avg_out = sum(all_out) / len(all_out)
    print("-"*55)
    print(f"{'AVERAGE':<25} {avg_mid:11.2f}% {avg_out:11.2f}%")
    
    # Combined (all spiking operations)
    all_spiking_ops = all_mid + all_out
    overall_avg = sum(all_spiking_ops) / len(all_spiking_ops)
    
    print(f"\n  Combined Statistics (36 spiking operations):")
    print(f"  Average per-spiking-op: {overall_avg:.2f}%")
    print(f"  After pw1 average: {avg_mid:.2f}%")
    print(f"  After pw2 average: {avg_out:.2f}%")
    print(f"  Min: {min(all_spiking_ops):.2f}% | Max: {max(all_spiking_ops):.2f}%")
    
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"  Scenario 1 (Block outputs):     {avg_final:.2f}% average sparsity")
    print(f"  Scenario 2 (Per-spiking-op):    {overall_avg:.2f}% average sparsity")
    print(f"  VGG16-SNN (literature):         ~30% average sparsity")
    print("="*100 + "\n")




def evaluate_sparsity(model, dataloader, device, args):
    """Run evaluation and collect sparsity stats per layer."""
    model.eval()
    # hooks = register_sparsity_hooks(model, args.ttfs_tmax)

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
            # for name, hook in hooks.items():
            #     if hasattr(hook, "sparsity"):
            #         layer_sparsities[name].append(hook.sparsity)
            #         if hasattr(hook, "num_silent"):      # activation sparsity
            #             batch_silent += hook.num_silent
            #             batch_total += hook.num_total
            #             global_silent += hook.num_silent
            #             global_neurons += hook.num_total
            #         elif hasattr(hook, "num_zero"):      # weight sparsity
            #             batch_silent += hook.num_zero
            #             batch_total += hook.num_total
            #             global_silent += hook.num_zero
            #             global_neurons += hook.num_total

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
    results = calculate_model_sparsity(model)

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
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
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
    parser.add_argument('--ttfs_stage_delays', type=str, default="0.4,0.0,0.00,0.0")

    # Checkpoint
    parser.add_argument('--load_weights', default='./ckpt_residual_Di_96.09/checkpoint_residual_Di_96.09.pth', type=str)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Evaluation & Fine-tuning
    parser.add_argument('--eval', type=str2bool, default=True)
    parser.add_argument('--do_finetune', type=str2bool, default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # ===== FINE-TUNING HYPERPARAMETERS =====
    parser.add_argument('--epochs', default=100, type=int, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--lambda_spike', default=0.01, type=float, help='Sparsity loss weight')
    parser.add_argument('--lambda_delay', default=0.5, type=float, help='Delay regularization weight')
    parser.add_argument('--clip_grad', default=None, type=float, help='Gradient clipping value')
    parser.add_argument('--update_freq', default=1, type=int, help='Update frequency')
    parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Checkpoint save frequency')
    parser.add_argument('--weight_decay_end', default=0.05, type=float, help='Final weight decay')
    # =====================================

    # Dummy arguments (for compatibility with other scripts)
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--finetune', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--weight_decay', default=0.05, type=float)  # Changed from None to 0.05
    parser.add_argument('--opt', default='adamw', type=str)  # Changed from None to 'adamw'
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--seed', default=0, type=int)

    return parser


def compute_sparsity_loss(model, target_sparsity=0.3):
    """
    Compute a differentiable sparsity loss that encourages neurons to be silent.
    This directly penalizes low sparsity in spiking layers.
    """
    sparsity_loss = 0.0
    num_layers = 0
    
    for name, module in model.named_modules():
        # Check for spiking layers (have t_max attribute)
        if hasattr(module, 't_max') and hasattr(module, 'latest_spike'):
            if module.latest_spike is not None:
                # Calculate activation sparsity for this layer
                spikes = module.latest_spike
                # Neurons that never spike (output close to t_max)
                silent = (spikes >= module.t_max - 1e-6).float()
                current_sparsity = silent.mean()
                
                # Penalize if sparsity is below target
                loss = torch.relu(target_sparsity - current_sparsity)
                sparsity_loss += loss
                num_layers += 1
        
        # Also handle downsample layers if they have spiking mechanism
        elif hasattr(module, 'D') and hasattr(module, 't_max'):  # SpikingDownsample
            # For downsample layers, encourage larger delays = more sparsity
            delay_penalty = torch.relu(0.1 - torch.sigmoid(module.D).mean())
            sparsity_loss += delay_penalty * 0.1
            num_layers += 1
    
    if num_layers > 0:
        sparsity_loss = sparsity_loss / num_layers
    
    return sparsity_loss


def compute_delay_regularization(model, max_delay=0.5):
    """
    Regularize delays to encourage sparsity.
    Larger delays → later spikes → more chance of being silent (capped at t_max)
    """
    delay_loss = 0.0
    num_delays = 0
    
    for name, module in model.named_modules():
        # SpikingBlock delays
        if hasattr(module, 'D_mid') and hasattr(module, 'D_out'):
            # Encourage delays to be larger (but not exceed max)
            d_mid_loss = torch.relu(max_delay - torch.sigmoid(module.D_mid).mean())
            d_out_loss = torch.relu(max_delay - torch.sigmoid(module.D_out).mean())
            delay_loss += d_mid_loss + d_out_loss
            num_delays += 2
        
        # SpikingDownsample delays
        elif hasattr(module, 'D'):
            d_loss = torch.relu(max_delay - torch.sigmoid(module.D).mean())
            delay_loss += d_loss
            num_delays += 1
    
    if num_delays > 0:
        delay_loss = delay_loss / num_delays
    
    return delay_loss

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
   # Create model WITH spiking downsample layers
    stage_delays = None
    if args.ttfs_stage_delays:
        try:
            stage_delays = [float(d.strip()) for d in args.ttfs_stage_delays.split(',')]
            if len(stage_delays) != 4:
                stage_delays = [0.3, 0.15, 0.05, 0.02]  # Default: higher delays for early layers
        except Exception:
            stage_delays = [0.3, 0.15, 0.05, 0.02]

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
        stage_delays=stage_delays  # This enables spiking in downsample layers!
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
        ckpt = torch.load(load_path, map_location='cpu')
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
    

    
    from torchinfo import summary
    summary(model, input_size=(1, 3, 32, 32), device="cpu" if not torch.cuda.is_available() else "cuda")



    # After evaluate_sparsity() call:

# ================================================================
# USAGE: Add this to your main() after evaluate_sparsity()
# ================================================================

    # After your evaluation:
    print("\n" + "="*70)
    print("ENERGY ANALYSIS: SNN vs ANN")
    print("="*70)

    # Compute SNN energy
    energy_results = compute_energy_correctly(model, loader, device, args.ttfs_tmax)

    # ANN values (from your verified numbers)
    ANN_MACS = 0.090  # G
    ANN_ENERGY_MJ = ANN_MACS * 1e9 * 4.6e-12 * 1000  # 0.414 mJ

    # Print SNN breakdown
    print(f"\nSNN Energy Breakdown:")
    print(f"  Non-spiking MACs:  {energy_results['total_macs']/1e6:.2f} M")
    print(f"  Spiking SynOps:    {energy_results['total_synops']/1e9:.3f} G")
    print(f"  MACs energy:       {energy_results['energy_macs_mj']:.4f} mJ")
    print(f"  SynOps energy:     {energy_results['energy_synops_mj']:.4f} mJ")
    print(f"  TOTAL SNN energy:  {energy_results['total_energy_mj']:.4f} mJ")

    print(f"\nANN Energy:          {ANN_ENERGY_MJ:.4f} mJ")

    if energy_results['total_energy_mj'] > 0:
        ratio = ANN_ENERGY_MJ / energy_results['total_energy_mj']
        if ratio >= 1:
            print(f"Energy Ratio:        {ratio:.1f}× less than ANN ✓")
        else:
            print(f"Energy Ratio:        {ratio:.3f}× (SNN uses {1/ratio:.1f}× more energy)")

    # Per-layer breakdown (top contributors)
    print(f"\n{'='*70}")
    print(f"Per-Layer Energy (top contributors):")
    print(f"{'='*70}")
    print(f"{'Layer':<25} {'dwconv MACs':>12} {'pw1 SynOps':>12} {'pw2 SynOps':>12}")
    print(f"{'-'*65}")

    for detail in energy_results['layer_details']:
        print(f"{detail['name']:<25} {detail['dwconv_macs']/1e6:>9.2f}M {detail['synops_pw1']/1e6:>9.2f}M {detail['synops_pw2']/1e6:>9.2f}M")

    # Paper-ready table row
    print(f"\n{'='*70}")
    print(f"PAPER-READY TABLE ROW:")
    print(f"{'='*70}")
    print(f"| ConvNeXt-T | ANN  | -     | {ANN_MACS:.3f} | -        | {ANN_ENERGY_MJ:.4f} | 1.0× |")
    print(f"| ConvNeXt-T | SNN  | TTFS  | 0.090 | {energy_results['total_synops']/1e9:.3f} | {energy_results['total_energy_mj']:.4f} | {ratio:.1f}× |")
    print(f"{'='*70}")


    # Evaluate accuracy + sparsity
    if not args.do_finetune:
        model.eval()
        print("\n==== Running sparsity-aware evaluation ====")
        acc, global_sparsity, layer_sparsities = evaluate_sparsity(model, loader, device, args)

        # Report results
        print("\n" + "="*60)
        print(f"Accuracy: {acc:.2f}%")
        print(f"Global sparsity (silent neurons): {global_sparsity:.2f}%")
        print(f"  -> Efficiency: {100 - global_sparsity:.2f}% of neurons spike at least once")
        print("="*60)

        # Per-layer sparsity (top 10 most silent layers)
        print("\n" + "="*80)
        print("SPARSITY ANALYSIS - Focus on increasing sparsity in low-sparsity layers")
        print("="*80)





        print(f"\n📊 MODEL SUMMARY - ALL LAYER SPARSITIES")
        print(f"{'='*80}")
        print(f"{'Layer Name':<45} {'Sparsity %':>12} {'Type':>15}")
        print(f"{'-'*80}")

        for name, sp in layer_sparsities.items():
            layer_type = "Spiking" if "spiking_blocks" in name or any(x in name.lower() for x in ["conv", "stage"]) else "Weight"
            print(f"{name:<45} {sp:12.2f}% {layer_type:>15}")

        print(f"{'-'*80}")
        print(f"\n📈 Overall Statistics:")
        print(f"  Total layers: {len(layer_sparsities)}")
        print(f"  Average sparsity: {sum(layer_sparsities.values())/len(layer_sparsities):.2f}%")
        print(f"  Min sparsity: {min(layer_sparsities.values()):.2f}%")
        print(f"  Max sparsity: {max(layer_sparsities.values()):.2f}%")
        print(f"{'-'*80}")

       


        # Save results to file
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, "sparsity_results.txt")
            with open(out_file, 'w') as f:
                f.write(f"Checkpoint: {args.load_weights}\n")
                f.write(f"Accuracy: {acc:.2f}%\n")
                f.write(f"Global sparsity: {global_sparsity:.2f}%\n")
                f.write("Per-layer sparsity:\n")
                # for name, sp in sorted_layers:
                #     f.write(f"  {name}: {sp:.2f}%\n")
            print(f"\nResults saved to {out_file}")

    # return acc, global_sparsity
    if not args.do_finetune:
        return

    # ================================================================
    # FINE-TUNING FOR SPARSITY (everything below is NEW)
    # ================================================================
    from engine import train_one_epoch
    from utils import NativeScalerWithGradNormCount as NativeScaler
    from optim_factory import create_optimizer
    import time
    import datetime
    # Build training dataset (needed for fine-tuning)
    dataset_train, _ = build_dataset(is_train=True, args=args)
    print(f"\nTraining samples: {len(dataset_train)}")
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    # Set up training parameters (using defaults if not specified)
    finetune_epochs = getattr(args, 'epochs', 100)
    finetune_lr = getattr(args, 'lr', 1e-4)
    finetune_min_lr = getattr(args, 'min_lr', 1e-6)
    finetune_warmup = getattr(args, 'warmup_epochs', 5)
    finetune_wd = getattr(args, 'weight_decay', 0.05)
    finetune_wd_end = getattr(args, 'weight_decay_end', 0.05)
    finetune_lambda_spike = getattr(args, 'lambda_spike', 0.005)
    finetune_lambda_delay = getattr(args, 'lambda_delay', 0.2)
    finetune_clip_grad = getattr(args, 'clip_grad', None)
    finetune_update_freq = getattr(args, 'update_freq', 1)
    finetune_save_freq = getattr(args, 'save_ckpt_freq', 10)
        # Ensure optimizer arguments exist with proper defaults
    if not hasattr(args, 'opt') or args.opt is None:
        args.opt = 'adamw'
    if not hasattr(args, 'opt_eps') or args.opt_eps is None:
        args.opt_eps = 1e-8
    if not hasattr(args, 'opt_betas') or args.opt_betas is None:
        args.opt_betas = [0.9, 0.999]
    if not hasattr(args, 'momentum') or args.momentum is None:
        args.momentum = 0.9
    if not hasattr(args, 'layer_decay') or args.layer_decay is None:
        args.layer_decay = 1.0
    if not hasattr(args, 'weight_decay') or args.weight_decay is None:
        args.weight_decay = 0.05
        # Fix None values
    if finetune_lr is None:
        finetune_lr = 1e-4
    if finetune_wd is None:
        finetune_wd = 0.05
    if finetune_min_lr is None:
        finetune_min_lr = 1e-6
    if finetune_warmup is None:
        finetune_warmup = 5
    if finetune_wd_end is None:
        finetune_wd_end = 0.05
    if finetune_lambda_spike is None:
        finetune_lambda_spike = 0.005
    if finetune_lambda_delay is None:
        finetune_lambda_delay = 0.2
    if finetune_update_freq is None:
        finetune_update_freq = 1
    if finetune_save_freq is None:
        finetune_save_freq = 10
    if finetune_epochs is None:
        finetune_epochs = 100
    # Verify
    print(f"Optimizer settings: opt={args.opt}, lr={finetune_lr}, wd={args.weight_decay}")
    print(f"\n{'='*60}")
    print(f"STARTING SPARSITY FINE-TUNING")
    print(f"{'='*60}")
    print(f"Epochs: {finetune_epochs}")
    print(f"Learning rate: {finetune_lr}")
    print(f"lambda_spike: {finetune_lambda_spike}")
    print(f"lambda_delay: {finetune_lambda_delay}")
    print(f"{'='*60}\n")
    
    # Create optimizer
        # Create optimizer with lr_scale for compatibility
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_lr,
        weight_decay=finetune_wd,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    # Add lr_scale to each param group (required by train_one_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr_scale'] = 1.0

        
    loss_scaler = NativeScaler()
    criterion = nn.CrossEntropyLoss()
    
    # LR and WD schedulers
    steps_per_epoch = len(train_loader) // finetune_update_freq
    lr_schedule = utils.cosine_scheduler(
        finetune_lr, finetune_min_lr, finetune_epochs, steps_per_epoch,
        warmup_epochs=finetune_warmup)
    wd_schedule = utils.cosine_scheduler(
        finetune_wd, finetune_wd_end, finetune_epochs, steps_per_epoch)
    
    # Create output directory
    finetune_output = getattr(args, 'output_dir', './sparsity_tuned')
    if not finetune_output:
        finetune_output = './sparsity_tuned'
    os.makedirs(finetune_output, exist_ok=True)
    
    # Tracking
    best_score = 0.0
    best_acc = 0.0
    best_sparsity = 0.0
    start_time = time.time()
    
        
        

        # Training loop with proper sparsity regularization
    model.train()
    # Training loop with proper sparsity regularization
    print(f"\nStarting training for {finetune_epochs} epochs")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Loss scaler: {loss_scaler}")
    print(f"Lambda spike: {finetune_lambda_spike}, Lambda delay: {finetune_lambda_delay}\n")

    for epoch in range(finetune_epochs):
        print(f'\n{"="*50}')
        print(f'EPOCH {epoch + 1}/{finetune_epochs}')
        print(f'{"="*50}')
        
        model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_delay_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Debug: print first few batches
            if batch_idx < 3:
                print(f"  Batch {batch_idx}: images shape={images.shape}, labels shape={labels.shape}")
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(images)
                ce_loss = criterion(outputs, labels)
                
                # Compute regularization losses
                sparsity_loss = compute_sparsity_loss(model, target_sparsity=0.3)
                delay_loss = compute_delay_regularization(model, max_delay=0.5)
                
                # Ensure losses are on correct device and have gradients
                if isinstance(sparsity_loss, torch.Tensor):
                    sparsity_loss = sparsity_loss.to(device)
                else:
                    sparsity_loss = torch.tensor(0.0, device=device)
                    
                if isinstance(delay_loss, torch.Tensor):
                    delay_loss = delay_loss.to(device)
                else:
                    delay_loss = torch.tensor(0.0, device=device)
                
                # Combined loss
                total_loss = ce_loss + finetune_lambda_spike * sparsity_loss + finetune_lambda_delay * delay_loss
            
            # Check for valid loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  WARNING: Invalid loss detected! CE={ce_loss.item()}, "
                    f"Sparsity={sparsity_loss.item()}, Delay={delay_loss.item()}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            
            # Use loss_scaler or direct backward
            if loss_scaler is not None:
                loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                        clip_grad=finetune_clip_grad)
            else:
                total_loss.backward()
                if finetune_clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), finetune_clip_grad)
                optimizer.step()
            
            # Accumulate losses
            epoch_ce_loss += ce_loss.item()
            epoch_sparsity_loss += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
            epoch_delay_loss += delay_loss.item() if isinstance(delay_loss, torch.Tensor) else delay_loss
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Print progress every 10 batches (more frequent for debugging)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                    f"CE={ce_loss.item():.4f}, "
                    f"Sparsity={sparsity_loss.item():.4f}, "
                    f"Delay={delay_loss.item():.4f}, "
                    f"Total={total_loss.item():.4f}")
            
            # Stop after 50 batches for debugging (remove this line later)
            if batch_idx >= 50:
                print("  Stopping after 50 batches for debugging...")
                break
        
        # Calculate average losses
        avg_ce_loss = epoch_ce_loss / num_batches if num_batches > 0 else 0
        avg_sparsity_loss = epoch_sparsity_loss / num_batches if num_batches > 0 else 0
        avg_delay_loss = epoch_delay_loss / num_batches if num_batches > 0 else 0
        avg_total_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    CE Loss: {avg_ce_loss:.4f}")
        print(f"    Sparsity Loss: {avg_sparsity_loss:.4f}")
        print(f"    Delay Loss: {avg_delay_loss:.4f}")
        print(f"    Total Loss: {avg_total_loss:.4f}")
        
        # Evaluate after each epoch (with timeout protection)
        print(f"\n  Evaluating...")
        try:
            acc, sparsity, layer_sparsities = evaluate_sparsity(model, loader, device, args)
            print(f"    Accuracy: {acc:.2f}%")
            print(f"    Global Sparsity: {sparsity:.2f}%")
        except Exception as e:
            print(f"    Evaluation failed: {e}")
            acc, sparsity = 0, 0
        
        # Save checkpoint
        if (epoch + 1) % finetune_save_freq == 0:
            save_path = os.path.join(finetune_output, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
                'sparsity': sparsity,
            }, save_path)
            print(f"  Saved checkpoint to {save_path}")
        
        # Save best model
        if sparsity > best_sparsity and acc > best_acc - 2:
            best_sparsity = sparsity
            best_acc = acc
            best_path = os.path.join(finetune_output, 'best_sparsity_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': acc,
                'sparsity': sparsity,
            }, best_path)
            print(f"  ★ New best sparsity model! (sparsity={sparsity:.2f}%, acc={acc:.2f}%)")
        
        # Early stopping if sparsity is high enough
        if sparsity > 70 and acc > 90:
            print(f"\n  Early stopping: Target sparsity reached!")
            break
            
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE ({total_time})")
    print(f"{'='*60}")
    print(f"Best model:")
    print(f"  Accuracy:  {best_acc:.2f}%")
    print(f"  Sparsity:  {best_sparsity:.2f}%")
    print(f"  Score:     {best_score:.1f}")
    print(f"Saved to: {finetune_output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt evaluation + fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)