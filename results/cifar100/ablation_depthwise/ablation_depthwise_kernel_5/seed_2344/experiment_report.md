# Experiment Report: cifar100_ablation_depthwise_kernel_5_seed2344

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_ablation_depthwise_kernel_5_seed2344 |
| Date Time | 2026-08-10T09:22:42+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_2344 |
| Notes | Analytic TTFS depthwise and downsampling convolutions |
| Seed | 2344 |
| Status | early_stopped |
| Updated At | 2026-08-10T11:33:05+03:30 |

## Dataset

| Field | Value |
|---|---|
| Dataset Name | CIFAR-100 |
| Number Of Classes | 100 |
| Input Resolution | [32,32] |
| Train Sample Count | 45000 |
| Validation Sample Count | 5000 |
| Test Sample Count | 10000 |
| Preprocessing | augmentation, ToTensor/RandomErasing, optional Mixup/CutMix, then continuous TTFS encoding |
| Augmentation | training: RandomCrop(32,padding=4), RandomHorizontalFlip, RandAugment(enabled=True,ops=2,magnitude=9), RandomErasing(p=0.1), Mixup(alpha=0.2), CutMix(alpha=1.0); validation/test: ToTensor only |

## Architecture

| Field | Value |
|---|---|
| Dims | [96,192,384,768] |
| Depths | [2,2,6,2] |
| Parameter Count | 20990692 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 5 |
| Depthwise Mode | dense |
| Downsample Kernel | 3 |
| Downsample Stride | 2 |
| Downsample Padding | 1 |
| Downsample Mode | dense |
| Residual Operator | min |
| Pw1 Mode | ttfs |
| Pw2 Mode | ttfs |
| Ttfs Norm Mode | score_layernorm |
| Final Score Norm | true |
| Spike Dropout | 0.0 |
| Delay Enabled | true |
| Stage Delays | [0.05,0.02,0.01,0.01] |
| Delay Parameterization | max_delay * sigmoid(raw_delay) |
| T Min | 0.0 |
| T Max | 1.0 |

## Training

| Field | Value |
|---|---|
| Epochs | 300 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 0.0002 |
| Lr Scheduler | ReduceLROnPlateau(mode=max) |
| Lr Scheduler Patience | 3 |
| Lr Scheduler Factor | 0.85 |
| Minimum Learning Rate | 1e-06 |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Head Dropout | 0.1 |
| Mixup Alpha | 0.2 |
| Cutmix Alpha | 1.0 |
| Randaugment | true |
| Randaugment Num Ops | 2 |
| Randaugment Magnitude | 9 |
| Random Erasing | 0.1 |
| Early Stopping Patience | 30 |
| Ema Enabled | true |
| Ema Decay | 0.9998 |

## Results

| Field | Value |
|---|---|
| Best Epoch | 254 |
| Best Validation Accuracy | 72.74 |
| Final Train Accuracy | 71.49534397493422 |
| Final Validation Accuracy | 72.44 |
| Test Accuracy | 72.96 |
| Test Loss | 1.7471488679885865 |
| Training Time Seconds | 7823.055646896362 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_2344\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
