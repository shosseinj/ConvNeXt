# Experiment Report: cifar100_delay_init_low_seed_7777_seed7777

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_delay_init_low_seed_7777_seed7777 |
| Date Time | 2026-08-18T14:54:21+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_7777 |
| Notes | Reviewer 1 Comment 4 - Delay initialization ablation |
| Seed | 7777 |
| Status | early_stopped |
| Updated At | 2026-08-18T17:56:06+03:30 |

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
| Parameter Count | 20920036 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 3 |
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
| Stage Delays | [0.025,0.01,0.005,0.005] |
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
| Convolution Delay Learning Rate | unknown |
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
| Pretrained Checkpoint | unknown |
| Ann Pretrained Checkpoint | unknown |
| Pretrained Initialization | unknown |
| Constrained Finetune Initialization | unknown |

## Results

| Field | Value |
|---|---|
| Best Epoch | 198 |
| Best Validation Accuracy | 75.56 |
| Final Train Accuracy | 71.26746684203924 |
| Final Validation Accuracy | 75.44 |
| Test Accuracy | 74.6 |
| Test Loss | 1.6627251640319824 |
| Training Time Seconds | 10898.09799194336 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_7777\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
