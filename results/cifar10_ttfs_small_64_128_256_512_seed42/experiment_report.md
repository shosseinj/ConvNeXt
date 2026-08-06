# Experiment Report: cifar10_ttfs_small_64_128_256_512_seed42

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar10_ttfs_small_64_128_256_512_seed42 |
| Date Time | 2026-08-05T17:31:31+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_small_64_128_256_512_seed42 |
| Notes | unknown |
| Seed | 42 |
| Status | running |
| Updated At | 2026-08-05T17:44:47+03:30 |

## Dataset

| Field | Value |
|---|---|
| Dataset Name | CIFAR-10 |
| Number Of Classes | 10 |
| Input Resolution | [32,32] |
| Train Sample Count | 45000 |
| Validation Sample Count | 5000 |
| Test Sample Count | 10000 |
| Preprocessing | ToTensor to raw [0,1], optional training Mixup, then continuous TTFS encoding |
| Augmentation | training: RandomCrop(32,padding=4), RandomHorizontalFlip, Mixup(alpha=0.2); validation/test: ToTensor only |

## Architecture

| Field | Value |
|---|---|
| Dims | [64,128,256,512] |
| Depths | [2,2,6,2] |
| Parameter Count | 8543242 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Residual Operator | min |
| Pw1 Mode | continuous TTFS |
| Pw2 Mode | continuous TTFS |
| Delay Enabled | true |
| Stage Delays | [0.4,0.0,0.0,0.0] |
| T Min | 0.0 |
| T Max | 1.0 |

## Training

| Field | Value |
|---|---|
| Epochs | 200 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 0.0002 |
| Weight Decay | 0.1 |
| Label Smoothing | 0.1 |
| Head Dropout | 0.2 |
| Mixup Alpha | 0.2 |
| Early Stopping Patience | 30 |

## Results

| Field | Value |
|---|---|
| Best Epoch | 153 |
| Best Validation Accuracy | 72.24 |
| Final Train Accuracy | 85.83062351590625 |
| Final Validation Accuracy | 71.74 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 795.9114592075348 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_small_64_128_256_512_seed42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
