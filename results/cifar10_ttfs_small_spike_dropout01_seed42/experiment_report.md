# Experiment Report: Small TTFS Spike Dropout 0.1

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | Small TTFS Spike Dropout 0.1 |
| Date Time | 2026-08-05T17:50:19+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_small_spike_dropout01_seed42 |
| Notes | TTFS-aware t_out dropout using t_max as the no-spike value |
| Seed | 42 |
| Status | running |
| Updated At | 2026-08-05T18:06:24+03:30 |

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
| Spike Dropout | 0.1 |
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
| Best Epoch | 68 |
| Best Validation Accuracy | 67.26 |
| Final Train Accuracy | 66.19164426888749 |
| Final Validation Accuracy | 67.26 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 964.6412854194641 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_small_spike_dropout01_seed42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
