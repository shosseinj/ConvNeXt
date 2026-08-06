# Experiment Report: cifar10_dense_native32_k3_ttfs_score_layernorm_seed42

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar10_dense_native32_k3_ttfs_score_layernorm_seed42 |
| Date Time | 2026-08-06T17:58:27+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_dense_native32_k3_ttfs_score_layernorm_seed42 |
| Notes | unknown |
| Seed | 42 |
| Status | running |
| Updated At | 2026-08-06T18:07:06+03:30 |

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
| Dims | [96,192,384,768] |
| Depths | [2,2,6,2] |
| Parameter Count | 20844874 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 3 |
| Downsample Kernel | 3 |
| Downsample Stride | 2 |
| Downsample Padding | 1 |
| Residual Operator | min |
| Pw1 Mode | continuous TTFS |
| Pw2 Mode | dense |
| Ttfs Norm Mode | score_layernorm |
| Spike Dropout | 0.0 |
| Delay Enabled | true |
| Stage Delays | [0.05,0.02,0.01,0.01] |
| Delay Parameterization | max_delay * sigmoid(raw_delay) |
| T Min | 0.0 |
| T Max | 1.0 |

## Training

| Field | Value |
|---|---|
| Epochs | 350 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 0.0004 |
| Lr Scheduler | ReduceLROnPlateau(mode=max) |
| Lr Scheduler Patience | 6 |
| Lr Scheduler Factor | 0.8 |
| Minimum Learning Rate | 1e-06 |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Head Dropout | 0.1 |
| Mixup Alpha | 0.2 |
| Early Stopping Patience | 20 |
| Ema Enabled | true |
| Ema Decay | 0.9998 |

## Results

| Field | Value |
|---|---|
| Best Epoch | 18 |
| Best Validation Accuracy | 60.06 |
| Final Train Accuracy | 64.24606179959082 |
| Final Validation Accuracy | 60.06 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 418.20691323280334 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_dense_native32_k3_ttfs_score_layernorm_seed42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
