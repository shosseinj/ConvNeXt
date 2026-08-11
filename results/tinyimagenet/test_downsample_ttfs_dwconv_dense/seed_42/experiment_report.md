# Experiment Report: tinyimagenet_test_downsample_ttfs_dwconv_dense_seed42

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | tinyimagenet_test_downsample_ttfs_dwconv_dense_seed42 |
| Date Time | 2026-08-11T06:50:21+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_42 |
| Notes | Native 64x64 Tiny ImageNet with analytic TTFS downsampling and dense depthwise convolutions |
| Seed | 42 |
| Status | running |
| Updated At | 2026-08-11T08:42:11+03:30 |

## Dataset

| Field | Value |
|---|---|
| Dataset Name | Tiny ImageNet |
| Number Of Classes | 200 |
| Input Resolution | [64,64] |
| Train Sample Count | 90000 |
| Validation Sample Count | 10000 |
| Test Sample Count | 10000 |
| Preprocessing | augmentation, ToTensor/RandomErasing, optional Mixup/CutMix, then continuous TTFS encoding |
| Augmentation | training: RandomCrop(64,padding=8), RandomHorizontalFlip, RandAugment(enabled=True,ops=2,magnitude=9), RandomErasing(p=0.1), Mixup(alpha=0.2), CutMix(alpha=1.0); validation/test: ToTensor only |

## Architecture

| Field | Value |
|---|---|
| Dims | [96,192,384,768] |
| Depths | [2,2,6,2] |
| Parameter Count | 20996936 |
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
| Stage Delays | [0.05,0.02,0.01,0.01] |
| Delay Parameterization | max_delay * sigmoid(raw_delay) |
| T Min | 0.0 |
| T Max | 1.0 |

## Training

| Field | Value |
|---|---|
| Epochs | 300 |
| Batch Size | 32 |
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
| Best Epoch | 42 |
| Best Validation Accuracy | 58.1 |
| Final Train Accuracy | 40.73557507329004 |
| Final Validation Accuracy | 58.1 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 6710.844173908234 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
