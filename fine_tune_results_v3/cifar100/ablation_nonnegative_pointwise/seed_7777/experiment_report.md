# Experiment Report: cifar100_nonnegative_pointwise_seed7777

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_nonnegative_pointwise_seed7777 |
| Date Time | 2026-08-16T06:59:30+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777 |
| Notes | Pointwise-only non-negative effective PW1/PW2 weights via ReLU, initialized from the matched fully-TTFS best checkpoint |
| Seed | 7777 |
| Status | completed |
| Updated At | 2026-08-16T07:16:31+03:30 |

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
| Parameter Count | 20925796 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 3 |
| Depthwise Mode | ttfs |
| Downsample Kernel | 3 |
| Downsample Stride | 2 |
| Downsample Padding | 1 |
| Downsample Mode | ttfs |
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
| Epochs | 50 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 1e-05 |
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
| Early Stopping Patience | 15 |
| Ema Enabled | false |
| Ema Decay | unknown |
| Pretrained Checkpoint | unknown |
| Pretrained Initialization | unknown |
| Constrained Finetune Initialization | {"kind":"fully_ttfs_to_nonnegative_pointwise","source_state":"model","transferred_parameter_keys":159,"missing_keys":[],"unexpected_keys":[],"source_checkpoint":"C:\\Users\\jafari.h\\Desktop\\ai_project\\ConvNeXt\\fine_tune_results_v3\\cifar100\\fully_ttfs\\seed_7777\\best_checkpoint.pth"} |

## Results

| Field | Value |
|---|---|
| Best Epoch | 48 |
| Best Validation Accuracy | 20.12 |
| Final Train Accuracy | 11.55151711285137 |
| Final Validation Accuracy | 19.46 |
| Test Accuracy | 20.17 |
| Test Loss | 3.6253286060333254 |
| Training Time Seconds | 1020.909006357193 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
