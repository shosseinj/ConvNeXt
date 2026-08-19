# Experiment Report: cifar100_delay_regularization_lambda_0p01_seed_7777_seed7777

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_delay_regularization_lambda_0p01_seed_7777_seed7777 |
| Date Time | 2026-08-19T01:10:33+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_7777 |
| Notes | Delay regularization ablation: explicit mean effective D_mid/D_out penalty |
| Seed | 7777 |
| Status | early_stopped |
| Updated At | 2026-08-19T02:15:09+03:30 |

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
| Convolution Delay Learning Rate | unknown |
| Delay Regularization Weight | 0.01 |
| Delay Regularization Definition | mean effective bounded D_mid/D_out delay |
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
| Best Epoch | 174 |
| Best Validation Accuracy | 74.74 |
| Final Train Accuracy | 70.65062507749423 |
| Final Validation Accuracy | 74.48 |
| Test Accuracy | 73.18 |
| Test Loss | 1.6686540975570678 |
| Training Time Seconds | 3875.239966392517 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_7777\best_checkpoint.pth |
| Delay Regularization Weight | 0.01 |
| Final Effective Delays | {"definition":"mean effective bounded D_mid/D_out delay","overall_mean":0.05131307989358902,"per_stage":[{"stage":0,"mid":0.010516820009797812,"out":0.010473318863660097},{"stage":1,"mid":0.08388028666377068,"out":0.011202673893421888},{"stage":2,"mid":0.05687240076561769,"out":0.020110672650237877},{"stage":3,"mid":0.05490817688405514,"out":0.05334380269050598}]} |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
