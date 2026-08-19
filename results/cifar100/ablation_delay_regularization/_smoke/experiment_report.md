# Experiment Report: cifar100_delay_regularization_smoke_seed42

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_delay_regularization_smoke_seed42 |
| Date Time | 2026-08-18T22:50:11+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\_smoke |
| Notes | unknown |
| Seed | 42 |
| Status | completed |
| Updated At | 2026-08-18T22:50:21+03:30 |

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
| Augmentation | training: RandomCrop(32,padding=4), RandomHorizontalFlip, RandAugment(enabled=False,ops=2,magnitude=9), RandomErasing(p=0.0), Mixup(alpha=0.0), CutMix(alpha=0.0); validation/test: ToTensor only |

## Architecture

| Field | Value |
|---|---|
| Dims | [8,16,32,64] |
| Depths | [1,1,1,1] |
| Parameter Count | 77436 |
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
| Epochs | 1 |
| Batch Size | 512 |
| Optimizer | AdamW |
| Learning Rate | 0.0002 |
| Convolution Delay Learning Rate | unknown |
| Delay Regularization Weight | 0.1 |
| Delay Regularization Definition | mean effective bounded D_mid/D_out delay |
| Lr Scheduler | ReduceLROnPlateau(mode=max) |
| Lr Scheduler Patience | 3 |
| Lr Scheduler Factor | 0.85 |
| Minimum Learning Rate | 1e-06 |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Head Dropout | 0.1 |
| Mixup Alpha | 0.0 |
| Cutmix Alpha | 0.0 |
| Randaugment | false |
| Randaugment Num Ops | 2 |
| Randaugment Magnitude | 9 |
| Random Erasing | 0.0 |
| Early Stopping Patience | 30 |
| Ema Enabled | false |
| Ema Decay | unknown |
| Pretrained Checkpoint | unknown |
| Ann Pretrained Checkpoint | unknown |
| Pretrained Initialization | unknown |
| Constrained Finetune Initialization | unknown |

## Results

| Field | Value |
|---|---|
| Best Epoch | 0 |
| Best Validation Accuracy | 3.42 |
| Final Train Accuracy | 2.1955555555555555 |
| Final Validation Accuracy | 3.42 |
| Test Accuracy | 3.04 |
| Test Loss | 4.457678929901123 |
| Training Time Seconds | 9.21849513053894 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\_smoke\best_checkpoint.pth |
| Delay Regularization Weight | 0.1 |
| Final Effective Delays | {"definition":"mean effective bounded D_mid/D_out delay","overall_mean":0.013817806728184223,"per_stage":[{"stage":0,"mid":0.049321793019771576,"out":0.049321793019771576},{"stage":1,"mid":0.019735727459192276,"out":0.019735727459192276},{"stage":2,"mid":0.009872819297015667,"out":0.009872819297015667},{"stage":3,"mid":0.009872819297015667,"out":0.009872819297015667}]} |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
