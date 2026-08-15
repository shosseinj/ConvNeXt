# Experiment Report: tinyimagenet_fully_ttfs_seed2344

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | tinyimagenet_fully_ttfs_seed2344 |
| Date Time | 2026-08-15T10:10:10+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_2344 |
| Notes | Full-rate fine-tuning of fully TTFS ConvNeXt from the matched dense best checkpoint |
| Seed | 2344 |
| Status | running |
| Updated At | 2026-08-15T11:16:15+03:30 |

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
| Parameter Count | 21002696 |
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
| Epochs | 250 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
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
| Ema Enabled | false |
| Ema Decay | unknown |
| Pretrained Checkpoint | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_2344\best_checkpoint.pth |
| Pretrained Initialization | {"source_state":"ema","source_parameter_keys":144,"transferred_parameter_keys":144,"initialized_delay_keys":["downsample_layers.1.0.D_conv","downsample_layers.2.0.D_conv","downsample_layers.3.0.D_conv","stages.0.0.dwconv.D_conv","stages.0.1.dwconv.D_conv","stages.1.0.dwconv.D_conv","stages.1.1.dwconv.D_conv","stages.2.0.dwconv.D_conv","stages.2.1.dwconv.D_conv","stages.2.2.dwconv.D_conv","stages.2.3.dwconv.D_conv","stages.2.4.dwconv.D_conv","stages.2.5.dwconv.D_conv","stages.3.0.dwconv.D_conv","stages.3.1.dwconv.D_conv"],"missing_keys":[],"unexpected_keys":[],"source_checkpoint":"C:\\Users\\jafari.h\\Desktop\\ai_project\\ConvNeXt\\results\\tinyimagenet\\test_downsample_ttfs_dwconv_dense\\seed_2344\\best_checkpoint.pth"} |

## Results

| Field | Value |
|---|---|
| Best Epoch | 22 |
| Best Validation Accuracy | 59.54 |
| Final Train Accuracy | 59.37348529395631 |
| Final Validation Accuracy | 59.34 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 3811.035980939865 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_2344\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
