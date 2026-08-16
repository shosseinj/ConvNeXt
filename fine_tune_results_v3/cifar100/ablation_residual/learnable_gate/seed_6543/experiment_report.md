# Experiment Report: cifar100_residual_learnable_gate_seed6543_seed6543

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_residual_learnable_gate_seed6543_seed6543 |
| Date Time | 2026-08-16T13:25:10+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_6543 |
| Notes | Reviewer 1 Comment 4: Fully-TTFS residual fusion ablation initialized from the matched dense best checkpoint |
| Seed | 6543 |
| Status | early_stopped |
| Updated At | 2026-08-16T15:04:06+03:30 |

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
| Parameter Count | 20930212 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 3 |
| Depthwise Mode | ttfs |
| Downsample Kernel | 3 |
| Downsample Stride | 2 |
| Downsample Padding | 1 |
| Downsample Mode | ttfs |
| Residual Operator | learnable_gate |
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
| Pretrained Checkpoint | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\downsample_dense_dwconv_dense\seed_6543\best_checkpoint.pth |
| Pretrained Initialization | {"source_state":"ema","source_parameter_keys":144,"transferred_parameter_keys":144,"initialized_delay_keys":["downsample_layers.1.0.D_conv","downsample_layers.2.0.D_conv","downsample_layers.3.0.D_conv","stages.0.0.dwconv.D_conv","stages.0.1.dwconv.D_conv","stages.1.0.dwconv.D_conv","stages.1.1.dwconv.D_conv","stages.2.0.dwconv.D_conv","stages.2.1.dwconv.D_conv","stages.2.2.dwconv.D_conv","stages.2.3.dwconv.D_conv","stages.2.4.dwconv.D_conv","stages.2.5.dwconv.D_conv","stages.3.0.dwconv.D_conv","stages.3.1.dwconv.D_conv"],"initialized_gate_keys":["stages.0.0.raw_residual_gate","stages.0.1.raw_residual_gate","stages.1.0.raw_residual_gate","stages.1.1.raw_residual_gate","stages.2.0.raw_residual_gate","stages.2.1.raw_residual_gate","stages.2.2.raw_residual_gate","stages.2.3.raw_residual_gate","stages.2.4.raw_residual_gate","stages.2.5.raw_residual_gate","stages.3.0.raw_residual_gate","stages.3.1.raw_residual_gate"],"source_residual_operator":"min","target_residual_operator":"learnable_gate","missing_keys":[],"unexpected_keys":[],"source_checkpoint":"C:\\Users\\jafari.h\\Desktop\\ai_project\\ConvNeXt\\results\\cifar100\\downsample_dense_dwconv_dense\\seed_6543\\best_checkpoint.pth"} |
| Constrained Finetune Initialization | unknown |

## Results

| Field | Value |
|---|---|
| Best Epoch | 183 |
| Best Validation Accuracy | 70.46 |
| Final Train Accuracy | 67.12548962648185 |
| Final Validation Accuracy | 70.44 |
| Test Accuracy | 70.84 |
| Test Loss | 1.7674924060821533 |
| Training Time Seconds | 5935.760269403458 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_6543\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
