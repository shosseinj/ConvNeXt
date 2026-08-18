# Experiment Report: cifar100_ann_to_fully_ttfs_seed42

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | cifar100_ann_to_fully_ttfs_seed42 |
| Date Time | 2026-08-17T18:28:15+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v4\cifar100\ann_to_fully_ttfs\seed_42 |
| Notes | Fully-TTFS fine-tuning from the CIFAR-100 seed-42 accuracy-oriented ANN EMA checkpoint |
| Seed | 42 |
| Status | completed |
| Updated At | 2026-08-17T21:26:11+03:30 |

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
| Pretrained Checkpoint | unknown |
| Ann Pretrained Checkpoint | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\accuracy_oriented_results_v7\cifar100\interpolated_imagenet_pretrain\seed_42\best_checkpoint.pth |
| Pretrained Initialization | {"initialization_type":"fully_dense_ann_to_fully_ttfs","source_state":"ema","source_parameter_keys":128,"transferred_parameter_keys":120,"parameter_mapping":[{"source":"downsample_layers.0.0.weight","target":"downsample_layers.0.0.weight"},{"source":"downsample_layers.0.0.bias","target":"downsample_layers.0.0.bias"},{"source":"downsample_layers.1.1.weight","target":"downsample_layers.1.0.conv.weight"},{"source":"downsample_layers.1.1.bias","target":"downsample_layers.1.0.conv.bias"},{"source":"downsample_layers.2.1.weight","target":"downsample_layers.2.0.conv.weight"},{"source":"downsample_layers.2.1.bias","target":"downsample_layers.2.0.conv.bias"},{"source":"downsample_layers.3.1.weight","target":"downsample_layers.3.0.conv.weight"},{"source":"downsample_layers.3.1.bias","target":"downsample_layers.3.0.conv.bias"},{"source":"stages.0.0.gamma","target":"stages.0.0.gamma"},{"source":"stages.0.0.dwconv.weight","target":"stages.0.0.dwconv.conv.weight"},{"source":"stages.0.0.dwconv.bias","target":"stages.0.0.dwconv.conv.bias"},{"source":"stages.0.0.norm.weight","target":"stages.0.0.norm.weight"},{"source":"stages.0.0.norm.bias","target":"stages.0.0.norm.bias"},{"source":"stages.0.0.pwconv1.weight","target":"stages.0.0.pw1.weight"},{"source":"stages.0.0.pwconv1.bias","target":"stages.0.0.pw1.bias"},{"source":"stages.0.0.pwconv2.weight","target":"stages.0.0.pw2.weight"},{"source":"stages.0.0.pwconv2.bias","target":"stages.0.0.pw2.bias"},{"source":"stages.0.1.gamma","target":"stages.0.1.gamma"},{"source":"stages.0.1.dwconv.weight","target":"stages.0.1.dwconv.conv.weight"},{"source":"stages.0.1.dwconv.bias","target":"stages.0.1.dwconv.conv.bias"},{"source":"stages.0.1.norm.weight","target":"stages.0.1.norm.weight"},{"source":"stages.0.1.norm.bias","target":"stages.0.1.norm.bias"},{"source":"stages.0.1.pwconv1.weight","target":"stages.0.1.pw1.weight"},{"source":"stages.0.1.pwconv1.bias","target":"stages.0.1.pw1.bias"},{"source":"stages.0.1.pwconv2.weight","target":"stages.0.1.pw2.weight"},{"source":"stages.0.1.pwconv2.bias","target":"stages.0.1.pw2.bias"},{"source":"stages.1.0.gamma","target":"stages.1.0.gamma"},{"source":"stages.1.0.dwconv.weight","target":"stages.1.0.dwconv.conv.weight"},{"source":"stages.1.0.dwconv.bias","target":"stages.1.0.dwconv.conv.bias"},{"source":"stages.1.0.norm.weight","target":"stages.1.0.norm.weight"},{"source":"stages.1.0.norm.bias","target":"stages.1.0.norm.bias"},{"source":"stages.1.0.pwconv1.weight","target":"stages.1.0.pw1.weight"},{"source":"stages.1.0.pwconv1.bias","target":"stages.1.0.pw1.bias"},{"source":"stages.1.0.pwconv2.weight","target":"stages.1.0.pw2.weight"},{"source":"stages.1.0.pwconv2.bias","target":"stages.1.0.pw2.bias"},{"source":"stages.1.1.gamma","target":"stages.1.1.gamma"},{"source":"stages.1.1.dwconv.weight","target":"stages.1.1.dwconv.conv.weight"},{"source":"stages.1.1.dwconv.bias","target":"stages.1.1.dwconv.conv.bias"},{"source":"stages.1.1.norm.weight","target":"stages.1.1.norm.weight"},{"source":"stages.1.1.norm.bias","target":"stages.1.1.norm.bias"},{"source":"stages.1.1.pwconv1.weight","target":"stages.1.1.pw1.weight"},{"source":"stages.1.1.pwconv1.bias","target":"stages.1.1.pw1.bias"},{"source":"stages.1.1.pwconv2.weight","target":"stages.1.1.pw2.weight"},{"source":"stages.1.1.pwconv2.bias","target":"stages.1.1.pw2.bias"},{"source":"stages.2.0.gamma","target":"stages.2.0.gamma"},{"source":"stages.2.0.dwconv.weight","target":"stages.2.0.dwconv.conv.weight"},{"source":"stages.2.0.dwconv.bias","target":"stages.2.0.dwconv.conv.bias"},{"source":"stages.2.0.norm.weight","target":"stages.2.0.norm.weight"},{"source":"stages.2.0.norm.bias","target":"stages.2.0.norm.bias"},{"source":"stages.2.0.pwconv1.weight","target":"stages.2.0.pw1.weight"},{"source":"stages.2.0.pwconv1.bias","target":"stages.2.0.pw1.bias"},{"source":"stages.2.0.pwconv2.weight","target":"stages.2.0.pw2.weight"},{"source":"stages.2.0.pwconv2.bias","target":"stages.2.0.pw2.bias"},{"source":"stages.2.1.gamma","target":"stages.2.1.gamma"},{"source":"stages.2.1.dwconv.weight","target":"stages.2.1.dwconv.conv.weight"},{"source":"stages.2.1.dwconv.bias","target":"stages.2.1.dwconv.conv.bias"},{"source":"stages.2.1.norm.weight","target":"stages.2.1.norm.weight"},{"source":"stages.2.1.norm.bias","target":"stages.2.1.norm.bias"},{"source":"stages.2.1.pwconv1.weight","target":"stages.2.1.pw1.weight"},{"source":"stages.2.1.pwconv1.bias","target":"stages.2.1.pw1.bias"},{"source":"stages.2.1.pwconv2.weight","target":"stages.2.1.pw2.weight"},{"source":"stages.2.1.pwconv2.bias","target":"stages.2.1.pw2.bias"},{"source":"stages.2.2.gamma","target":"stages.2.2.gamma"},{"source":"stages.2.2.dwconv.weight","target":"stages.2.2.dwconv.conv.weight"},{"source":"stages.2.2.dwconv.bias","target":"stages.2.2.dwconv.conv.bias"},{"source":"stages.2.2.norm.weight","target":"stages.2.2.norm.weight"},{"source":"stages.2.2.norm.bias","target":"stages.2.2.norm.bias"},{"source":"stages.2.2.pwconv1.weight","target":"stages.2.2.pw1.weight"},{"source":"stages.2.2.pwconv1.bias","target":"stages.2.2.pw1.bias"},{"source":"stages.2.2.pwconv2.weight","target":"stages.2.2.pw2.weight"},{"source":"stages.2.2.pwconv2.bias","target":"stages.2.2.pw2.bias"},{"source":"stages.2.3.gamma","target":"stages.2.3.gamma"},{"source":"stages.2.3.dwconv.weight","target":"stages.2.3.dwconv.conv.weight"},{"source":"stages.2.3.dwconv.bias","target":"stages.2.3.dwconv.conv.bias"},{"source":"stages.2.3.norm.weight","target":"stages.2.3.norm.weight"},{"source":"stages.2.3.norm.bias","target":"stages.2.3.norm.bias"},{"source":"stages.2.3.pwconv1.weight","target":"stages.2.3.pw1.weight"},{"source":"stages.2.3.pwconv1.bias","target":"stages.2.3.pw1.bias"},{"source":"stages.2.3.pwconv2.weight","target":"stages.2.3.pw2.weight"},{"source":"stages.2.3.pwconv2.bias","target":"stages.2.3.pw2.bias"},{"source":"stages.2.4.gamma","target":"stages.2.4.gamma"},{"source":"stages.2.4.dwconv.weight","target":"stages.2.4.dwconv.conv.weight"},{"source":"stages.2.4.dwconv.bias","target":"stages.2.4.dwconv.conv.bias"},{"source":"stages.2.4.norm.weight","target":"stages.2.4.norm.weight"},{"source":"stages.2.4.norm.bias","target":"stages.2.4.norm.bias"},{"source":"stages.2.4.pwconv1.weight","target":"stages.2.4.pw1.weight"},{"source":"stages.2.4.pwconv1.bias","target":"stages.2.4.pw1.bias"},{"source":"stages.2.4.pwconv2.weight","target":"stages.2.4.pw2.weight"},{"source":"stages.2.4.pwconv2.bias","target":"stages.2.4.pw2.bias"},{"source":"stages.2.5.gamma","target":"stages.2.5.gamma"},{"source":"stages.2.5.dwconv.weight","target":"stages.2.5.dwconv.conv.weight"},{"source":"stages.2.5.dwconv.bias","target":"stages.2.5.dwconv.conv.bias"},{"source":"stages.2.5.norm.weight","target":"stages.2.5.norm.weight"},{"source":"stages.2.5.norm.bias","target":"stages.2.5.norm.bias"},{"source":"stages.2.5.pwconv1.weight","target":"stages.2.5.pw1.weight"},{"source":"stages.2.5.pwconv1.bias","target":"stages.2.5.pw1.bias"},{"source":"stages.2.5.pwconv2.weight","target":"stages.2.5.pw2.weight"},{"source":"stages.2.5.pwconv2.bias","target":"stages.2.5.pw2.bias"},{"source":"stages.3.0.gamma","target":"stages.3.0.gamma"},{"source":"stages.3.0.dwconv.weight","target":"stages.3.0.dwconv.conv.weight"},{"source":"stages.3.0.dwconv.bias","target":"stages.3.0.dwconv.conv.bias"},{"source":"stages.3.0.norm.weight","target":"stages.3.0.norm.weight"},{"source":"stages.3.0.norm.bias","target":"stages.3.0.norm.bias"},{"source":"stages.3.0.pwconv1.weight","target":"stages.3.0.pw1.weight"},{"source":"stages.3.0.pwconv1.bias","target":"stages.3.0.pw1.bias"},{"source":"stages.3.0.pwconv2.weight","target":"stages.3.0.pw2.weight"},{"source":"stages.3.0.pwconv2.bias","target":"stages.3.0.pw2.bias"},{"source":"stages.3.1.gamma","target":"stages.3.1.gamma"},{"source":"stages.3.1.dwconv.weight","target":"stages.3.1.dwconv.conv.weight"},{"source":"stages.3.1.dwconv.bias","target":"stages.3.1.dwconv.conv.bias"},{"source":"stages.3.1.norm.weight","target":"stages.3.1.norm.weight"},{"source":"stages.3.1.norm.bias","target":"stages.3.1.norm.bias"},{"source":"stages.3.1.pwconv1.weight","target":"stages.3.1.pw1.weight"},{"source":"stages.3.1.pwconv1.bias","target":"stages.3.1.pw1.bias"},{"source":"stages.3.1.pwconv2.weight","target":"stages.3.1.pw2.weight"},{"source":"stages.3.1.pwconv2.bias","target":"stages.3.1.pw2.bias"},{"source":"norm.weight","target":"final_norm.weight"},{"source":"norm.bias","target":"final_norm.bias"},{"source":"head.weight","target":"head.weight"},{"source":"head.bias","target":"head.bias"}],"excluded_source_keys":["downsample_layers.0.1.bias","downsample_layers.0.1.weight","downsample_layers.1.0.bias","downsample_layers.1.0.weight","downsample_layers.2.0.bias","downsample_layers.2.0.weight","downsample_layers.3.0.bias","downsample_layers.3.0.weight"],"initialized_delay_keys":["downsample_layers.1.0.D_conv","downsample_layers.2.0.D_conv","downsample_layers.3.0.D_conv","stages.0.0.D_mid","stages.0.0.D_out","stages.0.0.dwconv.D_conv","stages.0.1.D_mid","stages.0.1.D_out","stages.0.1.dwconv.D_conv","stages.1.0.D_mid","stages.1.0.D_out","stages.1.0.dwconv.D_conv","stages.1.1.D_mid","stages.1.1.D_out","stages.1.1.dwconv.D_conv","stages.2.0.D_mid","stages.2.0.D_out","stages.2.0.dwconv.D_conv","stages.2.1.D_mid","stages.2.1.D_out","stages.2.1.dwconv.D_conv","stages.2.2.D_mid","stages.2.2.D_out","stages.2.2.dwconv.D_conv","stages.2.3.D_mid","stages.2.3.D_out","stages.2.3.dwconv.D_conv","stages.2.4.D_mid","stages.2.4.D_out","stages.2.4.dwconv.D_conv","stages.2.5.D_mid","stages.2.5.D_out","stages.2.5.dwconv.D_conv","stages.3.0.D_mid","stages.3.0.D_out","stages.3.0.dwconv.D_conv","stages.3.1.D_mid","stages.3.1.D_out","stages.3.1.dwconv.D_conv"],"missing_keys":[],"unexpected_keys":[],"source_best_epoch":295,"source_best_validation_accuracy":86.0,"source_checkpoint":"C:\\Users\\jafari.h\\Desktop\\ai_project\\ConvNeXt\\accuracy_oriented_results_v7\\cifar100\\interpolated_imagenet_pretrain\\seed_42\\best_checkpoint.pth"} |
| Constrained Finetune Initialization | unknown |

## Results

| Field | Value |
|---|---|
| Best Epoch | 228 |
| Best Validation Accuracy | 69.74 |
| Final Train Accuracy | 63.78433641409307 |
| Final Validation Accuracy | 69.46 |
| Test Accuracy | 68.79 |
| Test Loss | 1.8079791381835937 |
| Training Time Seconds | 10676.094804048538 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v4\cifar100\ann_to_fully_ttfs\seed_42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
