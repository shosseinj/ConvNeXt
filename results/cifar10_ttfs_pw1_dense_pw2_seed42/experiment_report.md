# Experiment Report: Native32 TTFS pw1 Dense pw2

## Experiment Information

| Field | Value |
|---|---|
| Experiment Name | Native32 TTFS pw1 Dense pw2 |
| Date Time | 2026-08-06T14:31:13+03:30 |
| Output Directory | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_pw1_dense_pw2_seed42 |
| Notes | One TTFS pointwise layer; dense score-domain pw2 |
| Seed | 42 |
| Status | running |
| Updated At | 2026-08-06T14:32:17+03:30 |

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
| Dims | [96,192,384,512] |
| Depths | [3,3,6,3] |
| Parameter Count | 15733322 |
| Stem Kernel | 3 |
| Stem Stride | 1 |
| Stem Padding | 1 |
| Depthwise Kernel Size | 3 |
| Residual Operator | min |
| Pw1 Mode | continuous TTFS |
| Pw2 Mode | dense |
| Spike Dropout | 0.0 |
| Delay Enabled | true |
| Stage Delays | [0.4,0.0,0.0,0.0] |
| T Min | 0.0 |
| T Max | 1.0 |

## Training

| Field | Value |
|---|---|
| Epochs | 300 |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Head Dropout | 0.1 |
| Mixup Alpha | 0.2 |
| Early Stopping Patience | 30 |

## Results

| Field | Value |
|---|---|
| Best Epoch | 0 |
| Best Validation Accuracy | 21.44 |
| Final Train Accuracy | 19.375003819768683 |
| Final Validation Accuracy | 21.22 |
| Test Accuracy | unknown |
| Test Loss | unknown |
| Training Time Seconds | 64.41954255104065 |
| Checkpoint Path | C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar10_ttfs_pw1_dense_pw2_seed42\best_checkpoint.pth |

## Optional Evaluation Results

| Field | Value |
|---|---|
| Activation Sparsity | unknown |
| Dense Macs Per Sample | unknown |
| Theoretical Synops Per Sample | unknown |
