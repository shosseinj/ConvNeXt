# TTFS convolution-layer activation sparsity

This table includes only analytic TTFS depthwise and downsampling convolutions. The available CIFAR-10 and CIFAR-100 values are seed-42 measurements over their complete 10,000-image test sets. Tiny ImageNet is marked unavailable because its three existing checkpoints use dense depthwise and dense downsampling convolutions.

## TTFS depthwise convolutions

| Stage | Block | CIFAR-10 sparsity (%) | CIFAR-100 sparsity (%) | Tiny ImageNet sparsity (%) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.03 | 0.00 | N/A |
| 1 | 2 | 0.00 | 0.03 | N/A |
| 2 | 1 | 23.94 | 20.00 | N/A |
| 2 | 2 | 12.86 | 15.16 | N/A |
| 3 | 1 | 37.82 | 39.30 | N/A |
| 3 | 2 | 33.23 | 34.27 | N/A |
| 3 | 3 | 30.89 | 29.61 | N/A |
| 3 | 4 | 28.68 | 30.77 | N/A |
| 3 | 5 | 30.79 | 32.12 | N/A |
| 3 | 6 | 31.48 | 33.43 | N/A |
| 4 | 1 | 38.76 | 47.50 | N/A |
| 4 | 2 | 44.91 | 46.78 | N/A |

## TTFS downsampling convolutions

| Downsampling transition | CIFAR-10 sparsity (%) | CIFAR-100 sparsity (%) | Tiny ImageNet sparsity (%) |
|---|---:|---:|---:|
| Stage 1 to Stage 2 | 35.47 | 36.10 | N/A |
| Stage 2 to Stage 3 | 72.63 | 69.52 | N/A |
| Stage 3 to Stage 4 | 78.63 | 80.28 | N/A |

## Evaluation status

| Dataset | DWConv mode | Downsampling mode | Seed coverage | Status |
|---|---|---|---|---|
| CIFAR-10 | TTFS | TTFS | 42 only | 15 convolution points measured |
| CIFAR-100 | TTFS | TTFS | 42 only | 15 convolution points measured |
| Tiny ImageNet | Dense | Dense | 42, 2344, 5435 | No TTFS convolution points available |

The CIFAR checkpoints also contain 24 pointwise TTFS measurements (12 PW1 and 12 PW2), giving 39 total TTFS points per checkpoint. They are omitted here because this report is restricted to depthwise and downsampling convolutions.

