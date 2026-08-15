# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v2\cifar10\fully_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar10
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)

Missing keys:    0
Unexpected keys: 0

==========================================================================================
MODEL SPARSITY STRUCTURE
==========================================================================================
Spiking blocks:              12
ContinuousTTFSConv2d:        15
PW1 TTFS outputs:            12
PW2 TTFS outputs:            12
Expected total TTFS points:  39
==========================================================================================

TTFS Conv modules:
  downsample_layers.1.0
  downsample_layers.2.0
  downsample_layers.3.0
  stages.0.0.dwconv
  stages.0.1.dwconv
  stages.1.0.dwconv
  stages.1.1.dwconv
  stages.2.0.dwconv
  stages.2.1.dwconv
  stages.2.2.dwconv
  stages.2.3.dwconv
  stages.2.4.dwconv
  stages.2.5.dwconv
  stages.3.0.dwconv
  stages.3.1.dwconv
Batch    1/  79 | samples=   128 | accuracy= 93.75%
Batch   20/  79 | samples=  2560 | accuracy= 92.42%
Batch   40/  79 | samples=  5120 | accuracy= 92.42%
Batch   60/  79 | samples=  7680 | accuracy= 92.64%
Batch   79/  79 | samples= 10000 | accuracy= 92.66%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      63,059,011   491,520,000    12.83%
downsample_layers.2.0                                            downsample      63,215,144   245,760,000    25.72%
downsample_layers.3.0                                            downsample      72,780,555   122,880,000    59.23%
stages.0.0.dwconv                                                dwconv              10,000   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1              5,520,229 3,932,160,000     0.14%
stages.0.0.pw2_ttfs                                              pw2            264,863,266   983,040,000    26.94%
stages.0.1.dwconv                                                dwconv              10,991   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1              7,783,779 3,932,160,000     0.20%
stages.0.1.pw2_ttfs                                              pw2            215,191,238   983,040,000    21.89%
stages.1.0.dwconv                                                dwconv          35,541,571   491,520,000     7.23%
stages.1.0.pw1_ttfs                                              pw1            923,189,148 1,966,080,000    46.96%
stages.1.0.pw2_ttfs                                              pw2            482,429,592   491,520,000    98.15%
stages.1.1.dwconv                                                dwconv          82,716,298   491,520,000    16.83%
stages.1.1.pw1_ttfs                                              pw1            895,519,505 1,966,080,000    45.55%
stages.1.1.pw2_ttfs                                              pw2            479,040,109   491,520,000    97.46%
stages.2.0.dwconv                                                dwconv          64,903,749   245,760,000    26.41%
stages.2.0.pw1_ttfs                                              pw1            476,841,354   983,040,000    48.51%
stages.2.0.pw2_ttfs                                              pw2            236,102,896   245,760,000    96.07%
stages.2.1.dwconv                                                dwconv          75,393,471   245,760,000    30.68%
stages.2.1.pw1_ttfs                                              pw1            516,195,088   983,040,000    52.51%
stages.2.1.pw2_ttfs                                              pw2            238,000,394   245,760,000    96.84%
stages.2.2.dwconv                                                dwconv          83,266,324   245,760,000    33.88%
stages.2.2.pw1_ttfs                                              pw1            502,099,596   983,040,000    51.08%
stages.2.2.pw2_ttfs                                              pw2            241,228,150   245,760,000    98.16%
stages.2.3.dwconv                                                dwconv          85,524,840   245,760,000    34.80%
stages.2.3.pw1_ttfs                                              pw1            477,472,441   983,040,000    48.57%
stages.2.3.pw2_ttfs                                              pw2            242,282,376   245,760,000    98.58%
stages.2.4.dwconv                                                dwconv          83,565,099   245,760,000    34.00%
stages.2.4.pw1_ttfs                                              pw1            542,024,333   983,040,000    55.14%
stages.2.4.pw2_ttfs                                              pw2            238,794,158   245,760,000    97.17%
stages.2.5.dwconv                                                dwconv          86,929,913   245,760,000    35.37%
stages.2.5.pw1_ttfs                                              pw1            485,428,164   983,040,000    49.38%
stages.2.5.pw2_ttfs                                              pw2            232,774,583   245,760,000    94.72%
stages.3.0.dwconv                                                dwconv          36,255,560   122,880,000    29.50%
stages.3.0.pw1_ttfs                                              pw1            239,808,338   491,520,000    48.79%
stages.3.0.pw2_ttfs                                              pw2            110,607,853   122,880,000    90.01%
stages.3.1.dwconv                                                dwconv          47,338,629   122,880,000    38.52%
stages.3.1.pw1_ttfs                                              pw1            263,586,352   491,520,000    53.63%
stages.3.1.pw2_ttfs                                              pw2            106,571,096   122,880,000    86.73%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     32.22%
===================================================================================================================

Classification accuracy: 92.66%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.59% | silent=681,456,445 | total=4,669,440,000
pw1             12 layers | sparsity=  28.57% | silent=5,335,468,327 | total=18,677,760,000
pw2             12 layers | sparsity=  66.13% | silent=3,087,885,711 | total=4,669,440,000
downsample       3 layers | sparsity=  23.14% | silent=199,054,710 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    4.18% | TTFS points= 6 | silent=493,379,503 | total=11,796,480,000
Stage 1:   46.35% | TTFS points= 7 | silent=2,961,495,234 | total=6,389,760,000
Stage 2:   54.68% | TTFS points=19 | silent=4,972,042,073 | total=9,093,120,000
Stage 3:   54.90% | TTFS points= 7 | silent=876,948,383 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             92.66%
Activation sparsity:  32.22%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v2\cifar10\fully_ttfs\seed_42\activation_sparsity.md
```
