# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_42\initial_constraint\initial_constrained_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)
Detected non-negative effective pointwise weights: True (metadata)

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
Batch    1/  79 | samples=   128 | accuracy=  1.56%
Batch   20/  79 | samples=  2560 | accuracy=  0.94%
Batch   40/  79 | samples=  5120 | accuracy=  1.19%
Batch   60/  79 | samples=  7680 | accuracy=  1.18%
Batch   79/  79 | samples= 10000 | accuracy=  1.31%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      16,858,310   491,520,000     3.43%
downsample_layers.2.0                                            downsample      57,328,780   245,760,000    23.33%
downsample_layers.3.0                                            downsample      41,156,134   122,880,000    33.49%
stages.0.0.dwconv                                                dwconv          18,848,549   983,040,000     1.92%
stages.0.0.pw1_ttfs                                              pw1          1,412,725,130 3,932,160,000    35.93%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv             317,229   983,040,000     0.03%
stages.0.1.pw1_ttfs                                              pw1                 76,834 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          32,940,849   491,520,000     6.70%
stages.1.0.pw1_ttfs                                              pw1            591,966,556 1,966,080,000    30.11%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          75,485,099   491,520,000    15.36%
stages.1.1.pw1_ttfs                                              pw1            547,738,187 1,966,080,000    27.86%
stages.1.1.pw2_ttfs                                              pw2            491,519,997   491,520,000   100.00%
stages.2.0.dwconv                                                dwconv          75,020,024   245,760,000    30.53%
stages.2.0.pw1_ttfs                                              pw1            324,103,597   983,040,000    32.97%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          84,142,227   245,760,000    34.24%
stages.2.1.pw1_ttfs                                              pw1            341,574,750   983,040,000    34.75%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          95,162,915   245,760,000    38.72%
stages.2.2.pw1_ttfs                                              pw1            377,387,288   983,040,000    38.39%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          90,487,750   245,760,000    36.82%
stages.2.3.pw1_ttfs                                              pw1            445,783,121   983,040,000    45.35%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          89,799,461   245,760,000    36.54%
stages.2.4.pw1_ttfs                                              pw1            457,218,799   983,040,000    46.51%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          83,015,908   245,760,000    33.78%
stages.2.5.pw1_ttfs                                              pw1            353,050,440   983,040,000    35.91%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          46,421,302   122,880,000    37.78%
stages.3.0.pw1_ttfs                                              pw1            245,906,200   491,520,000    50.03%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          47,701,160   122,880,000    38.82%
stages.3.1.pw1_ttfs                                              pw1            255,064,011   491,520,000    51.89%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     37.67%
===================================================================================================================

Classification accuracy: 1.31%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  15.83% | silent=739,342,473 | total=4,669,440,000
pw1             12 layers | sparsity=  28.66% | silent=5,352,594,913 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,439,997 | total=4,669,440,000
downsample       3 layers | sparsity=  13.41% | silent=115,343,224 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   28.81% | TTFS points= 6 | silent=3,398,047,742 | total=11,796,480,000
Stage 1:   35.18% | TTFS points= 7 | silent=2,248,028,998 | total=6,389,760,000
Stage 2:   47.82% | TTFS points=19 | silent=4,348,635,060 | total=9,093,120,000
Stage 3:   55.21% | TTFS points= 7 | silent=882,008,807 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             1.31%
Activation sparsity:  37.67%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_42\initial_constraint\activation_sparsity.md
```
