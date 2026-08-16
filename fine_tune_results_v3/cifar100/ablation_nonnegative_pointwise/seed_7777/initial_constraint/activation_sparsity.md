# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777\initial_constraint\initial_constrained_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy=  2.34%
Batch   20/  79 | samples=  2560 | accuracy=  2.11%
Batch   40/  79 | samples=  5120 | accuracy=  1.80%
Batch   60/  79 | samples=  7680 | accuracy=  1.65%
Batch   79/  79 | samples= 10000 | accuracy=  1.64%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      26,552,331   491,520,000     5.40%
downsample_layers.2.0                                            downsample      39,773,300   245,760,000    16.18%
downsample_layers.3.0                                            downsample      50,551,246   122,880,000    41.14%
stages.0.0.dwconv                                                dwconv              23,888   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv              10,000   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          42,614,634   491,520,000     8.67%
stages.1.0.pw1_ttfs                                              pw1            627,013,057 1,966,080,000    31.89%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          55,404,494   491,520,000    11.27%
stages.1.1.pw1_ttfs                                              pw1            419,930,730 1,966,080,000    21.36%
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.2.0.dwconv                                                dwconv          69,295,795   245,760,000    28.20%
stages.2.0.pw1_ttfs                                              pw1            224,304,613   983,040,000    22.82%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          85,119,647   245,760,000    34.64%
stages.2.1.pw1_ttfs                                              pw1            356,102,775   983,040,000    36.22%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          82,676,267   245,760,000    33.64%
stages.2.2.pw1_ttfs                                              pw1            355,111,767   983,040,000    36.12%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          93,454,735   245,760,000    38.03%
stages.2.3.pw1_ttfs                                              pw1            281,037,241   983,040,000    28.59%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          87,396,083   245,760,000    35.56%
stages.2.4.pw1_ttfs                                              pw1            267,978,164   983,040,000    27.26%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          93,375,310   245,760,000    37.99%
stages.2.5.pw1_ttfs                                              pw1            309,136,777   983,040,000    31.45%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          32,510,285   122,880,000    26.46%
stages.3.0.pw1_ttfs                                              pw1            222,663,390   491,520,000    45.30%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          43,707,929   122,880,000    35.57%
stages.3.1.pw1_ttfs                                              pw1            260,636,142   491,520,000    53.03%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     30.46%
===================================================================================================================

Classification accuracy: 1.64%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.68% | silent=685,589,067 | total=4,669,440,000
pw1             12 layers | sparsity=  17.80% | silent=3,323,914,656 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  13.59% | silent=116,876,877 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.67% | TTFS points= 6 | silent=1,966,113,888 | total=11,796,480,000
Stage 1:   33.72% | TTFS points= 7 | silent=2,154,555,246 | total=6,389,760,000
Stage 2:   42.00% | TTFS points=19 | silent=3,819,322,474 | total=9,093,120,000
Stage 3:   53.58% | TTFS points= 7 | silent=855,828,992 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             1.64%
Activation sparsity:  30.46%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777\initial_constraint\activation_sparsity.md
```
