# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_6543\initial_constraint\initial_constrained_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy=  0.70%
Batch   40/  79 | samples=  5120 | accuracy=  0.76%
Batch   60/  79 | samples=  7680 | accuracy=  0.79%
Batch   79/  79 | samples= 10000 | accuracy=  0.86%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      34,438,862   491,520,000     7.01%
downsample_layers.2.0                                            downsample      50,189,427   245,760,000    20.42%
downsample_layers.3.0                                            downsample      43,172,126   122,880,000    35.13%
stages.0.0.dwconv                                                dwconv              20,668   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1                  2,119 3,932,160,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv             310,139   983,040,000     0.03%
stages.0.1.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          26,947,225   491,520,000     5.48%
stages.1.0.pw1_ttfs                                              pw1            698,329,606 1,966,080,000    35.52%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          68,808,666   491,520,000    14.00%
stages.1.1.pw1_ttfs                                              pw1            612,949,256 1,966,080,000    31.18%
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.2.0.dwconv                                                dwconv          68,095,709   245,760,000    27.71%
stages.2.0.pw1_ttfs                                              pw1            274,135,725   983,040,000    27.89%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          79,041,392   245,760,000    32.16%
stages.2.1.pw1_ttfs                                              pw1            376,203,078   983,040,000    38.27%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          81,894,185   245,760,000    33.32%
stages.2.2.pw1_ttfs                                              pw1            312,141,630   983,040,000    31.75%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          94,797,290   245,760,000    38.57%
stages.2.3.pw1_ttfs                                              pw1            366,522,322   983,040,000    37.28%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          74,575,116   245,760,000    30.34%
stages.2.4.pw1_ttfs                                              pw1            482,155,206   983,040,000    49.05%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          92,573,015   245,760,000    37.67%
stages.2.5.pw1_ttfs                                              pw1            296,815,708   983,040,000    30.19%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          41,012,576   122,880,000    33.38%
stages.3.0.pw1_ttfs                                              pw1            241,654,255   491,520,000    49.16%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          47,326,859   122,880,000    38.51%
stages.3.1.pw1_ttfs                                              pw1            259,932,124   491,520,000    52.88%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     32.53%
===================================================================================================================

Classification accuracy: 0.86%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.46% | silent=675,402,840 | total=4,669,440,000
pw1             12 layers | sparsity=  20.99% | silent=3,920,841,029 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  14.86% | silent=127,800,415 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.67% | TTFS points= 6 | silent=1,966,412,926 | total=11,796,480,000
Stage 1:   37.94% | TTFS points= 7 | silent=2,424,513,615 | total=6,389,760,000
Stage 2:   45.35% | TTFS points=19 | silent=4,123,699,803 | total=9,093,120,000
Stage 3:   55.02% | TTFS points= 7 | silent=878,857,940 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             0.86%
Activation sparsity:  32.53%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_6543\initial_constraint\activation_sparsity.md
```
