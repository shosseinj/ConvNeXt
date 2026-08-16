# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 21.88%
Batch   20/  79 | samples=  2560 | accuracy= 20.78%
Batch   40/  79 | samples=  5120 | accuracy= 20.62%
Batch   60/  79 | samples=  7680 | accuracy= 20.16%
Batch   79/  79 | samples= 10000 | accuracy= 20.18%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      28,174,383   491,520,000     5.73%
downsample_layers.2.0                                            downsample      52,069,281   245,760,000    21.19%
downsample_layers.3.0                                            downsample      95,994,193   122,880,000    78.12%
stages.0.0.dwconv                                                dwconv              20,901   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv              10,000   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          53,050,169   491,520,000    10.79%
stages.1.0.pw1_ttfs                                              pw1            648,939,687 1,966,080,000    33.01%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          61,549,284   491,520,000    12.52%
stages.1.1.pw1_ttfs                                              pw1            421,470,422 1,966,080,000    21.44%
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.2.0.dwconv                                                dwconv          67,275,340   245,760,000    27.37%
stages.2.0.pw1_ttfs                                              pw1            245,257,345   983,040,000    24.95%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          85,175,049   245,760,000    34.66%
stages.2.1.pw1_ttfs                                              pw1            355,667,955   983,040,000    36.18%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          83,470,494   245,760,000    33.96%
stages.2.2.pw1_ttfs                                              pw1            384,238,436   983,040,000    39.09%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          95,498,216   245,760,000    38.86%
stages.2.3.pw1_ttfs                                              pw1            335,146,361   983,040,000    34.09%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          92,572,916   245,760,000    37.67%
stages.2.4.pw1_ttfs                                              pw1            372,356,842   983,040,000    37.88%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          97,649,898   245,760,000    39.73%
stages.2.5.pw1_ttfs                                              pw1            327,534,867   983,040,000    33.32%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          17,342,032   122,880,000    14.11%
stages.3.0.pw1_ttfs                                              pw1            174,843,166   491,520,000    35.57%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          52,331,939   122,880,000    42.59%
stages.3.1.pw1_ttfs                                              pw1            272,327,377   491,520,000    55.41%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     31.48%
===================================================================================================================

Classification accuracy: 20.18%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  15.12% | silent=705,946,238 | total=4,669,440,000
pw1             12 layers | sparsity=  18.94% | silent=3,537,782,458 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  20.49% | silent=176,237,857 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.67% | TTFS points= 6 | silent=1,966,110,901 | total=11,796,480,000
Stage 1:   34.37% | TTFS points= 7 | silent=2,196,223,945 | total=6,389,760,000
Stage 2:   44.74% | TTFS points=19 | silent=4,068,473,000 | total=9,093,120,000
Stage 3:   53.75% | TTFS points= 7 | silent=858,598,707 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             20.18%
Activation sparsity:  31.48%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_7777\activation_sparsity.md
```
