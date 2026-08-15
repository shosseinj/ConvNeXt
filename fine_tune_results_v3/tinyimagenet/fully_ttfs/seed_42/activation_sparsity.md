# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: tinyimagenet
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
Batch    1/  79 | samples=   128 | accuracy= 63.28%
Batch   20/  79 | samples=  2560 | accuracy= 62.19%
Batch   40/  79 | samples=  5120 | accuracy= 62.09%
Batch   60/  79 | samples=  7680 | accuracy= 61.99%
Batch   79/  79 | samples= 10000 | accuracy= 61.78%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     170,098,788 1,966,080,000     8.65%
downsample_layers.2.0                                            downsample     292,648,157   983,040,000    29.77%
downsample_layers.3.0                                            downsample     392,729,621   491,520,000    79.90%
stages.0.0.dwconv                                                dwconv             303,548 3,932,160,000     0.01%
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            609,312,852 3,932,160,000    15.50%
stages.0.1.dwconv                                                dwconv          19,875,239 3,932,160,000     0.51%
stages.0.1.pw1_ttfs                                              pw1          5,077,056,38415,728,640,000    32.28%
stages.0.1.pw2_ttfs                                              pw2            730,805,063 3,932,160,000    18.59%
stages.1.0.dwconv                                                dwconv          95,295,442 1,966,080,000     4.85%
stages.1.0.pw1_ttfs                                              pw1          1,779,365,797 7,864,320,000    22.63%
stages.1.0.pw2_ttfs                                              pw2          1,585,869,913 1,966,080,000    80.66%
stages.1.1.dwconv                                                dwconv          68,013,603 1,966,080,000     3.46%
stages.1.1.pw1_ttfs                                              pw1            605,397,239 7,864,320,000     7.70%
stages.1.1.pw2_ttfs                                              pw2          1,692,613,971 1,966,080,000    86.09%
stages.2.0.dwconv                                                dwconv         150,977,945   983,040,000    15.36%
stages.2.0.pw1_ttfs                                              pw1            495,182,459 3,932,160,000    12.59%
stages.2.0.pw2_ttfs                                              pw2            920,672,375   983,040,000    93.66%
stages.2.1.dwconv                                                dwconv         219,497,846   983,040,000    22.33%
stages.2.1.pw1_ttfs                                              pw1            548,392,396 3,932,160,000    13.95%
stages.2.1.pw2_ttfs                                              pw2            826,310,209   983,040,000    84.06%
stages.2.2.dwconv                                                dwconv         314,029,374   983,040,000    31.94%
stages.2.2.pw1_ttfs                                              pw1            124,223,645 3,932,160,000     3.16%
stages.2.2.pw2_ttfs                                              pw2            929,958,991   983,040,000    94.60%
stages.2.3.dwconv                                                dwconv         364,513,045   983,040,000    37.08%
stages.2.3.pw1_ttfs                                              pw1            132,206,616 3,932,160,000     3.36%
stages.2.3.pw2_ttfs                                              pw2            944,605,084   983,040,000    96.09%
stages.2.4.dwconv                                                dwconv         343,967,655   983,040,000    34.99%
stages.2.4.pw1_ttfs                                              pw1            259,034,222 3,932,160,000     6.59%
stages.2.4.pw2_ttfs                                              pw2            952,976,966   983,040,000    96.94%
stages.2.5.dwconv                                                dwconv         324,348,657   983,040,000    32.99%
stages.2.5.pw1_ttfs                                              pw1            155,536,000 3,932,160,000     3.96%
stages.2.5.pw2_ttfs                                              pw2            936,981,238   983,040,000    95.31%
stages.3.0.dwconv                                                dwconv          71,798,246   491,520,000    14.61%
stages.3.0.pw1_ttfs                                              pw1            908,753,918 1,966,080,000    46.22%
stages.3.0.pw2_ttfs                                              pw2            433,680,473   491,520,000    88.23%
stages.3.1.dwconv                                                dwconv         183,928,890   491,520,000    37.42%
stages.3.1.pw1_ttfs                                              pw1            978,040,313 1,966,080,000    49.75%
stages.3.1.pw2_ttfs                                              pw2            363,709,049   491,520,000    74.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     21.65%
===================================================================================================================

Classification accuracy: 61.78%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  11.55% | silent=2,156,549,490 | total=18,677,760,000
pw1             12 layers | sparsity=  14.81% | silent=11,063,188,989 | total=74,711,040,000
pw2             12 layers | sparsity=  58.51% | silent=10,927,496,184 | total=18,677,760,000
downsample       3 layers | sparsity=  24.86% | silent=855,476,566 | total=3,440,640,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   13.64% | TTFS points= 6 | silent=6,437,353,086 | total=47,185,920,000
Stage 1:   23.46% | TTFS points= 7 | silent=5,996,654,753 | total=25,559,040,000
Stage 2:   25.39% | TTFS points=19 | silent=9,236,062,880 | total=36,372,480,000
Stage 3:   52.16% | TTFS points= 7 | silent=3,332,640,510 | total=6,389,760,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             61.78%
Activation sparsity:  21.65%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_42\activation_sparsity.md
```
