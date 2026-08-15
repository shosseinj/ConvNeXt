# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 92.97%
Batch   20/  79 | samples=  2560 | accuracy= 93.12%
Batch   40/  79 | samples=  5120 | accuracy= 93.16%
Batch   60/  79 | samples=  7680 | accuracy= 93.61%
Batch   79/  79 | samples= 10000 | accuracy= 93.68%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      28,967,063   491,520,000     5.89%
downsample_layers.2.0                                            downsample      65,062,035   245,760,000    26.47%
downsample_layers.3.0                                            downsample      75,047,523   122,880,000    61.07%
stages.0.0.dwconv                                                dwconv              30,179   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1             41,047,393 3,932,160,000     1.04%
stages.0.0.pw2_ttfs                                              pw2            143,565,716   983,040,000    14.60%
stages.0.1.dwconv                                                dwconv               9,562   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1                188,613 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            122,787,666   983,040,000    12.49%
stages.1.0.dwconv                                                dwconv          38,369,059   491,520,000     7.81%
stages.1.0.pw1_ttfs                                              pw1            901,670,051 1,966,080,000    45.86%
stages.1.0.pw2_ttfs                                              pw2            486,832,270   491,520,000    99.05%
stages.1.1.dwconv                                                dwconv          41,590,172   491,520,000     8.46%
stages.1.1.pw1_ttfs                                              pw1            909,639,607 1,966,080,000    46.27%
stages.1.1.pw2_ttfs                                              pw2            473,344,278   491,520,000    96.30%
stages.2.0.dwconv                                                dwconv          70,620,782   245,760,000    28.74%
stages.2.0.pw1_ttfs                                              pw1            464,745,367   983,040,000    47.28%
stages.2.0.pw2_ttfs                                              pw2            236,226,843   245,760,000    96.12%
stages.2.1.dwconv                                                dwconv          80,932,358   245,760,000    32.93%
stages.2.1.pw1_ttfs                                              pw1            493,646,915   983,040,000    50.22%
stages.2.1.pw2_ttfs                                              pw2            238,188,857   245,760,000    96.92%
stages.2.2.dwconv                                                dwconv          81,798,377   245,760,000    33.28%
stages.2.2.pw1_ttfs                                              pw1            535,142,231   983,040,000    54.44%
stages.2.2.pw2_ttfs                                              pw2            241,269,587   245,760,000    98.17%
stages.2.3.dwconv                                                dwconv          85,895,148   245,760,000    34.95%
stages.2.3.pw1_ttfs                                              pw1            510,178,948   983,040,000    51.90%
stages.2.3.pw2_ttfs                                              pw2            237,066,022   245,760,000    96.46%
stages.2.4.dwconv                                                dwconv          90,612,306   245,760,000    36.87%
stages.2.4.pw1_ttfs                                              pw1            468,842,757   983,040,000    47.69%
stages.2.4.pw2_ttfs                                              pw2            237,930,684   245,760,000    96.81%
stages.2.5.dwconv                                                dwconv          95,713,676   245,760,000    38.95%
stages.2.5.pw1_ttfs                                              pw1            463,912,325   983,040,000    47.19%
stages.2.5.pw2_ttfs                                              pw2            235,780,981   245,760,000    95.94%
stages.3.0.dwconv                                                dwconv          36,525,387   122,880,000    29.72%
stages.3.0.pw1_ttfs                                              pw1            245,956,982   491,520,000    50.04%
stages.3.0.pw2_ttfs                                              pw2            108,329,093   122,880,000    88.16%
stages.3.1.dwconv                                                dwconv          47,824,539   122,880,000    38.92%
stages.3.1.pw1_ttfs                                              pw1            242,822,904   491,520,000    49.40%
stages.3.1.pw2_ttfs                                              pw2            105,314,475   122,880,000    85.71%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     31.11%
===================================================================================================================

Classification accuracy: 93.68%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.35% | silent=669,921,545 | total=4,669,440,000
pw1             12 layers | sparsity=  28.26% | silent=5,277,794,093 | total=18,677,760,000
pw2             12 layers | sparsity=  61.39% | silent=2,866,636,472 | total=4,669,440,000
downsample       3 layers | sparsity=  19.66% | silent=169,076,621 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    2.61% | TTFS points= 6 | silent=307,629,129 | total=11,796,480,000
Stage 1:   45.08% | TTFS points= 7 | silent=2,880,412,500 | total=6,389,760,000
Stage 2:   54.26% | TTFS points=19 | silent=4,933,566,199 | total=9,093,120,000
Stage 3:   53.95% | TTFS points= 7 | silent=861,820,903 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.68%
Activation sparsity:  31.11%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_7777\activation_sparsity.md
```
