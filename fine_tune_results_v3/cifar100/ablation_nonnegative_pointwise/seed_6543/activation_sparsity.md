# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_6543\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 19.53%
Batch   20/  79 | samples=  2560 | accuracy= 20.82%
Batch   40/  79 | samples=  5120 | accuracy= 20.74%
Batch   60/  79 | samples=  7680 | accuracy= 20.40%
Batch   79/  79 | samples= 10000 | accuracy= 20.40%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      34,304,328   491,520,000     6.98%
downsample_layers.2.0                                            downsample      56,472,786   245,760,000    22.98%
downsample_layers.3.0                                            downsample     100,006,009   122,880,000    81.39%
stages.0.0.dwconv                                                dwconv              17,736   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1                  1,298 3,932,160,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv             310,028   983,040,000     0.03%
stages.0.1.pw1_ttfs                                              pw1                      0 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          30,700,337   491,520,000     6.25%
stages.1.0.pw1_ttfs                                              pw1            696,076,850 1,966,080,000    35.40%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          82,475,428   491,520,000    16.78%
stages.1.1.pw1_ttfs                                              pw1            253,094,156 1,966,080,000    12.87%
stages.1.1.pw2_ttfs                                              pw2            489,399,892   491,520,000    99.57%
stages.2.0.dwconv                                                dwconv          68,963,847   245,760,000    28.06%
stages.2.0.pw1_ttfs                                              pw1            302,547,995   983,040,000    30.78%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          80,692,133   245,760,000    32.83%
stages.2.1.pw1_ttfs                                              pw1            396,425,065   983,040,000    40.33%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          81,823,258   245,760,000    33.29%
stages.2.2.pw1_ttfs                                              pw1            357,150,870   983,040,000    36.33%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          94,698,685   245,760,000    38.53%
stages.2.3.pw1_ttfs                                              pw1            361,717,486   983,040,000    36.80%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          77,132,800   245,760,000    31.39%
stages.2.4.pw1_ttfs                                              pw1            529,369,167   983,040,000    53.85%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          94,474,872   245,760,000    38.44%
stages.2.5.pw1_ttfs                                              pw1            323,474,880   983,040,000    32.91%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          20,894,347   122,880,000    17.00%
stages.3.0.pw1_ttfs                                              pw1            233,049,309   491,520,000    47.41%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          52,071,105   122,880,000    42.38%
stages.3.1.pw1_ttfs                                              pw1            251,411,956   491,520,000    51.15%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     32.02%
===================================================================================================================

Classification accuracy: 20.40%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.65% | silent=684,254,576 | total=4,669,440,000
pw1             12 layers | sparsity=  19.83% | silent=3,704,319,032 | total=18,677,760,000
pw2             12 layers | sparsity=  99.95% | silent=4,667,319,892 | total=4,669,440,000
downsample       3 layers | sparsity=  22.18% | silent=190,783,123 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.67% | TTFS points= 6 | silent=1,966,409,062 | total=11,796,480,000
Stage 1:   32.51% | TTFS points= 7 | silent=2,077,570,991 | total=6,389,760,000
Stage 2:   47.28% | TTFS points=19 | silent=4,299,503,844 | total=9,093,120,000
Stage 3:   56.54% | TTFS points= 7 | silent=903,192,726 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             20.40%
Activation sparsity:  32.02%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_6543\activation_sparsity.md
```
