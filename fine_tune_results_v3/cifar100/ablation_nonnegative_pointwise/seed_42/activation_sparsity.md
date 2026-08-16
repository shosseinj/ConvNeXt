# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 25.00%
Batch   20/  79 | samples=  2560 | accuracy= 21.95%
Batch   40/  79 | samples=  5120 | accuracy= 21.89%
Batch   60/  79 | samples=  7680 | accuracy= 21.76%
Batch   79/  79 | samples= 10000 | accuracy= 21.66%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      16,184,910   491,520,000     3.29%
downsample_layers.2.0                                            downsample      63,313,056   245,760,000    25.76%
downsample_layers.3.0                                            downsample      94,958,148   122,880,000    77.28%
stages.0.0.dwconv                                                dwconv          18,345,928   983,040,000     1.87%
stages.0.0.pw1_ttfs                                              pw1          1,392,003,738 3,932,160,000    35.40%
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.0.1.dwconv                                                dwconv              81,567   983,040,000     0.01%
stages.0.1.pw1_ttfs                                              pw1                 73,122 3,932,160,000     0.00%
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%
stages.1.0.dwconv                                                dwconv          36,952,813   491,520,000     7.52%
stages.1.0.pw1_ttfs                                              pw1            639,374,474 1,966,080,000    32.52%
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%
stages.1.1.dwconv                                                dwconv          91,093,081   491,520,000    18.53%
stages.1.1.pw1_ttfs                                              pw1            244,804,401 1,966,080,000    12.45%
stages.1.1.pw2_ttfs                                              pw2            490,120,018   491,520,000    99.72%
stages.2.0.dwconv                                                dwconv          67,836,132   245,760,000    27.60%
stages.2.0.pw1_ttfs                                              pw1            325,371,122   983,040,000    33.10%
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.1.dwconv                                                dwconv          90,036,275   245,760,000    36.64%
stages.2.1.pw1_ttfs                                              pw1            355,584,663   983,040,000    36.17%
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.2.dwconv                                                dwconv          91,942,752   245,760,000    37.41%
stages.2.2.pw1_ttfs                                              pw1            389,502,791   983,040,000    39.62%
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.3.dwconv                                                dwconv          86,837,048   245,760,000    35.33%
stages.2.3.pw1_ttfs                                              pw1            399,241,754   983,040,000    40.61%
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.4.dwconv                                                dwconv          90,610,957   245,760,000    36.87%
stages.2.4.pw1_ttfs                                              pw1            486,996,315   983,040,000    49.54%
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.2.5.dwconv                                                dwconv          87,593,475   245,760,000    35.64%
stages.2.5.pw1_ttfs                                              pw1            398,219,435   983,040,000    40.51%
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%
stages.3.0.dwconv                                                dwconv          19,622,052   122,880,000    15.97%
stages.3.0.pw1_ttfs                                              pw1            224,035,239   491,520,000    45.58%
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
stages.3.1.dwconv                                                dwconv          48,821,786   122,880,000    39.73%
stages.3.1.pw1_ttfs                                              pw1            234,064,353   491,520,000    47.62%
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.92%
===================================================================================================================

Classification accuracy: 21.66%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  15.63% | silent=729,773,866 | total=4,669,440,000
pw1             12 layers | sparsity=  27.25% | silent=5,089,271,407 | total=18,677,760,000
pw2             12 layers | sparsity=  99.97% | silent=4,668,040,018 | total=4,669,440,000
downsample       3 layers | sparsity=  20.28% | silent=174,456,114 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   28.62% | TTFS points= 6 | silent=3,376,584,355 | total=11,796,480,000
Stage 1:   31.46% | TTFS points= 7 | silent=2,010,049,697 | total=6,389,760,000
Stage 2:   48.47% | TTFS points=19 | silent=4,407,645,775 | total=9,093,120,000
Stage 3:   54.29% | TTFS points= 7 | silent=867,261,578 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             21.66%
Activation sparsity:  36.92%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_42\activation_sparsity.md
```
