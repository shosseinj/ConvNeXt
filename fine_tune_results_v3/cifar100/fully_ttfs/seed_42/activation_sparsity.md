# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar100
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
Batch    1/  79 | samples=   128 | accuracy= 73.44%
Batch   20/  79 | samples=  2560 | accuracy= 72.97%
Batch   40/  79 | samples=  5120 | accuracy= 71.88%
Batch   60/  79 | samples=  7680 | accuracy= 72.07%
Batch   79/  79 | samples= 10000 | accuracy= 72.44%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      17,034,843   491,520,000     3.47%
downsample_layers.2.0                                            downsample      62,420,493   245,760,000    25.40%
downsample_layers.3.0                                            downsample      76,440,905   122,880,000    62.21%
stages.0.0.dwconv                                                dwconv          18,848,549   983,040,000     1.92%
stages.0.0.pw1_ttfs                                              pw1          1,667,118,150 3,932,160,000    42.40%
stages.0.0.pw2_ttfs                                              pw2            238,316,028   983,040,000    24.24%
stages.0.1.dwconv                                                dwconv             317,228   983,040,000     0.03%
stages.0.1.pw1_ttfs                                              pw1             15,934,243 3,932,160,000     0.41%
stages.0.1.pw2_ttfs                                              pw2            235,518,133   983,040,000    23.96%
stages.1.0.dwconv                                                dwconv          34,281,105   491,520,000     6.97%
stages.1.0.pw1_ttfs                                              pw1            921,723,772 1,966,080,000    46.88%
stages.1.0.pw2_ttfs                                              pw2            483,835,995   491,520,000    98.44%
stages.1.1.dwconv                                                dwconv          64,883,464   491,520,000    13.20%
stages.1.1.pw1_ttfs                                              pw1            807,957,677 1,966,080,000    41.09%
stages.1.1.pw2_ttfs                                              pw2            471,616,031   491,520,000    95.95%
stages.2.0.dwconv                                                dwconv          59,609,476   245,760,000    24.26%
stages.2.0.pw1_ttfs                                              pw1            382,596,700   983,040,000    38.92%
stages.2.0.pw2_ttfs                                              pw2            225,144,324   245,760,000    91.61%
stages.2.1.dwconv                                                dwconv          85,118,851   245,760,000    34.63%
stages.2.1.pw1_ttfs                                              pw1            442,631,830   983,040,000    45.03%
stages.2.1.pw2_ttfs                                              pw2            238,341,506   245,760,000    96.98%
stages.2.2.dwconv                                                dwconv          88,078,979   245,760,000    35.84%
stages.2.2.pw1_ttfs                                              pw1            451,074,779   983,040,000    45.89%
stages.2.2.pw2_ttfs                                              pw2            240,188,915   245,760,000    97.73%
stages.2.3.dwconv                                                dwconv          86,934,785   245,760,000    35.37%
stages.2.3.pw1_ttfs                                              pw1            476,812,711   983,040,000    48.50%
stages.2.3.pw2_ttfs                                              pw2            240,874,450   245,760,000    98.01%
stages.2.4.dwconv                                                dwconv          90,514,446   245,760,000    36.83%
stages.2.4.pw1_ttfs                                              pw1            513,055,857   983,040,000    52.19%
stages.2.4.pw2_ttfs                                              pw2            239,428,312   245,760,000    97.42%
stages.2.5.dwconv                                                dwconv          86,492,832   245,760,000    35.19%
stages.2.5.pw1_ttfs                                              pw1            474,338,122   983,040,000    48.25%
stages.2.5.pw2_ttfs                                              pw2            237,117,932   245,760,000    96.48%
stages.3.0.dwconv                                                dwconv          33,755,726   122,880,000    27.47%
stages.3.0.pw1_ttfs                                              pw1            265,382,758   491,520,000    53.99%
stages.3.0.pw2_ttfs                                              pw2            102,072,684   122,880,000    83.07%
stages.3.1.dwconv                                                dwconv          44,652,350   122,880,000    36.34%
stages.3.1.pw1_ttfs                                              pw1            255,937,515   491,520,000    52.07%
stages.3.1.pw2_ttfs                                              pw2             92,432,380   122,880,000    75.22%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.60%
===================================================================================================================

Classification accuracy: 72.44%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.85% | silent=693,487,791 | total=4,669,440,000
pw1             12 layers | sparsity=  35.74% | silent=6,674,564,114 | total=18,677,760,000
pw2             12 layers | sparsity=  65.21% | silent=3,044,886,690 | total=4,669,440,000
downsample       3 layers | sparsity=  18.12% | silent=155,896,241 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   18.45% | TTFS points= 6 | silent=2,176,052,331 | total=11,796,480,000
Stage 1:   43.84% | TTFS points= 7 | silent=2,801,332,887 | total=6,389,760,000
Stage 2:   51.92% | TTFS points=19 | silent=4,720,775,300 | total=9,093,120,000
Stage 3:   54.50% | TTFS points= 7 | silent=870,674,318 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.44%
Activation sparsity:  36.60%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_42\activation_sparsity.md
```
