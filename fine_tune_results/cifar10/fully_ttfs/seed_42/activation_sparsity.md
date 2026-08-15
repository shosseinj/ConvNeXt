# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results\cifar10\fully_ttfs\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 92.19%
Batch   20/  79 | samples=  2560 | accuracy= 93.20%
Batch   40/  79 | samples=  5120 | accuracy= 93.42%
Batch   60/  79 | samples=  7680 | accuracy= 93.66%
Batch   79/  79 | samples= 10000 | accuracy= 93.61%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      27,645,820   491,520,000     5.62%
downsample_layers.2.0                                            downsample      71,979,062   245,760,000    29.29%
downsample_layers.3.0                                            downsample      79,242,225   122,880,000    64.49%
stages.0.0.dwconv                                                dwconv              19,763   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1             14,719,659 3,932,160,000     0.37%
stages.0.0.pw2_ttfs                                              pw2            212,512,666   983,040,000    21.62%
stages.0.1.dwconv                                                dwconv             543,757   983,040,000     0.06%
stages.0.1.pw1_ttfs                                              pw1              6,780,359 3,932,160,000     0.17%
stages.0.1.pw2_ttfs                                              pw2            191,801,153   983,040,000    19.51%
stages.1.0.dwconv                                                dwconv          30,429,985   491,520,000     6.19%
stages.1.0.pw1_ttfs                                              pw1            926,381,749 1,966,080,000    47.12%
stages.1.0.pw2_ttfs                                              pw2            484,467,647   491,520,000    98.57%
stages.1.1.dwconv                                                dwconv          42,093,667   491,520,000     8.56%
stages.1.1.pw1_ttfs                                              pw1            896,432,982 1,966,080,000    45.59%
stages.1.1.pw2_ttfs                                              pw2            481,409,613   491,520,000    97.94%
stages.2.0.dwconv                                                dwconv          68,644,830   245,760,000    27.93%
stages.2.0.pw1_ttfs                                              pw1            464,598,501   983,040,000    47.26%
stages.2.0.pw2_ttfs                                              pw2            236,071,459   245,760,000    96.06%
stages.2.1.dwconv                                                dwconv          80,432,474   245,760,000    32.73%
stages.2.1.pw1_ttfs                                              pw1            507,501,756   983,040,000    51.63%
stages.2.1.pw2_ttfs                                              pw2            237,839,497   245,760,000    96.78%
stages.2.2.dwconv                                                dwconv          84,938,646   245,760,000    34.56%
stages.2.2.pw1_ttfs                                              pw1            480,552,856   983,040,000    48.88%
stages.2.2.pw2_ttfs                                              pw2            241,585,892   245,760,000    98.30%
stages.2.3.dwconv                                                dwconv          89,689,448   245,760,000    36.49%
stages.2.3.pw1_ttfs                                              pw1            477,119,042   983,040,000    48.54%
stages.2.3.pw2_ttfs                                              pw2            242,417,800   245,760,000    98.64%
stages.2.4.dwconv                                                dwconv          89,404,987   245,760,000    36.38%
stages.2.4.pw1_ttfs                                              pw1            517,330,045   983,040,000    52.63%
stages.2.4.pw2_ttfs                                              pw2            238,468,343   245,760,000    97.03%
stages.2.5.dwconv                                                dwconv          91,534,423   245,760,000    37.25%
stages.2.5.pw1_ttfs                                              pw1            454,753,930   983,040,000    46.26%
stages.2.5.pw2_ttfs                                              pw2            232,138,091   245,760,000    94.46%
stages.3.0.dwconv                                                dwconv          33,050,698   122,880,000    26.90%
stages.3.0.pw1_ttfs                                              pw1            241,747,531   491,520,000    49.18%
stages.3.0.pw2_ttfs                                              pw2            109,813,515   122,880,000    89.37%
stages.3.1.dwconv                                                dwconv          47,237,930   122,880,000    38.44%
stages.3.1.pw1_ttfs                                              pw1            267,604,518   491,520,000    54.44%
stages.3.1.pw2_ttfs                                              pw2            106,465,548   122,880,000    86.64%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     31.54%
===================================================================================================================

Classification accuracy: 93.61%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.09% | silent=658,020,608 | total=4,669,440,000
pw1             12 layers | sparsity=  28.14% | silent=5,255,522,928 | total=18,677,760,000
pw2             12 layers | sparsity=  64.57% | silent=3,014,991,224 | total=4,669,440,000
downsample       3 layers | sparsity=  20.79% | silent=178,867,107 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    3.61% | TTFS points= 6 | silent=426,377,357 | total=11,796,480,000
Stage 1:   45.21% | TTFS points= 7 | silent=2,888,861,463 | total=6,389,760,000
Stage 2:   53.96% | TTFS points=19 | silent=4,907,001,082 | total=9,093,120,000
Stage 3:   55.41% | TTFS points= 7 | silent=885,161,965 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.61%
Activation sparsity:  31.54%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results\cifar10\fully_ttfs\seed_42\activation_sparsity.md
```
