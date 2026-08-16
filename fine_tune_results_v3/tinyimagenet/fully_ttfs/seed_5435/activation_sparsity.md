# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_5435\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 62.50%
Batch   20/  79 | samples=  2560 | accuracy= 63.28%
Batch   40/  79 | samples=  5120 | accuracy= 62.95%
Batch   60/  79 | samples=  7680 | accuracy= 62.24%
Batch   79/  79 | samples= 10000 | accuracy= 62.29%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     179,604,709 1,966,080,000     9.14%
downsample_layers.2.0                                            downsample     339,897,860   983,040,000    34.58%
downsample_layers.3.0                                            downsample     395,297,511   491,520,000    80.42%
stages.0.0.dwconv                                                dwconv          38,555,650 3,932,160,000     0.98%
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            491,520,000 3,932,160,000    12.50%
stages.0.1.dwconv                                                dwconv          39,424,323 3,932,160,000     1.00%
stages.0.1.pw1_ttfs                                              pw1          5,199,684,75615,728,640,000    33.06%
stages.0.1.pw2_ttfs                                              pw2            323,782,382 3,932,160,000     8.23%
stages.1.0.dwconv                                                dwconv         254,537,376 1,966,080,000    12.95%
stages.1.0.pw1_ttfs                                              pw1          1,270,260,938 7,864,320,000    16.15%
stages.1.0.pw2_ttfs                                              pw2          1,749,174,970 1,966,080,000    88.97%
stages.1.1.dwconv                                                dwconv         125,398,167 1,966,080,000     6.38%
stages.1.1.pw1_ttfs                                              pw1          1,392,204,009 7,864,320,000    17.70%
stages.1.1.pw2_ttfs                                              pw2          1,622,800,546 1,966,080,000    82.54%
stages.2.0.dwconv                                                dwconv         191,747,172   983,040,000    19.51%
stages.2.0.pw1_ttfs                                              pw1            783,915,297 3,932,160,000    19.94%
stages.2.0.pw2_ttfs                                              pw2            877,992,194   983,040,000    89.31%
stages.2.1.dwconv                                                dwconv         343,778,359   983,040,000    34.97%
stages.2.1.pw1_ttfs                                              pw1            430,484,617 3,932,160,000    10.95%
stages.2.1.pw2_ttfs                                              pw2            941,984,319   983,040,000    95.82%
stages.2.2.dwconv                                                dwconv         341,307,238   983,040,000    34.72%
stages.2.2.pw1_ttfs                                              pw1            381,680,586 3,932,160,000     9.71%
stages.2.2.pw2_ttfs                                              pw2            961,441,594   983,040,000    97.80%
stages.2.3.dwconv                                                dwconv         324,977,626   983,040,000    33.06%
stages.2.3.pw1_ttfs                                              pw1            290,661,649 3,932,160,000     7.39%
stages.2.3.pw2_ttfs                                              pw2            919,427,456   983,040,000    93.53%
stages.2.4.dwconv                                                dwconv         364,189,231   983,040,000    37.05%
stages.2.4.pw1_ttfs                                              pw1            384,329,491 3,932,160,000     9.77%
stages.2.4.pw2_ttfs                                              pw2            954,604,584   983,040,000    97.11%
stages.2.5.dwconv                                                dwconv         402,192,830   983,040,000    40.91%
stages.2.5.pw1_ttfs                                              pw1            253,727,861 3,932,160,000     6.45%
stages.2.5.pw2_ttfs                                              pw2            953,217,726   983,040,000    96.97%
stages.3.0.dwconv                                                dwconv          95,701,573   491,520,000    19.47%
stages.3.0.pw1_ttfs                                              pw1          1,023,745,336 1,966,080,000    52.07%
stages.3.0.pw2_ttfs                                              pw2            430,317,276   491,520,000    87.55%
stages.3.1.dwconv                                                dwconv         197,067,385   491,520,000    40.09%
stages.3.1.pw1_ttfs                                              pw1            936,062,916 1,966,080,000    47.61%
stages.3.1.pw2_ttfs                                              pw2            381,661,904   491,520,000    77.65%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     23.02%
===================================================================================================================

Classification accuracy: 62.29%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.56% | silent=2,718,876,930 | total=18,677,760,000
pw1             12 layers | sparsity=  16.53% | silent=12,346,757,456 | total=74,711,040,000
pw2             12 layers | sparsity=  56.79% | silent=10,607,924,951 | total=18,677,760,000
downsample       3 layers | sparsity=  26.59% | silent=914,800,080 | total=3,440,640,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   12.91% | TTFS points= 6 | silent=6,092,967,111 | total=47,185,920,000
Stage 1:   25.80% | TTFS points= 7 | silent=6,593,980,715 | total=25,559,040,000
Stage 2:   28.71% | TTFS points=19 | silent=10,441,557,690 | total=36,372,480,000
Stage 3:   54.15% | TTFS points= 7 | silent=3,459,853,901 | total=6,389,760,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             62.29%
Activation sparsity:  23.02%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_5435\activation_sparsity.md
```
