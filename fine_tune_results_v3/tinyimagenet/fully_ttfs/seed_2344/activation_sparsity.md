# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_2344\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 60.94%
Batch   20/  79 | samples=  2560 | accuracy= 60.78%
Batch   40/  79 | samples=  5120 | accuracy= 61.48%
Batch   60/  79 | samples=  7680 | accuracy= 61.28%
Batch   79/  79 | samples= 10000 | accuracy= 61.32%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     102,806,657 1,966,080,000     5.23%
downsample_layers.2.0                                            downsample     310,287,820   983,040,000    31.56%
downsample_layers.3.0                                            downsample     385,665,909   491,520,000    78.46%
stages.0.0.dwconv                                                dwconv          43,562,710 3,932,160,000     1.11%
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            286,720,000 3,932,160,000     7.29%
stages.0.1.dwconv                                                dwconv          27,560,960 3,932,160,000     0.70%
stages.0.1.pw1_ttfs                                              pw1          3,202,248,88615,728,640,000    20.36%
stages.0.1.pw2_ttfs                                              pw2             61,651,706 3,932,160,000     1.57%
stages.1.0.dwconv                                                dwconv         129,840,391 1,966,080,000     6.60%
stages.1.0.pw1_ttfs                                              pw1          1,776,091,428 7,864,320,000    22.58%
stages.1.0.pw2_ttfs                                              pw2          1,854,111,874 1,966,080,000    94.31%
stages.1.1.dwconv                                                dwconv         135,919,532 1,966,080,000     6.91%
stages.1.1.pw1_ttfs                                              pw1          1,454,977,585 7,864,320,000    18.50%
stages.1.1.pw2_ttfs                                              pw2          1,793,043,091 1,966,080,000    91.20%
stages.2.0.dwconv                                                dwconv         205,187,342   983,040,000    20.87%
stages.2.0.pw1_ttfs                                              pw1            501,121,406 3,932,160,000    12.74%
stages.2.0.pw2_ttfs                                              pw2            840,642,302   983,040,000    85.51%
stages.2.1.dwconv                                                dwconv         312,455,405   983,040,000    31.78%
stages.2.1.pw1_ttfs                                              pw1            300,588,047 3,932,160,000     7.64%
stages.2.1.pw2_ttfs                                              pw2            942,508,832   983,040,000    95.88%
stages.2.2.dwconv                                                dwconv         368,719,606   983,040,000    37.51%
stages.2.2.pw1_ttfs                                              pw1            161,833,262 3,932,160,000     4.12%
stages.2.2.pw2_ttfs                                              pw2            942,614,365   983,040,000    95.89%
stages.2.3.dwconv                                                dwconv         378,979,447   983,040,000    38.55%
stages.2.3.pw1_ttfs                                              pw1            147,770,688 3,932,160,000     3.76%
stages.2.3.pw2_ttfs                                              pw2            928,443,030   983,040,000    94.45%
stages.2.4.dwconv                                                dwconv         390,263,585   983,040,000    39.70%
stages.2.4.pw1_ttfs                                              pw1             34,438,822 3,932,160,000     0.88%
stages.2.4.pw2_ttfs                                              pw2            962,365,384   983,040,000    97.90%
stages.2.5.dwconv                                                dwconv         372,353,417   983,040,000    37.88%
stages.2.5.pw1_ttfs                                              pw1            234,805,371 3,932,160,000     5.97%
stages.2.5.pw2_ttfs                                              pw2            931,349,235   983,040,000    94.74%
stages.3.0.dwconv                                                dwconv         153,129,087   491,520,000    31.15%
stages.3.0.pw1_ttfs                                              pw1          1,044,013,684 1,966,080,000    53.10%
stages.3.0.pw2_ttfs                                              pw2            417,064,181   491,520,000    84.85%
stages.3.1.dwconv                                                dwconv         192,733,887   491,520,000    39.21%
stages.3.1.pw1_ttfs                                              pw1            873,111,005 1,966,080,000    44.41%
stages.3.1.pw2_ttfs                                              pw2            407,364,733   491,520,000    82.88%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     20.44%
===================================================================================================================

Classification accuracy: 61.32%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.51% | silent=2,710,705,369 | total=18,677,760,000
pw1             12 layers | sparsity=  13.02% | silent=9,731,000,184 | total=74,711,040,000
pw2             12 layers | sparsity=  55.51% | silent=10,367,878,733 | total=18,677,760,000
downsample       3 layers | sparsity=  23.22% | silent=798,760,386 | total=3,440,640,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    7.68% | TTFS points= 6 | silent=3,621,744,262 | total=47,185,920,000
Stage 1:   28.35% | TTFS points= 7 | silent=7,246,790,558 | total=25,559,040,000
Stage 2:   25.48% | TTFS points=19 | silent=9,266,727,366 | total=36,372,480,000
Stage 3:   54.35% | TTFS points= 7 | silent=3,473,082,486 | total=6,389,760,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             61.32%
Activation sparsity:  20.44%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\tinyimagenet\fully_ttfs\seed_2344\activation_sparsity.md
```
