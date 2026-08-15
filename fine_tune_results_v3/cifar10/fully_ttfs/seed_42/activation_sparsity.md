# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_42\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 92.54%
Batch   40/  79 | samples=  5120 | accuracy= 92.85%
Batch   60/  79 | samples=  7680 | accuracy= 93.12%
Batch   79/  79 | samples= 10000 | accuracy= 93.21%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      27,261,993   491,520,000     5.55%
downsample_layers.2.0                                            downsample      73,633,719   245,760,000    29.96%
downsample_layers.3.0                                            downsample      81,434,613   122,880,000    66.27%
stages.0.0.dwconv                                                dwconv              10,000   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1                 15,006 3,932,160,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            211,449,837   983,040,000    21.51%
stages.0.1.dwconv                                                dwconv             752,367   983,040,000     0.08%
stages.0.1.pw1_ttfs                                              pw1             12,977,426 3,932,160,000     0.33%
stages.0.1.pw2_ttfs                                              pw2            202,687,627   983,040,000    20.62%
stages.1.0.dwconv                                                dwconv          28,851,005   491,520,000     5.87%
stages.1.0.pw1_ttfs                                              pw1            921,077,471 1,966,080,000    46.85%
stages.1.0.pw2_ttfs                                              pw2            484,500,268   491,520,000    98.57%
stages.1.1.dwconv                                                dwconv          46,139,052   491,520,000     9.39%
stages.1.1.pw1_ttfs                                              pw1            900,971,535 1,966,080,000    45.83%
stages.1.1.pw2_ttfs                                              pw2            480,140,053   491,520,000    97.68%
stages.2.0.dwconv                                                dwconv          71,244,379   245,760,000    28.99%
stages.2.0.pw1_ttfs                                              pw1            460,934,525   983,040,000    46.89%
stages.2.0.pw2_ttfs                                              pw2            236,499,130   245,760,000    96.23%
stages.2.1.dwconv                                                dwconv          84,266,183   245,760,000    34.29%
stages.2.1.pw1_ttfs                                              pw1            492,507,385   983,040,000    50.10%
stages.2.1.pw2_ttfs                                              pw2            238,216,315   245,760,000    96.93%
stages.2.2.dwconv                                                dwconv          89,777,472   245,760,000    36.53%
stages.2.2.pw1_ttfs                                              pw1            473,684,878   983,040,000    48.19%
stages.2.2.pw2_ttfs                                              pw2            241,575,520   245,760,000    98.30%
stages.2.3.dwconv                                                dwconv          90,341,915   245,760,000    36.76%
stages.2.3.pw1_ttfs                                              pw1            465,563,019   983,040,000    47.36%
stages.2.3.pw2_ttfs                                              pw2            242,597,000   245,760,000    98.71%
stages.2.4.dwconv                                                dwconv          92,443,972   245,760,000    37.62%
stages.2.4.pw1_ttfs                                              pw1            505,304,065   983,040,000    51.40%
stages.2.4.pw2_ttfs                                              pw2            239,026,805   245,760,000    97.26%
stages.2.5.dwconv                                                dwconv          93,446,010   245,760,000    38.02%
stages.2.5.pw1_ttfs                                              pw1            446,395,038   983,040,000    45.41%
stages.2.5.pw2_ttfs                                              pw2            232,285,960   245,760,000    94.52%
stages.3.0.dwconv                                                dwconv          33,070,775   122,880,000    26.91%
stages.3.0.pw1_ttfs                                              pw1            239,069,366   491,520,000    48.64%
stages.3.0.pw2_ttfs                                              pw2            109,951,561   122,880,000    89.48%
stages.3.1.dwconv                                                dwconv          47,944,064   122,880,000    39.02%
stages.3.1.pw1_ttfs                                              pw1            265,970,450   491,520,000    54.11%
stages.3.1.pw2_ttfs                                              pw2            106,566,544   122,880,000    86.72%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     31.41%
===================================================================================================================

Classification accuracy: 93.21%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.53% | silent=678,287,194 | total=4,669,440,000
pw1             12 layers | sparsity=  27.76% | silent=5,184,470,164 | total=18,677,760,000
pw2             12 layers | sparsity=  64.79% | silent=3,025,496,620 | total=4,669,440,000
downsample       3 layers | sparsity=  21.20% | silent=182,330,325 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    3.63% | TTFS points= 6 | silent=427,892,263 | total=11,796,480,000
Stage 1:   45.21% | TTFS points= 7 | silent=2,888,941,377 | total=6,389,760,000
Stage 2:   53.55% | TTFS points=19 | silent=4,869,743,290 | total=9,093,120,000
Stage 3:   55.34% | TTFS points= 7 | silent=884,007,373 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.21%
Activation sparsity:  31.41%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_42\activation_sparsity.md
```
