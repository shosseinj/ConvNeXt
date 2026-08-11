# Activation Sparsity Evaluation

```text
Using checkpoint: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_42\best_checkpoint.pth

Device: cuda
Dataset: tinyimagenet
Evaluation samples: 10000

Detected depthwise convolution mode: dense (metadata)
Detected downsampling convolution mode: dense (metadata)

Missing keys:    0
Unexpected keys: 0

==========================================================================================
MODEL SPARSITY STRUCTURE
==========================================================================================
Spiking blocks:              12
ContinuousTTFSConv2d:        0
PW1 TTFS outputs:            12
PW2 TTFS outputs:            12
Expected total TTFS points:  24
==========================================================================================

TTFS Conv modules:
Batch    1/ 313 | samples=    32 | accuracy= 56.25%
Batch   20/ 313 | samples=   640 | accuracy= 61.88%
Batch   40/ 313 | samples=  1280 | accuracy= 61.02%
Batch   60/ 313 | samples=  1920 | accuracy= 61.93%
Batch   80/ 313 | samples=  2560 | accuracy= 62.73%
Batch  100/ 313 | samples=  3200 | accuracy= 62.81%
Batch  120/ 313 | samples=  3840 | accuracy= 63.05%
Batch  140/ 313 | samples=  4480 | accuracy= 63.30%
Batch  160/ 313 | samples=  5120 | accuracy= 63.38%
Batch  180/ 313 | samples=  5760 | accuracy= 63.18%
Batch  200/ 313 | samples=  6400 | accuracy= 63.34%
Batch  220/ 313 | samples=  7040 | accuracy= 63.07%
Batch  240/ 313 | samples=  7680 | accuracy= 62.97%
Batch  260/ 313 | samples=  8320 | accuracy= 62.73%
Batch  280/ 313 | samples=  8960 | accuracy= 62.86%
Batch  300/ 313 | samples=  9600 | accuracy= 62.77%
Batch  313/ 313 | samples= 10000 | accuracy= 62.80%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            286,720,000 3,932,160,000     7.29%
stages.0.1.pw1_ttfs                                              pw1          5,233,016,89215,728,640,000    33.27%
stages.0.1.pw2_ttfs                                              pw2            202,996,451 3,932,160,000     5.16%
stages.1.0.pw1_ttfs                                              pw1          1,899,236,589 7,864,320,000    24.15%
stages.1.0.pw2_ttfs                                              pw2          1,456,630,122 1,966,080,000    74.09%
stages.1.1.pw1_ttfs                                              pw1            787,080,079 7,864,320,000    10.01%
stages.1.1.pw2_ttfs                                              pw2          1,498,509,560 1,966,080,000    76.22%
stages.2.0.pw1_ttfs                                              pw1            591,874,289 3,932,160,000    15.05%
stages.2.0.pw2_ttfs                                              pw2            943,469,030   983,040,000    95.97%
stages.2.1.pw1_ttfs                                              pw1            677,647,993 3,932,160,000    17.23%
stages.2.1.pw2_ttfs                                              pw2            867,450,358   983,040,000    88.24%
stages.2.2.pw1_ttfs                                              pw1            276,928,865 3,932,160,000     7.04%
stages.2.2.pw2_ttfs                                              pw2            947,905,110   983,040,000    96.43%
stages.2.3.pw1_ttfs                                              pw1            242,281,775 3,932,160,000     6.16%
stages.2.3.pw2_ttfs                                              pw2            959,952,568   983,040,000    97.65%
stages.2.4.pw1_ttfs                                              pw1            455,333,640 3,932,160,000    11.58%
stages.2.4.pw2_ttfs                                              pw2            960,940,249   983,040,000    97.75%
stages.2.5.pw1_ttfs                                              pw1            327,541,081 3,932,160,000     8.33%
stages.2.5.pw2_ttfs                                              pw2            950,270,082   983,040,000    96.67%
stages.3.0.pw1_ttfs                                              pw1            861,193,161 1,966,080,000    43.80%
stages.3.0.pw2_ttfs                                              pw2            449,775,988   491,520,000    91.51%
stages.3.1.pw1_ttfs                                              pw1            911,040,275 1,966,080,000    46.34%
stages.3.1.pw2_ttfs                                              pw2            385,312,024   491,520,000    78.39%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     23.74%
===================================================================================================================

Classification accuracy: 62.80%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  16.41% | silent=12,263,174,639 | total=74,711,040,000
pw2             12 layers | sparsity=  53.06% | silent=9,909,931,542 | total=18,677,760,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   14.55% | TTFS points= 4 | silent=5,722,733,343 | total=39,321,600,000
Stage 1:   28.69% | TTFS points= 4 | silent=5,641,456,350 | total=19,660,800,000
Stage 2:   27.81% | TTFS points=12 | silent=8,201,595,040 | total=29,491,200,000
Stage 3:   53.05% | TTFS points= 4 | silent=2,607,321,448 | total=4,915,200,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             62.80%
Activation sparsity:  23.74%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_42\activation_sparsity.md
```
