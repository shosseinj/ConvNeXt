# Activation Sparsity Evaluation

```text
Using checkpoint: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_2344\best_checkpoint.pth

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
Batch    1/ 313 | samples=    32 | accuracy= 59.38%
Batch   20/ 313 | samples=   640 | accuracy= 57.97%
Batch   40/ 313 | samples=  1280 | accuracy= 60.16%
Batch   60/ 313 | samples=  1920 | accuracy= 62.34%
Batch   80/ 313 | samples=  2560 | accuracy= 63.71%
Batch  100/ 313 | samples=  3200 | accuracy= 64.22%
Batch  120/ 313 | samples=  3840 | accuracy= 64.38%
Batch  140/ 313 | samples=  4480 | accuracy= 64.26%
Batch  160/ 313 | samples=  5120 | accuracy= 64.24%
Batch  180/ 313 | samples=  5760 | accuracy= 63.52%
Batch  200/ 313 | samples=  6400 | accuracy= 63.59%
Batch  220/ 313 | samples=  7040 | accuracy= 63.47%
Batch  240/ 313 | samples=  7680 | accuracy= 63.58%
Batch  260/ 313 | samples=  8320 | accuracy= 63.40%
Batch  280/ 313 | samples=  8960 | accuracy= 63.50%
Batch  300/ 313 | samples=  9600 | accuracy= 63.28%
Batch  313/ 313 | samples= 10000 | accuracy= 63.44%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            286,720,000 3,932,160,000     7.29%
stages.0.1.pw1_ttfs                                              pw1          3,333,333,24315,728,640,000    21.19%
stages.0.1.pw2_ttfs                                              pw2              7,800,109 3,932,160,000     0.20%
stages.1.0.pw1_ttfs                                              pw1          2,099,102,937 7,864,320,000    26.69%
stages.1.0.pw2_ttfs                                              pw2          1,752,931,736 1,966,080,000    89.16%
stages.1.1.pw1_ttfs                                              pw1          1,625,151,778 7,864,320,000    20.66%
stages.1.1.pw2_ttfs                                              pw2          1,641,543,278 1,966,080,000    83.49%
stages.2.0.pw1_ttfs                                              pw1            639,789,118 3,932,160,000    16.27%
stages.2.0.pw2_ttfs                                              pw2            874,304,277   983,040,000    88.94%
stages.2.1.pw1_ttfs                                              pw1            423,675,248 3,932,160,000    10.77%
stages.2.1.pw2_ttfs                                              pw2            952,857,304   983,040,000    96.93%
stages.2.2.pw1_ttfs                                              pw1            356,036,772 3,932,160,000     9.05%
stages.2.2.pw2_ttfs                                              pw2            953,536,818   983,040,000    97.00%
stages.2.3.pw1_ttfs                                              pw1            299,797,254 3,932,160,000     7.62%
stages.2.3.pw2_ttfs                                              pw2            945,899,460   983,040,000    96.22%
stages.2.4.pw1_ttfs                                              pw1            145,731,594 3,932,160,000     3.71%
stages.2.4.pw2_ttfs                                              pw2            971,467,474   983,040,000    98.82%
stages.2.5.pw1_ttfs                                              pw1            370,721,949 3,932,160,000     9.43%
stages.2.5.pw2_ttfs                                              pw2            941,840,140   983,040,000    95.81%
stages.3.0.pw1_ttfs                                              pw1          1,028,263,942 1,966,080,000    52.30%
stages.3.0.pw2_ttfs                                              pw2            434,182,457   491,520,000    88.33%
stages.3.1.pw1_ttfs                                              pw1            817,228,203 1,966,080,000    41.57%
stages.3.1.pw2_ttfs                                              pw2            424,013,387   491,520,000    86.27%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     22.84%
===================================================================================================================

Classification accuracy: 63.44%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  14.91% | silent=11,138,832,038 | total=74,711,040,000
pw2             12 layers | sparsity=  54.54% | silent=10,187,096,440 | total=18,677,760,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    9.23% | TTFS points= 4 | silent=3,627,853,352 | total=39,321,600,000
Stage 1:   36.21% | TTFS points= 4 | silent=7,118,729,729 | total=19,660,800,000
Stage 2:   26.71% | TTFS points=12 | silent=7,875,657,408 | total=29,491,200,000
Stage 3:   55.01% | TTFS points= 4 | silent=2,703,687,989 | total=4,915,200,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             63.44%
Activation sparsity:  22.84%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_2344\activation_sparsity.md
```
