# Activation Sparsity Evaluation

```text
Using checkpoint: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_5435\best_checkpoint.pth

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
Batch    1/ 313 | samples=    32 | accuracy= 65.62%
Batch   20/ 313 | samples=   640 | accuracy= 61.25%
Batch   40/ 313 | samples=  1280 | accuracy= 61.80%
Batch   60/ 313 | samples=  1920 | accuracy= 62.76%
Batch   80/ 313 | samples=  2560 | accuracy= 63.95%
Batch  100/ 313 | samples=  3200 | accuracy= 64.06%
Batch  120/ 313 | samples=  3840 | accuracy= 64.17%
Batch  140/ 313 | samples=  4480 | accuracy= 64.15%
Batch  160/ 313 | samples=  5120 | accuracy= 64.20%
Batch  180/ 313 | samples=  5760 | accuracy= 63.75%
Batch  200/ 313 | samples=  6400 | accuracy= 63.59%
Batch  220/ 313 | samples=  7040 | accuracy= 63.35%
Batch  240/ 313 | samples=  7680 | accuracy= 63.31%
Batch  260/ 313 | samples=  8320 | accuracy= 63.03%
Batch  280/ 313 | samples=  8960 | accuracy= 63.14%
Batch  300/ 313 | samples=  9600 | accuracy= 63.00%
Batch  313/ 313 | samples= 10000 | accuracy= 63.09%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - TINYIMAGENET
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                      015,728,640,000     0.00%
stages.0.0.pw2_ttfs                                              pw2            368,640,000 3,932,160,000     9.38%
stages.0.1.pw1_ttfs                                              pw1          5,056,256,88815,728,640,000    32.15%
stages.0.1.pw2_ttfs                                              pw2            157,045,372 3,932,160,000     3.99%
stages.1.0.pw1_ttfs                                              pw1          1,626,981,237 7,864,320,000    20.69%
stages.1.0.pw2_ttfs                                              pw2          1,517,860,739 1,966,080,000    77.20%
stages.1.1.pw1_ttfs                                              pw1          1,630,482,566 7,864,320,000    20.73%
stages.1.1.pw2_ttfs                                              pw2          1,462,622,780 1,966,080,000    74.39%
stages.2.0.pw1_ttfs                                              pw1            955,984,985 3,932,160,000    24.31%
stages.2.0.pw2_ttfs                                              pw2            904,856,069   983,040,000    92.05%
stages.2.1.pw1_ttfs                                              pw1            598,558,333 3,932,160,000    15.22%
stages.2.1.pw2_ttfs                                              pw2            952,176,558   983,040,000    96.86%
stages.2.2.pw1_ttfs                                              pw1            712,906,953 3,932,160,000    18.13%
stages.2.2.pw2_ttfs                                              pw2            967,438,073   983,040,000    98.41%
stages.2.3.pw1_ttfs                                              pw1            490,673,108 3,932,160,000    12.48%
stages.2.3.pw2_ttfs                                              pw2            934,324,803   983,040,000    95.04%
stages.2.4.pw1_ttfs                                              pw1            649,225,657 3,932,160,000    16.51%
stages.2.4.pw2_ttfs                                              pw2            961,851,406   983,040,000    97.84%
stages.2.5.pw1_ttfs                                              pw1            486,786,046 3,932,160,000    12.38%
stages.2.5.pw2_ttfs                                              pw2            961,100,576   983,040,000    97.77%
stages.3.0.pw1_ttfs                                              pw1            965,250,978 1,966,080,000    49.10%
stages.3.0.pw2_ttfs                                              pw2            444,906,116   491,520,000    90.52%
stages.3.1.pw1_ttfs                                              pw1            875,678,376 1,966,080,000    44.54%
stages.3.1.pw2_ttfs                                              pw2            394,459,278   491,520,000    80.25%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     25.78%
===================================================================================================================

Classification accuracy: 63.09%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  18.80% | silent=14,048,785,127 | total=74,711,040,000
pw2             12 layers | sparsity=  53.69% | silent=10,027,281,770 | total=18,677,760,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   14.20% | TTFS points= 4 | silent=5,581,942,260 | total=39,321,600,000
Stage 1:   31.73% | TTFS points= 4 | silent=6,237,947,322 | total=19,660,800,000
Stage 2:   32.47% | TTFS points=12 | silent=9,575,882,567 | total=29,491,200,000
Stage 3:   54.53% | TTFS points= 4 | silent=2,680,294,748 | total=4,915,200,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              tinyimagenet
Accuracy:             63.09%
Activation sparsity:  25.78%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\tinyimagenet\test_downsample_ttfs_dwconv_dense\seed_5435\activation_sparsity.md
```
