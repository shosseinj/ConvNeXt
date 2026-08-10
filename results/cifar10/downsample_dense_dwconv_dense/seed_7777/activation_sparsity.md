# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar10\downsample_dense_dwconv_dense\seed_7777\best_checkpoint.pth

Device: cuda
Dataset: cifar10
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
Batch    1/  79 | samples=   128 | accuracy= 93.75%
Batch   20/  79 | samples=  2560 | accuracy= 93.75%
Batch   40/  79 | samples=  5120 | accuracy= 93.65%
Batch   60/  79 | samples=  7680 | accuracy= 93.92%
Batch   79/  79 | samples= 10000 | accuracy= 93.85%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                743,255 3,932,160,000     0.02%
stages.0.0.pw2_ttfs                                              pw2            187,030,886   983,040,000    19.03%
stages.0.1.pw1_ttfs                                              pw1             31,575,645 3,932,160,000     0.80%
stages.0.1.pw2_ttfs                                              pw2            194,524,958   983,040,000    19.79%
stages.1.0.pw1_ttfs                                              pw1            941,297,970 1,966,080,000    47.88%
stages.1.0.pw2_ttfs                                              pw2            483,557,251   491,520,000    98.38%
stages.1.1.pw1_ttfs                                              pw1            912,601,620 1,966,080,000    46.42%
stages.1.1.pw2_ttfs                                              pw2            472,550,893   491,520,000    96.14%
stages.2.0.pw1_ttfs                                              pw1            470,561,455   983,040,000    47.87%
stages.2.0.pw2_ttfs                                              pw2            236,993,489   245,760,000    96.43%
stages.2.1.pw1_ttfs                                              pw1            503,312,123   983,040,000    51.20%
stages.2.1.pw2_ttfs                                              pw2            238,813,030   245,760,000    97.17%
stages.2.2.pw1_ttfs                                              pw1            551,365,442   983,040,000    56.09%
stages.2.2.pw2_ttfs                                              pw2            242,077,406   245,760,000    98.50%
stages.2.3.pw1_ttfs                                              pw1            534,706,778   983,040,000    54.39%
stages.2.3.pw2_ttfs                                              pw2            238,119,872   245,760,000    96.89%
stages.2.4.pw1_ttfs                                              pw1            491,705,522   983,040,000    50.02%
stages.2.4.pw2_ttfs                                              pw2            239,176,141   245,760,000    97.32%
stages.2.5.pw1_ttfs                                              pw1            487,139,404   983,040,000    49.55%
stages.2.5.pw2_ttfs                                              pw2            237,025,800   245,760,000    96.45%
stages.3.0.pw1_ttfs                                              pw1            250,829,311   491,520,000    51.03%
stages.3.0.pw2_ttfs                                              pw2            109,414,240   122,880,000    89.04%
stages.3.1.pw1_ttfs                                              pw1            249,143,120   491,520,000    50.69%
stages.3.1.pw2_ttfs                                              pw2            107,258,146   122,880,000    87.29%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.03%
===================================================================================================================

Classification accuracy: 93.85%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  29.05% | silent=5,424,981,645 | total=18,677,760,000
pw2             12 layers | sparsity=  63.96% | silent=2,986,542,112 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    4.21% | TTFS points= 4 | silent=413,874,744 | total=9,830,400,000
Stage 1:   57.17% | TTFS points= 4 | silent=2,810,007,734 | total=4,915,200,000
Stage 2:   60.64% | TTFS points=12 | silent=4,470,996,462 | total=7,372,800,000
Stage 3:   58.32% | TTFS points= 4 | silent=716,644,817 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.85%
Activation sparsity:  36.03%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar10\downsample_dense_dwconv_dense\seed_7777\activation_sparsity.md
```
