# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\downsample_dense_dwconv_dense\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar100
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
Batch    1/  79 | samples=   128 | accuracy= 75.00%
Batch   20/  79 | samples=  2560 | accuracy= 73.71%
Batch   40/  79 | samples=  5120 | accuracy= 73.16%
Batch   60/  79 | samples=  7680 | accuracy= 73.15%
Batch   79/  79 | samples= 10000 | accuracy= 73.29%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1             32,953,019 3,932,160,000     0.84%
stages.0.0.pw2_ttfs                                              pw2            187,812,734   983,040,000    19.11%
stages.0.1.pw1_ttfs                                              pw1            102,511,176 3,932,160,000     2.61%
stages.0.1.pw2_ttfs                                              pw2            234,757,693   983,040,000    23.88%
stages.1.0.pw1_ttfs                                              pw1            885,277,138 1,966,080,000    45.03%
stages.1.0.pw2_ttfs                                              pw2            474,901,531   491,520,000    96.62%
stages.1.1.pw1_ttfs                                              pw1            850,511,669 1,966,080,000    43.26%
stages.1.1.pw2_ttfs                                              pw2            455,938,359   491,520,000    92.76%
stages.2.0.pw1_ttfs                                              pw1            456,362,003   983,040,000    46.42%
stages.2.0.pw2_ttfs                                              pw2            238,174,226   245,760,000    96.91%
stages.2.1.pw1_ttfs                                              pw1            456,824,380   983,040,000    46.47%
stages.2.1.pw2_ttfs                                              pw2            236,790,998   245,760,000    96.35%
stages.2.2.pw1_ttfs                                              pw1            461,174,136   983,040,000    46.91%
stages.2.2.pw2_ttfs                                              pw2            242,094,268   245,760,000    98.51%
stages.2.3.pw1_ttfs                                              pw1            469,260,952   983,040,000    47.74%
stages.2.3.pw2_ttfs                                              pw2            242,081,489   245,760,000    98.50%
stages.2.4.pw1_ttfs                                              pw1            468,952,360   983,040,000    47.70%
stages.2.4.pw2_ttfs                                              pw2            230,775,078   245,760,000    93.90%
stages.2.5.pw1_ttfs                                              pw1            472,923,880   983,040,000    48.11%
stages.2.5.pw2_ttfs                                              pw2            242,948,086   245,760,000    98.86%
stages.3.0.pw1_ttfs                                              pw1            263,750,016   491,520,000    53.66%
stages.3.0.pw2_ttfs                                              pw2            106,611,299   122,880,000    86.76%
stages.3.1.pw1_ttfs                                              pw1            246,999,085   491,520,000    50.25%
stages.3.1.pw2_ttfs                                              pw2             92,706,362   122,880,000    75.44%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     34.92%
===================================================================================================================

Classification accuracy: 73.29%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  27.67% | silent=5,167,499,814 | total=18,677,760,000
pw2             12 layers | sparsity=  63.94% | silent=2,985,592,123 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    5.68% | TTFS points= 4 | silent=558,034,622 | total=9,830,400,000
Stage 1:   54.25% | TTFS points= 4 | silent=2,666,628,697 | total=4,915,200,000
Stage 2:   57.22% | TTFS points=12 | silent=4,218,361,856 | total=7,372,800,000
Stage 3:   57.79% | TTFS points= 4 | silent=710,066,762 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.29%
Activation sparsity:  34.92%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\downsample_dense_dwconv_dense\seed_6543\activation_sparsity.md
```
