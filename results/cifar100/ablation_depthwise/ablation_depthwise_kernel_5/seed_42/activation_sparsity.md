# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 76.56%
Batch   20/  79 | samples=  2560 | accuracy= 73.20%
Batch   40/  79 | samples=  5120 | accuracy= 72.83%
Batch   60/  79 | samples=  7680 | accuracy= 73.11%
Batch   79/  79 | samples= 10000 | accuracy= 73.24%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,841,492,490 3,932,160,000    46.83%
stages.0.0.pw2_ttfs                                              pw2            267,999,839   983,040,000    27.26%
stages.0.1.pw1_ttfs                                              pw1            987,827,337 3,932,160,000    25.12%
stages.0.1.pw2_ttfs                                              pw2            202,153,426   983,040,000    20.56%
stages.1.0.pw1_ttfs                                              pw1            885,776,331 1,966,080,000    45.05%
stages.1.0.pw2_ttfs                                              pw2            479,297,162   491,520,000    97.51%
stages.1.1.pw1_ttfs                                              pw1            681,093,857 1,966,080,000    34.64%
stages.1.1.pw2_ttfs                                              pw2            464,450,715   491,520,000    94.49%
stages.2.0.pw1_ttfs                                              pw1            448,566,444   983,040,000    45.63%
stages.2.0.pw2_ttfs                                              pw2            234,183,103   245,760,000    95.29%
stages.2.1.pw1_ttfs                                              pw1            466,176,028   983,040,000    47.42%
stages.2.1.pw2_ttfs                                              pw2            238,296,002   245,760,000    96.96%
stages.2.2.pw1_ttfs                                              pw1            509,604,367   983,040,000    51.84%
stages.2.2.pw2_ttfs                                              pw2            237,760,307   245,760,000    96.74%
stages.2.3.pw1_ttfs                                              pw1            491,728,896   983,040,000    50.02%
stages.2.3.pw2_ttfs                                              pw2            238,628,689   245,760,000    97.10%
stages.2.4.pw1_ttfs                                              pw1            490,584,362   983,040,000    49.90%
stages.2.4.pw2_ttfs                                              pw2            236,730,493   245,760,000    96.33%
stages.2.5.pw1_ttfs                                              pw1            473,931,300   983,040,000    48.21%
stages.2.5.pw2_ttfs                                              pw2            241,107,924   245,760,000    98.11%
stages.3.0.pw1_ttfs                                              pw1            260,228,136   491,520,000    52.94%
stages.3.0.pw2_ttfs                                              pw2            102,436,617   122,880,000    83.36%
stages.3.1.pw1_ttfs                                              pw1            239,643,190   491,520,000    48.76%
stages.3.1.pw2_ttfs                                              pw2             98,167,252   122,880,000    79.89%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     46.33%
===================================================================================================================

Classification accuracy: 73.24%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  41.64% | silent=7,776,652,738 | total=18,677,760,000
pw2             12 layers | sparsity=  65.13% | silent=3,041,211,529 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   33.56% | TTFS points= 4 | silent=3,299,473,092 | total=9,830,400,000
Stage 1:   51.08% | TTFS points= 4 | silent=2,510,618,065 | total=4,915,200,000
Stage 2:   58.42% | TTFS points=12 | silent=4,307,297,915 | total=7,372,800,000
Stage 3:   57.00% | TTFS points= 4 | silent=700,475,195 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.24%
Activation sparsity:  46.33%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_42\activation_sparsity.md
```
