# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_5435\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 74.22%
Batch   20/  79 | samples=  2560 | accuracy= 72.15%
Batch   40/  79 | samples=  5120 | accuracy= 71.66%
Batch   60/  79 | samples=  7680 | accuracy= 71.80%
Batch   79/  79 | samples= 10000 | accuracy= 72.29%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1            165,649,217 3,932,160,000     4.21%
stages.0.0.pw2_ttfs                                              pw2            216,996,837   983,040,000    22.07%
stages.0.1.pw1_ttfs                                              pw1            204,287,532 3,932,160,000     5.20%
stages.0.1.pw2_ttfs                                              pw2            306,792,015   983,040,000    31.21%
stages.1.0.pw1_ttfs                                              pw1            888,743,052 1,966,080,000    45.20%
stages.1.0.pw2_ttfs                                              pw2            471,709,841   491,520,000    95.97%
stages.1.1.pw1_ttfs                                              pw1            816,883,770 1,966,080,000    41.55%
stages.1.1.pw2_ttfs                                              pw2            473,582,726   491,520,000    96.35%
stages.2.0.pw1_ttfs                                              pw1            456,648,565   983,040,000    46.45%
stages.2.0.pw2_ttfs                                              pw2            229,418,336   245,760,000    93.35%
stages.2.1.pw1_ttfs                                              pw1            493,323,245   983,040,000    50.18%
stages.2.1.pw2_ttfs                                              pw2            240,437,269   245,760,000    97.83%
stages.2.2.pw1_ttfs                                              pw1            499,795,152   983,040,000    50.84%
stages.2.2.pw2_ttfs                                              pw2            242,529,086   245,760,000    98.69%
stages.2.3.pw1_ttfs                                              pw1            513,933,159   983,040,000    52.28%
stages.2.3.pw2_ttfs                                              pw2            243,790,010   245,760,000    99.20%
stages.2.4.pw1_ttfs                                              pw1            486,695,256   983,040,000    49.51%
stages.2.4.pw2_ttfs                                              pw2            240,239,937   245,760,000    97.75%
stages.2.5.pw1_ttfs                                              pw1            488,199,922   983,040,000    49.66%
stages.2.5.pw2_ttfs                                              pw2            241,707,822   245,760,000    98.35%
stages.3.0.pw1_ttfs                                              pw1            254,197,787   491,520,000    51.72%
stages.3.0.pw2_ttfs                                              pw2            105,382,041   122,880,000    85.76%
stages.3.1.pw1_ttfs                                              pw1            233,588,727   491,520,000    47.52%
stages.3.1.pw2_ttfs                                              pw2             94,015,469   122,880,000    76.51%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.87%
===================================================================================================================

Classification accuracy: 72.29%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  29.46% | silent=5,501,945,384 | total=18,677,760,000
pw2             12 layers | sparsity=  66.53% | silent=3,106,601,389 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    9.09% | TTFS points= 4 | silent=893,725,601 | total=9,830,400,000
Stage 1:   53.93% | TTFS points= 4 | silent=2,650,919,389 | total=4,915,200,000
Stage 2:   59.36% | TTFS points=12 | silent=4,376,717,759 | total=7,372,800,000
Stage 3:   55.92% | TTFS points= 4 | silent=687,184,024 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.29%
Activation sparsity:  36.87%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_5435\activation_sparsity.md
```
