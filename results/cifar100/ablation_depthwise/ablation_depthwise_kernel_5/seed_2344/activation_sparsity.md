# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_2344\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 73.05%
Batch   40/  79 | samples=  5120 | accuracy= 72.89%
Batch   60/  79 | samples=  7680 | accuracy= 72.68%
Batch   79/  79 | samples= 10000 | accuracy= 72.91%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,796,616,884 3,932,160,000    45.69%
stages.0.0.pw2_ttfs                                              pw2            262,405,188   983,040,000    26.69%
stages.0.1.pw1_ttfs                                              pw1          1,577,093,734 3,932,160,000    40.11%
stages.0.1.pw2_ttfs                                              pw2            257,226,621   983,040,000    26.17%
stages.1.0.pw1_ttfs                                              pw1            731,907,034 1,966,080,000    37.23%
stages.1.0.pw2_ttfs                                              pw2            468,314,998   491,520,000    95.28%
stages.1.1.pw1_ttfs                                              pw1            655,400,923 1,966,080,000    33.34%
stages.1.1.pw2_ttfs                                              pw2            474,469,287   491,520,000    96.53%
stages.2.0.pw1_ttfs                                              pw1            458,098,458   983,040,000    46.60%
stages.2.0.pw2_ttfs                                              pw2            233,947,455   245,760,000    95.19%
stages.2.1.pw1_ttfs                                              pw1            508,565,852   983,040,000    51.73%
stages.2.1.pw2_ttfs                                              pw2            238,507,551   245,760,000    97.05%
stages.2.2.pw1_ttfs                                              pw1            494,397,135   983,040,000    50.29%
stages.2.2.pw2_ttfs                                              pw2            232,659,785   245,760,000    94.67%
stages.2.3.pw1_ttfs                                              pw1            497,837,773   983,040,000    50.64%
stages.2.3.pw2_ttfs                                              pw2            237,941,033   245,760,000    96.82%
stages.2.4.pw1_ttfs                                              pw1            513,348,048   983,040,000    52.22%
stages.2.4.pw2_ttfs                                              pw2            238,781,879   245,760,000    97.16%
stages.2.5.pw1_ttfs                                              pw1            470,078,341   983,040,000    47.82%
stages.2.5.pw2_ttfs                                              pw2            236,302,346   245,760,000    96.15%
stages.3.0.pw1_ttfs                                              pw1            255,129,277   491,520,000    51.91%
stages.3.0.pw2_ttfs                                              pw2            107,937,130   122,880,000    87.84%
stages.3.1.pw1_ttfs                                              pw1            248,571,929   491,520,000    50.57%
stages.3.1.pw2_ttfs                                              pw2             92,157,251   122,880,000    75.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     48.35%
===================================================================================================================

Classification accuracy: 72.91%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  43.94% | silent=8,207,045,388 | total=18,677,760,000
pw2             12 layers | sparsity=  65.97% | silent=3,080,650,524 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   39.61% | TTFS points= 4 | silent=3,893,342,427 | total=9,830,400,000
Stage 1:   47.41% | TTFS points= 4 | silent=2,330,092,242 | total=4,915,200,000
Stage 2:   59.14% | TTFS points=12 | silent=4,360,465,656 | total=7,372,800,000
Stage 3:   57.28% | TTFS points= 4 | silent=703,795,587 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.91%
Activation sparsity:  48.35%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_5\seed_2344\activation_sparsity.md
```
