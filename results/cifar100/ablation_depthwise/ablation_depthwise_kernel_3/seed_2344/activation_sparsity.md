# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_3\seed_2344\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 75.78%
Batch   20/  79 | samples=  2560 | accuracy= 74.77%
Batch   40/  79 | samples=  5120 | accuracy= 74.02%
Batch   60/  79 | samples=  7680 | accuracy= 74.31%
Batch   79/  79 | samples= 10000 | accuracy= 74.51%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,686,989,166 3,932,160,000    42.90%
stages.0.0.pw2_ttfs                                              pw2            273,473,284   983,040,000    27.82%
stages.0.1.pw1_ttfs                                              pw1          1,384,776,010 3,932,160,000    35.22%
stages.0.1.pw2_ttfs                                              pw2            253,893,295   983,040,000    25.83%
stages.1.0.pw1_ttfs                                              pw1            832,040,331 1,966,080,000    42.32%
stages.1.0.pw2_ttfs                                              pw2            473,241,070   491,520,000    96.28%
stages.1.1.pw1_ttfs                                              pw1            843,804,883 1,966,080,000    42.92%
stages.1.1.pw2_ttfs                                              pw2            478,983,543   491,520,000    97.45%
stages.2.0.pw1_ttfs                                              pw1            414,093,546   983,040,000    42.12%
stages.2.0.pw2_ttfs                                              pw2            227,733,656   245,760,000    92.67%
stages.2.1.pw1_ttfs                                              pw1            512,109,168   983,040,000    52.09%
stages.2.1.pw2_ttfs                                              pw2            237,194,219   245,760,000    96.51%
stages.2.2.pw1_ttfs                                              pw1            530,546,990   983,040,000    53.97%
stages.2.2.pw2_ttfs                                              pw2            242,276,378   245,760,000    98.58%
stages.2.3.pw1_ttfs                                              pw1            474,933,214   983,040,000    48.31%
stages.2.3.pw2_ttfs                                              pw2            237,684,169   245,760,000    96.71%
stages.2.4.pw1_ttfs                                              pw1            491,688,019   983,040,000    50.02%
stages.2.4.pw2_ttfs                                              pw2            240,543,111   245,760,000    97.88%
stages.2.5.pw1_ttfs                                              pw1            485,859,290   983,040,000    49.42%
stages.2.5.pw2_ttfs                                              pw2            238,634,712   245,760,000    97.10%
stages.3.0.pw1_ttfs                                              pw1            256,503,590   491,520,000    52.19%
stages.3.0.pw2_ttfs                                              pw2            104,521,870   122,880,000    85.06%
stages.3.1.pw1_ttfs                                              pw1            258,819,855   491,520,000    52.66%
stages.3.1.pw2_ttfs                                              pw2             97,999,665   122,880,000    79.75%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     48.31%
===================================================================================================================

Classification accuracy: 74.51%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  43.75% | silent=8,172,164,062 | total=18,677,760,000
pw2             12 layers | sparsity=  66.52% | silent=3,106,178,972 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   36.61% | TTFS points= 4 | silent=3,599,131,755 | total=9,830,400,000
Stage 1:   53.47% | TTFS points= 4 | silent=2,628,069,827 | total=4,915,200,000
Stage 2:   58.77% | TTFS points=12 | silent=4,333,296,472 | total=7,372,800,000
Stage 3:   58.42% | TTFS points= 4 | silent=717,844,980 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.51%
Activation sparsity:  48.31%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_3\seed_2344\activation_sparsity.md
```
