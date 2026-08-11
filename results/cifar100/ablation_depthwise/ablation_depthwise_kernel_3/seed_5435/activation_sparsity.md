# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_3\seed_5435\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 73.63%
Batch   40/  79 | samples=  5120 | accuracy= 73.32%
Batch   60/  79 | samples=  7680 | accuracy= 73.65%
Batch   79/  79 | samples= 10000 | accuracy= 73.86%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,887,585,903 3,932,160,000    48.00%
stages.0.0.pw2_ttfs                                              pw2            304,195,755   983,040,000    30.94%
stages.0.1.pw1_ttfs                                              pw1             39,336,943 3,932,160,000     1.00%
stages.0.1.pw2_ttfs                                              pw2            268,534,082   983,040,000    27.32%
stages.1.0.pw1_ttfs                                              pw1            861,151,872 1,966,080,000    43.80%
stages.1.0.pw2_ttfs                                              pw2            475,074,450   491,520,000    96.65%
stages.1.1.pw1_ttfs                                              pw1            907,115,810 1,966,080,000    46.14%
stages.1.1.pw2_ttfs                                              pw2            477,122,334   491,520,000    97.07%
stages.2.0.pw1_ttfs                                              pw1            392,843,632   983,040,000    39.96%
stages.2.0.pw2_ttfs                                              pw2            227,638,717   245,760,000    92.63%
stages.2.1.pw1_ttfs                                              pw1            460,193,377   983,040,000    46.81%
stages.2.1.pw2_ttfs                                              pw2            238,723,563   245,760,000    97.14%
stages.2.2.pw1_ttfs                                              pw1            491,988,849   983,040,000    50.05%
stages.2.2.pw2_ttfs                                              pw2            241,271,611   245,760,000    98.17%
stages.2.3.pw1_ttfs                                              pw1            469,164,809   983,040,000    47.73%
stages.2.3.pw2_ttfs                                              pw2            241,050,816   245,760,000    98.08%
stages.2.4.pw1_ttfs                                              pw1            465,156,231   983,040,000    47.32%
stages.2.4.pw2_ttfs                                              pw2            241,396,486   245,760,000    98.22%
stages.2.5.pw1_ttfs                                              pw1            470,752,450   983,040,000    47.89%
stages.2.5.pw2_ttfs                                              pw2            239,167,651   245,760,000    97.32%
stages.3.0.pw1_ttfs                                              pw1            254,183,925   491,520,000    51.71%
stages.3.0.pw2_ttfs                                              pw2            104,529,320   122,880,000    85.07%
stages.3.1.pw1_ttfs                                              pw1            260,630,013   491,520,000    53.03%
stages.3.1.pw2_ttfs                                              pw2             97,887,307   122,880,000    79.66%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     43.33%
===================================================================================================================

Classification accuracy: 73.86%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  37.26% | silent=6,960,103,814 | total=18,677,760,000
pw2             12 layers | sparsity=  67.60% | silent=3,156,592,092 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   25.43% | TTFS points= 4 | silent=2,499,652,683 | total=9,830,400,000
Stage 1:   55.35% | TTFS points= 4 | silent=2,720,464,466 | total=4,915,200,000
Stage 2:   56.69% | TTFS points=12 | silent=4,179,348,192 | total=7,372,800,000
Stage 3:   58.37% | TTFS points= 4 | silent=717,230,565 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.86%
Activation sparsity:  43.33%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_3\seed_5435\activation_sparsity.md
```
