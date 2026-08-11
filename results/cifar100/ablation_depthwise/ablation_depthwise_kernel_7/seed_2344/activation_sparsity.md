# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_2344\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 71.45%
Batch   40/  79 | samples=  5120 | accuracy= 70.80%
Batch   60/  79 | samples=  7680 | accuracy= 70.82%
Batch   79/  79 | samples= 10000 | accuracy= 70.97%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          2,126,102,199 3,932,160,000    54.07%
stages.0.0.pw2_ttfs                                              pw2            278,005,425   983,040,000    28.28%
stages.0.1.pw1_ttfs                                              pw1          1,713,114,822 3,932,160,000    43.57%
stages.0.1.pw2_ttfs                                              pw2            347,172,949   983,040,000    35.32%
stages.1.0.pw1_ttfs                                              pw1            730,426,429 1,966,080,000    37.15%
stages.1.0.pw2_ttfs                                              pw2            472,680,036   491,520,000    96.17%
stages.1.1.pw1_ttfs                                              pw1            694,870,000 1,966,080,000    35.34%
stages.1.1.pw2_ttfs                                              pw2            474,127,114   491,520,000    96.46%
stages.2.0.pw1_ttfs                                              pw1            478,418,056   983,040,000    48.67%
stages.2.0.pw2_ttfs                                              pw2            229,800,906   245,760,000    93.51%
stages.2.1.pw1_ttfs                                              pw1            467,590,461   983,040,000    47.57%
stages.2.1.pw2_ttfs                                              pw2            231,678,245   245,760,000    94.27%
stages.2.2.pw1_ttfs                                              pw1            463,640,535   983,040,000    47.16%
stages.2.2.pw2_ttfs                                              pw2            236,850,714   245,760,000    96.37%
stages.2.3.pw1_ttfs                                              pw1            476,127,853   983,040,000    48.43%
stages.2.3.pw2_ttfs                                              pw2            239,321,023   245,760,000    97.38%
stages.2.4.pw1_ttfs                                              pw1            476,341,006   983,040,000    48.46%
stages.2.4.pw2_ttfs                                              pw2            241,933,258   245,760,000    98.44%
stages.2.5.pw1_ttfs                                              pw1            431,053,376   983,040,000    43.85%
stages.2.5.pw2_ttfs                                              pw2            235,825,566   245,760,000    95.96%
stages.3.0.pw1_ttfs                                              pw1            247,514,527   491,520,000    50.36%
stages.3.0.pw2_ttfs                                              pw2            110,470,591   122,880,000    89.90%
stages.3.1.pw1_ttfs                                              pw1            243,448,927   491,520,000    49.53%
stages.3.1.pw2_ttfs                                              pw2             97,257,293   122,880,000    79.15%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     50.30%
===================================================================================================================

Classification accuracy: 70.97%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  45.77% | silent=8,548,648,191 | total=18,677,760,000
pw2             12 layers | sparsity=  68.43% | silent=3,195,123,120 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   45.41% | TTFS points= 4 | silent=4,464,395,395 | total=9,830,400,000
Stage 1:   48.26% | TTFS points= 4 | silent=2,372,103,579 | total=4,915,200,000
Stage 2:   57.08% | TTFS points=12 | silent=4,208,580,999 | total=7,372,800,000
Stage 3:   56.86% | TTFS points= 4 | silent=698,691,338 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             70.97%
Activation sparsity:  50.30%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_2344\activation_sparsity.md
```
