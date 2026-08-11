# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 77.34%
Batch   20/  79 | samples=  2560 | accuracy= 72.34%
Batch   40/  79 | samples=  5120 | accuracy= 71.11%
Batch   60/  79 | samples=  7680 | accuracy= 71.50%
Batch   79/  79 | samples= 10000 | accuracy= 71.62%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,408,913,693 3,932,160,000    35.83%
stages.0.0.pw2_ttfs                                              pw2            172,019,548   983,040,000    17.50%
stages.0.1.pw1_ttfs                                              pw1          1,855,159,543 3,932,160,000    47.18%
stages.0.1.pw2_ttfs                                              pw2            217,434,732   983,040,000    22.12%
stages.1.0.pw1_ttfs                                              pw1            897,516,615 1,966,080,000    45.65%
stages.1.0.pw2_ttfs                                              pw2            478,867,579   491,520,000    97.43%
stages.1.1.pw1_ttfs                                              pw1            714,004,496 1,966,080,000    36.32%
stages.1.1.pw2_ttfs                                              pw2            469,378,737   491,520,000    95.50%
stages.2.0.pw1_ttfs                                              pw1            441,452,328   983,040,000    44.91%
stages.2.0.pw2_ttfs                                              pw2            228,290,951   245,760,000    92.89%
stages.2.1.pw1_ttfs                                              pw1            464,992,692   983,040,000    47.30%
stages.2.1.pw2_ttfs                                              pw2            240,068,932   245,760,000    97.68%
stages.2.2.pw1_ttfs                                              pw1            454,966,753   983,040,000    46.28%
stages.2.2.pw2_ttfs                                              pw2            237,863,427   245,760,000    96.79%
stages.2.3.pw1_ttfs                                              pw1            438,979,017   983,040,000    44.66%
stages.2.3.pw2_ttfs                                              pw2            238,017,868   245,760,000    96.85%
stages.2.4.pw1_ttfs                                              pw1            459,516,919   983,040,000    46.74%
stages.2.4.pw2_ttfs                                              pw2            242,057,480   245,760,000    98.49%
stages.2.5.pw1_ttfs                                              pw1            446,131,152   983,040,000    45.38%
stages.2.5.pw2_ttfs                                              pw2            241,410,733   245,760,000    98.23%
stages.3.0.pw1_ttfs                                              pw1            255,647,055   491,520,000    52.01%
stages.3.0.pw2_ttfs                                              pw2            110,980,200   122,880,000    90.32%
stages.3.1.pw1_ttfs                                              pw1            245,463,464   491,520,000    49.94%
stages.3.1.pw2_ttfs                                              pw2             93,611,479   122,880,000    76.18%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     47.34%
===================================================================================================================

Classification accuracy: 71.62%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  43.27% | silent=8,082,743,727 | total=18,677,760,000
pw2             12 layers | sparsity=  63.61% | silent=2,970,001,666 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   37.17% | TTFS points= 4 | silent=3,653,527,516 | total=9,830,400,000
Stage 1:   52.08% | TTFS points= 4 | silent=2,559,767,427 | total=4,915,200,000
Stage 2:   56.07% | TTFS points=12 | silent=4,133,748,252 | total=7,372,800,000
Stage 3:   57.43% | TTFS points= 4 | silent=705,702,198 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             71.62%
Activation sparsity:  47.34%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_42\activation_sparsity.md
```
