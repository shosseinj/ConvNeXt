# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: dense (metadata)
Detected downsampling convolution mode: dense (metadata)
Detected residual operator: min (metadata)
Detected non-negative effective pointwise weights: False (metadata)

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
Batch   20/  79 | samples=  2560 | accuracy= 73.55%
Batch   40/  79 | samples=  5120 | accuracy= 72.48%
Batch   60/  79 | samples=  7680 | accuracy= 72.54%
Batch   79/  79 | samples= 10000 | accuracy= 72.87%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1              7,740,439 3,932,160,000     0.20%        37,204,382
stages.0.0.pw2_ttfs                                              pw2            325,798,445   983,040,000    33.14%        37,674,428
stages.0.1.pw1_ttfs                                              pw1            980,942,789 3,932,160,000    24.95%        36,925,554
stages.0.1.pw2_ttfs                                              pw2            346,088,120   983,040,000    35.21%        28,331,685
stages.1.0.pw1_ttfs                                              pw1            888,059,803 1,966,080,000    45.17%        35,623,678
stages.1.0.pw2_ttfs                                              pw2            476,793,479   491,520,000    97.00%        20,697,988
stages.1.1.pw1_ttfs                                              pw1            882,007,611 1,966,080,000    44.86%        36,084,008
stages.1.1.pw2_ttfs                                              pw2            473,678,943   491,520,000    96.37%        20,814,190
stages.2.0.pw1_ttfs                                              pw1            400,866,386   983,040,000    40.78%        35,266,020
stages.2.0.pw2_ttfs                                              pw2            230,314,948   245,760,000    93.72%        22,355,467
stages.2.1.pw1_ttfs                                              pw1            488,164,380   983,040,000    49.66%        35,601,516
stages.2.1.pw2_ttfs                                              pw2            240,104,046   245,760,000    97.70%        19,003,224
stages.2.2.pw1_ttfs                                              pw1            451,914,737   983,040,000    45.97%        35,924,316
stages.2.2.pw2_ttfs                                              pw2            239,604,440   245,760,000    97.50%        20,395,210
stages.2.3.pw1_ttfs                                              pw1            487,909,319   983,040,000    49.63%        36,159,463
stages.2.3.pw2_ttfs                                              pw2            241,855,053   245,760,000    98.41%        19,013,018
stages.2.4.pw1_ttfs                                              pw1            479,226,958   983,040,000    48.75%        36,068,907
stages.2.4.pw2_ttfs                                              pw2            239,552,173   245,760,000    97.47%        19,346,421
stages.2.5.pw1_ttfs                                              pw1            449,108,611   983,040,000    45.69%        36,123,403
stages.2.5.pw2_ttfs                                              pw2            239,881,187   245,760,000    97.61%        20,502,965
stages.3.0.pw1_ttfs                                              pw1            268,051,625   491,520,000    54.54%        34,136,810
stages.3.0.pw2_ttfs                                              pw2            105,044,209   122,880,000    85.49%        17,162,371
stages.3.1.pw1_ttfs                                              pw1            260,131,697   491,520,000    52.92%        34,327,665
stages.3.1.pw2_ttfs                                              pw2             94,507,331   122,880,000    76.91%        17,770,622
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     39.82%
===================================================================================================================

Classification accuracy: 72.87%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 692,513,310
Layerwise SynOps total:    6,925,133,097,600

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  32.36% | silent=6,044,124,355 | total=18,677,760,000
pw2             12 layers | sparsity=  69.67% | silent=3,253,222,374 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.89% | TTFS points= 4 | silent=1,660,569,793 | total=9,830,400,000
Stage 1:   55.35% | TTFS points= 4 | silent=2,720,539,836 | total=4,915,200,000
Stage 2:   56.81% | TTFS points=12 | silent=4,188,502,238 | total=7,372,800,000
Stage 3:   59.22% | TTFS points= 4 | silent=727,734,862 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.87%
Activation sparsity:  39.82%
Theoretical SynOps:   692,513,310 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_42\activation_sparsity.md
```
