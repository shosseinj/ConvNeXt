# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 78.12%
Batch   20/  79 | samples=  2560 | accuracy= 74.57%
Batch   40/  79 | samples=  5120 | accuracy= 74.30%
Batch   60/  79 | samples=  7680 | accuracy= 73.98%
Batch   79/  79 | samples= 10000 | accuracy= 74.21%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                323,867 3,932,160,000     0.01%        37,020,216
stages.0.0.pw2_ttfs                                              pw2             77,443,882   983,040,000     7.88%        37,745,627
stages.0.1.pw1_ttfs                                              pw1                 14,196 3,932,160,000     0.00%        37,521,903
stages.0.1.pw2_ttfs                                              pw2            150,141,018   983,040,000    15.27%        37,748,600
stages.1.0.pw1_ttfs                                              pw1            875,850,896 1,966,080,000    44.55%        35,151,383
stages.1.0.pw2_ttfs                                              pw2            472,064,646   491,520,000    96.04%        20,932,399
stages.1.1.pw1_ttfs                                              pw1            874,315,859 1,966,080,000    44.47%        35,961,114
stages.1.1.pw2_ttfs                                              pw2            451,338,943   491,520,000    91.83%        20,961,872
stages.2.0.pw1_ttfs                                              pw1            413,013,915   983,040,000    42.01%        34,815,265
stages.2.0.pw2_ttfs                                              pw2            233,866,745   245,760,000    95.16%        21,889,002
stages.2.1.pw1_ttfs                                              pw1            456,766,025   983,040,000    46.46%        35,894,798
stages.2.1.pw2_ttfs                                              pw2            242,362,981   245,760,000    98.62%        20,208,921
stages.2.2.pw1_ttfs                                              pw1            454,925,944   983,040,000    46.28%        35,196,655
stages.2.2.pw2_ttfs                                              pw2            235,323,144   245,760,000    95.75%        20,279,580
stages.2.3.pw1_ttfs                                              pw1            485,652,253   983,040,000    49.40%        35,709,381
stages.2.3.pw2_ttfs                                              pw2            239,300,181   245,760,000    97.37%        19,099,689
stages.2.4.pw1_ttfs                                              pw1            477,586,400   983,040,000    48.58%        35,699,604
stages.2.4.pw2_ttfs                                              pw2            235,660,483   245,760,000    95.89%        19,409,418
stages.2.5.pw1_ttfs                                              pw1            468,571,982   983,040,000    47.67%        35,889,969
stages.2.5.pw2_ttfs                                              pw2            238,133,823   245,760,000    96.90%        19,755,572
stages.3.0.pw1_ttfs                                              pw1            256,252,644   491,520,000    52.13%        34,077,655
stages.3.0.pw2_ttfs                                              pw2            103,670,432   122,880,000    84.37%        18,068,533
stages.3.1.pw1_ttfs                                              pw1            258,400,246   491,520,000    52.57%        34,378,855
stages.3.1.pw2_ttfs                                              pw2             94,924,599   122,880,000    77.25%        17,903,597
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     33.39%
===================================================================================================================

Classification accuracy: 74.21%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 701,319,607
Layerwise SynOps total:    7,013,196,065,184

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  26.89% | silent=5,021,674,227 | total=18,677,760,000
pw2             12 layers | sparsity=  59.41% | silent=2,774,230,877 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    2.32% | TTFS points= 4 | silent=227,922,963 | total=9,830,400,000
Stage 1:   54.39% | TTFS points= 4 | silent=2,673,570,344 | total=4,915,200,000
Stage 2:   56.71% | TTFS points=12 | silent=4,181,163,876 | total=7,372,800,000
Stage 3:   58.04% | TTFS points= 4 | silent=713,247,921 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.21%
Activation sparsity:  33.39%
Theoretical SynOps:   701,319,607 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_7777\activation_sparsity.md
```
