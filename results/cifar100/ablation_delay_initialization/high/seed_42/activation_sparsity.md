# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 74.22%
Batch   20/  79 | samples=  2560 | accuracy= 73.95%
Batch   40/  79 | samples=  5120 | accuracy= 72.95%
Batch   60/  79 | samples=  7680 | accuracy= 73.24%
Batch   79/  79 | samples= 10000 | accuracy= 73.48%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1            132,862,973 3,932,160,000     3.38%        36,412,394
stages.0.0.pw2_ttfs                                              pw2             99,304,493   983,040,000    10.10%        36,473,251
stages.0.1.pw1_ttfs                                              pw1                    147 3,932,160,000     0.00%        37,203,368
stages.0.1.pw2_ttfs                                              pw2            150,982,929   983,040,000    15.36%        37,748,735
stages.1.0.pw1_ttfs                                              pw1            889,009,158 1,966,080,000    45.22%        35,003,662
stages.1.0.pw2_ttfs                                              pw2            476,836,652   491,520,000    97.01%        20,679,760
stages.1.1.pw1_ttfs                                              pw1            765,395,851 1,966,080,000    38.93%        35,799,987
stages.1.1.pw2_ttfs                                              pw2            434,756,434   491,520,000    88.45%        23,053,136
stages.2.0.pw1_ttfs                                              pw1            377,188,224   983,040,000    38.37%        35,360,424
stages.2.0.pw2_ttfs                                              pw2            231,594,025   245,760,000    94.24%        23,264,708
stages.2.1.pw1_ttfs                                              pw1            466,584,255   983,040,000    47.46%        35,218,030
stages.2.1.pw2_ttfs                                              pw2            235,922,122   245,760,000    96.00%        19,831,901
stages.2.2.pw1_ttfs                                              pw1            464,146,066   983,040,000    47.22%        35,745,050
stages.2.2.pw2_ttfs                                              pw2            237,774,702   245,760,000    96.75%        19,925,527
stages.2.3.pw1_ttfs                                              pw1            439,386,659   983,040,000    44.70%        36,085,088
stages.2.3.pw2_ttfs                                              pw2            242,017,858   245,760,000    98.48%        20,876,288
stages.2.4.pw1_ttfs                                              pw1            475,648,043   983,040,000    48.39%        36,111,759
stages.2.4.pw2_ttfs                                              pw2            242,138,488   245,760,000    98.53%        19,483,851
stages.2.5.pw1_ttfs                                              pw1            427,109,563   983,040,000    43.45%        36,102,548
stages.2.5.pw2_ttfs                                              pw2            240,054,658   245,760,000    97.68%        21,347,729
stages.3.0.pw1_ttfs                                              pw1            262,905,711   491,520,000    53.49%        34,298,498
stages.3.0.pw2_ttfs                                              pw2            105,733,005   122,880,000    86.05%        17,557,577
stages.3.1.pw1_ttfs                                              pw1            254,427,398   491,520,000    51.76%        34,161,335
stages.3.1.pw2_ttfs                                              pw2             89,986,102   122,880,000    73.23%        18,208,712
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     33.16%
===================================================================================================================

Classification accuracy: 73.48%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 705,953,320
Layerwise SynOps total:    7,059,533,196,096

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  26.53% | silent=4,954,664,048 | total=18,677,760,000
pw2             12 layers | sparsity=  59.69% | silent=2,787,101,468 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    3.90% | TTFS points= 4 | silent=383,150,542 | total=9,830,400,000
Stage 1:   52.21% | TTFS points= 4 | silent=2,565,998,095 | total=4,915,200,000
Stage 2:   55.33% | TTFS points=12 | silent=4,079,564,663 | total=7,372,800,000
Stage 3:   58.03% | TTFS points= 4 | silent=713,052,216 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.48%
Activation sparsity:  33.16%
Theoretical SynOps:   705,953,320 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_42\activation_sparsity.md
```
