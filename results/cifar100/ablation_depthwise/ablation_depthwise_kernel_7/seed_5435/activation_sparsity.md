# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_5435\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 70.62%
Batch   40/  79 | samples=  5120 | accuracy= 70.62%
Batch   60/  79 | samples=  7680 | accuracy= 70.39%
Batch   79/  79 | samples= 10000 | accuracy= 70.78%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,794,551,098 3,932,160,000    45.64%
stages.0.0.pw2_ttfs                                              pw2            391,758,441   983,040,000    39.85%
stages.0.1.pw1_ttfs                                              pw1          1,563,199,358 3,932,160,000    39.75%
stages.0.1.pw2_ttfs                                              pw2            228,460,146   983,040,000    23.24%
stages.1.0.pw1_ttfs                                              pw1            690,614,922 1,966,080,000    35.13%
stages.1.0.pw2_ttfs                                              pw2            474,382,256   491,520,000    96.51%
stages.1.1.pw1_ttfs                                              pw1            796,906,040 1,966,080,000    40.53%
stages.1.1.pw2_ttfs                                              pw2            475,290,672   491,520,000    96.70%
stages.2.0.pw1_ttfs                                              pw1            474,428,072   983,040,000    48.26%
stages.2.0.pw2_ttfs                                              pw2            230,126,030   245,760,000    93.64%
stages.2.1.pw1_ttfs                                              pw1            481,455,924   983,040,000    48.98%
stages.2.1.pw2_ttfs                                              pw2            240,484,262   245,760,000    97.85%
stages.2.2.pw1_ttfs                                              pw1            464,488,699   983,040,000    47.25%
stages.2.2.pw2_ttfs                                              pw2            234,755,001   245,760,000    95.52%
stages.2.3.pw1_ttfs                                              pw1            471,483,022   983,040,000    47.96%
stages.2.3.pw2_ttfs                                              pw2            239,351,966   245,760,000    97.39%
stages.2.4.pw1_ttfs                                              pw1            485,820,685   983,040,000    49.42%
stages.2.4.pw2_ttfs                                              pw2            243,389,835   245,760,000    99.04%
stages.2.5.pw1_ttfs                                              pw1            439,351,193   983,040,000    44.69%
stages.2.5.pw2_ttfs                                              pw2            239,436,435   245,760,000    97.43%
stages.3.0.pw1_ttfs                                              pw1            250,298,762   491,520,000    50.92%
stages.3.0.pw2_ttfs                                              pw2            110,979,158   122,880,000    90.32%
stages.3.1.pw1_ttfs                                              pw1            231,195,359   491,520,000    47.04%
stages.3.1.pw2_ttfs                                              pw2             90,220,464   122,880,000    73.42%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     48.58%
===================================================================================================================

Classification accuracy: 70.78%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  43.60% | silent=8,143,793,134 | total=18,677,760,000
pw2             12 layers | sparsity=  68.50% | silent=3,198,634,666 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   40.47% | TTFS points= 4 | silent=3,977,969,043 | total=9,830,400,000
Stage 1:   49.58% | TTFS points= 4 | silent=2,437,193,890 | total=4,915,200,000
Stage 2:   57.57% | TTFS points=12 | silent=4,244,571,124 | total=7,372,800,000
Stage 3:   55.56% | TTFS points= 4 | silent=682,693,743 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             70.78%
Activation sparsity:  48.58%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\ablation_depthwise\ablation_depthwise_kernel_7\seed_5435\activation_sparsity.md
```
