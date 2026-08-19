# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 76.56%
Batch   20/  79 | samples=  2560 | accuracy= 74.84%
Batch   40/  79 | samples=  5120 | accuracy= 73.75%
Batch   60/  79 | samples=  7680 | accuracy= 74.11%
Batch   79/  79 | samples= 10000 | accuracy= 74.19%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,506,247,852 3,932,160,000    38.31%        37,067,659
stages.0.0.pw2_ttfs                                              pw2            516,844,793   983,040,000    52.58%        23,288,757
stages.0.1.pw1_ttfs                                              pw1          2,388,899,397 3,932,160,000    60.75%        36,451,089
stages.0.1.pw2_ttfs                                              pw2            491,520,027   983,040,000    50.00%        14,815,302
stages.1.0.pw1_ttfs                                              pw1            896,843,966 1,966,080,000    45.62%        35,317,852
stages.1.0.pw2_ttfs                                              pw2            477,045,883   491,520,000    97.06%        20,529,332
stages.1.1.pw1_ttfs                                              pw1            940,439,653 1,966,080,000    47.83%        35,438,964
stages.1.1.pw2_ttfs                                              pw2            480,154,582   491,520,000    97.69%        19,692,295
stages.2.0.pw1_ttfs                                              pw1            390,982,916   983,040,000    39.77%        35,312,257
stages.2.0.pw2_ttfs                                              pw2            234,250,086   245,760,000    95.32%        22,734,992
stages.2.1.pw1_ttfs                                              pw1            467,386,231   983,040,000    47.54%        35,336,947
stages.2.1.pw2_ttfs                                              pw2            239,770,907   245,760,000    97.56%        19,801,105
stages.2.2.pw1_ttfs                                              pw1            446,979,282   983,040,000    45.47%        35,352,121
stages.2.2.pw2_ttfs                                              pw2            238,346,946   245,760,000    96.98%        20,584,732
stages.2.3.pw1_ttfs                                              pw1            480,156,373   983,040,000    48.84%        35,736,952
stages.2.3.pw2_ttfs                                              pw2            241,637,845   245,760,000    98.32%        19,310,731
stages.2.4.pw1_ttfs                                              pw1            491,540,473   983,040,000    50.00%        35,894,909
stages.2.4.pw2_ttfs                                              pw2            241,948,655   245,760,000    98.45%        18,873,582
stages.2.5.pw1_ttfs                                              pw1            493,758,978   983,040,000    50.23%        35,809,651
stages.2.5.pw2_ttfs                                              pw2            237,880,575   245,760,000    96.79%        18,788,391
stages.3.0.pw1_ttfs                                              pw1            265,065,755   491,520,000    53.93%        33,989,580
stages.3.0.pw2_ttfs                                              pw2            105,049,713   122,880,000    85.49%        17,391,686
stages.3.1.pw1_ttfs                                              pw1            257,199,278   491,520,000    52.33%        34,477,983
stages.3.1.pw2_ttfs                                              pw2             95,398,160   122,880,000    77.64%        17,995,831
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     54.08%
===================================================================================================================

Classification accuracy: 74.19%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 659,992,699
Layerwise SynOps total:    6,599,926,993,248

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  48.32% | silent=9,025,500,154 | total=18,677,760,000
pw2             12 layers | sparsity=  77.09% | silent=3,599,848,172 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   49.88% | TTFS points= 4 | silent=4,903,512,069 | total=9,830,400,000
Stage 1:   56.85% | TTFS points= 4 | silent=2,794,484,084 | total=4,915,200,000
Stage 2:   57.03% | TTFS points=12 | silent=4,204,639,267 | total=7,372,800,000
Stage 3:   58.81% | TTFS points= 4 | silent=722,712,906 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.19%
Activation sparsity:  54.08%
Theoretical SynOps:   659,992,699 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_42\activation_sparsity.md
```
