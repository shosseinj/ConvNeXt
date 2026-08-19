# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_42\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 72.77%
Batch   40/  79 | samples=  5120 | accuracy= 72.03%
Batch   60/  79 | samples=  7680 | accuracy= 72.27%
Batch   79/  79 | samples= 10000 | accuracy= 72.81%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,902,233,670 3,932,160,000    48.38%        36,964,127
stages.0.0.pw2_ttfs                                              pw2            529,804,495   983,040,000    53.89%        19,487,293
stages.0.1.pw1_ttfs                                              pw1          2,013,990,840 3,932,160,000    51.22%        37,196,509
stages.0.1.pw2_ttfs                                              pw2            499,169,283   983,040,000    50.78%        18,414,424
stages.1.0.pw1_ttfs                                              pw1            924,296,019 1,966,080,000    47.01%        34,972,507
stages.1.0.pw2_ttfs                                              pw2            481,651,918   491,520,000    97.99%        20,002,252
stages.1.1.pw1_ttfs                                              pw1            879,290,271 1,966,080,000    44.72%        35,710,307
stages.1.1.pw2_ttfs                                              pw2            470,928,137   491,520,000    95.81%        20,866,363
stages.2.0.pw1_ttfs                                              pw1            408,823,359   983,040,000    41.59%        35,036,312
stages.2.0.pw2_ttfs                                              pw2            231,771,055   245,760,000    94.31%        22,049,919
stages.2.1.pw1_ttfs                                              pw1            479,120,230   983,040,000    48.74%        35,352,807
stages.2.1.pw2_ttfs                                              pw2            240,399,010   245,760,000    97.82%        19,350,519
stages.2.2.pw1_ttfs                                              pw1            479,933,362   983,040,000    48.82%        35,433,175
stages.2.2.pw2_ttfs                                              pw2            239,302,380   245,760,000    97.37%        19,319,295
stages.2.3.pw1_ttfs                                              pw1            483,509,745   983,040,000    49.19%        35,985,535
stages.2.3.pw2_ttfs                                              pw2            241,921,335   245,760,000    98.44%        19,181,962
stages.2.4.pw1_ttfs                                              pw1            438,868,069   983,040,000    44.64%        36,125,116
stages.2.4.pw2_ttfs                                              pw2            242,325,993   245,760,000    98.60%        20,896,202
stages.2.5.pw1_ttfs                                              pw1            484,253,998   983,040,000    49.26%        35,989,328
stages.2.5.pw2_ttfs                                              pw2            241,742,394   245,760,000    98.37%        19,153,382
stages.3.0.pw1_ttfs                                              pw1            265,399,438   491,520,000    54.00%        34,017,046
stages.3.0.pw2_ttfs                                              pw2            105,032,730   122,880,000    85.48%        17,366,059
stages.3.1.pw1_ttfs                                              pw1            249,833,686   491,520,000    50.83%        34,403,211
stages.3.1.pw2_ttfs                                              pw2             93,221,105   122,880,000    75.86%        18,561,509
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     54.08%
===================================================================================================================

Classification accuracy: 72.81%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 661,835,159
Layerwise SynOps total:    6,618,351,587,904

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  48.24% | silent=9,009,552,687 | total=18,677,760,000
pw2             12 layers | sparsity=  77.47% | silent=3,617,269,835 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   50.31% | TTFS points= 4 | silent=4,945,198,288 | total=9,830,400,000
Stage 1:   56.07% | TTFS points= 4 | silent=2,756,166,345 | total=4,915,200,000
Stage 2:   57.13% | TTFS points=12 | silent=4,211,970,930 | total=7,372,800,000
Stage 3:   58.06% | TTFS points= 4 | silent=713,486,959 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.81%
Activation sparsity:  54.08%
Theoretical SynOps:   661,835,159 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_42\activation_sparsity.md
```
