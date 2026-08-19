# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_6543\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 78.91%
Batch   20/  79 | samples=  2560 | accuracy= 73.98%
Batch   40/  79 | samples=  5120 | accuracy= 73.63%
Batch   60/  79 | samples=  7680 | accuracy= 73.72%
Batch   79/  79 | samples= 10000 | accuracy= 73.98%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1             70,196,314 3,932,160,000     1.79%        37,304,202
stages.0.0.pw2_ttfs                                              pw2            214,576,866   983,040,000    21.83%        37,074,851
stages.0.1.pw1_ttfs                                              pw1             23,399,379 3,932,160,000     0.60%        37,196,265
stages.0.1.pw2_ttfs                                              pw2            257,143,849   983,040,000    26.16%        37,524,102
stages.1.0.pw1_ttfs                                              pw1            885,225,487 1,966,080,000    45.02%        35,182,781
stages.1.0.pw2_ttfs                                              pw2            472,461,689   491,520,000    96.12%        20,752,407
stages.1.1.pw1_ttfs                                              pw1            875,529,621 1,966,080,000    44.53%        36,492,214
stages.1.1.pw2_ttfs                                              pw2            447,400,374   491,520,000    91.02%        20,938,567
stages.2.0.pw1_ttfs                                              pw1            451,453,563   983,040,000    45.92%        35,292,137
stages.2.0.pw2_ttfs                                              pw2            237,622,897   245,760,000    96.69%        20,412,919
stages.2.1.pw1_ttfs                                              pw1            473,031,878   983,040,000    48.12%        34,865,873
stages.2.1.pw2_ttfs                                              pw2            235,781,703   245,760,000    95.94%        19,584,312
stages.2.2.pw1_ttfs                                              pw1            494,171,334   983,040,000    50.27%        35,539,414
stages.2.2.pw2_ttfs                                              pw2            239,442,703   245,760,000    97.43%        18,772,557
stages.2.3.pw1_ttfs                                              pw1            456,266,996   983,040,000    46.41%        35,901,086
stages.2.3.pw2_ttfs                                              pw2            241,277,541   245,760,000    98.18%        20,228,083
stages.2.4.pw1_ttfs                                              pw1            499,316,342   983,040,000    50.79%        35,580,784
stages.2.4.pw2_ttfs                                              pw2            236,932,206   245,760,000    96.41%        18,574,988
stages.2.5.pw1_ttfs                                              pw1            466,442,837   983,040,000    47.45%        35,998,861
stages.2.5.pw2_ttfs                                              pw2            242,321,021   245,760,000    98.60%        19,837,331
stages.3.0.pw1_ttfs                                              pw1            262,802,825   491,520,000    53.47%        34,161,543
stages.3.0.pw2_ttfs                                              pw2            106,382,224   122,880,000    86.57%        17,565,479
stages.3.1.pw1_ttfs                                              pw1            258,028,077   491,520,000    52.50%        34,198,474
stages.3.1.pw2_ttfs                                              pw2             91,430,548   122,880,000    74.41%        17,932,180
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     35.29%
===================================================================================================================

Classification accuracy: 73.98%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 696,911,409
Layerwise SynOps total:    6,969,114,090,912

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  27.93% | silent=5,215,864,653 | total=18,677,760,000
pw2             12 layers | sparsity=  64.74% | silent=3,022,773,621 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    5.75% | TTFS points= 4 | silent=565,316,408 | total=9,830,400,000
Stage 1:   54.54% | TTFS points= 4 | silent=2,680,617,171 | total=4,915,200,000
Stage 2:   57.97% | TTFS points=12 | silent=4,274,061,021 | total=7,372,800,000
Stage 3:   58.48% | TTFS points= 4 | silent=718,643,674 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.98%
Activation sparsity:  35.29%
Theoretical SynOps:   696,911,409 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_6543\activation_sparsity.md
```
