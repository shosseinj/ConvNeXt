# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_6543\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 72.66%
Batch   20/  79 | samples=  2560 | accuracy= 74.80%
Batch   40/  79 | samples=  5120 | accuracy= 73.50%
Batch   60/  79 | samples=  7680 | accuracy= 73.59%
Batch   79/  79 | samples= 10000 | accuracy= 73.79%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,690,747,498 3,932,160,000    43.00%        37,376,907
stages.0.0.pw2_ttfs                                              pw2            470,299,108   983,040,000    47.84%        21,517,560
stages.0.1.pw1_ttfs                                              pw1          1,621,649,867 3,932,160,000    41.24%        36,998,154
stages.0.1.pw2_ttfs                                              pw2            454,345,990   983,040,000    46.22%        22,180,897
stages.1.0.pw1_ttfs                                              pw1            853,338,474 1,966,080,000    43.40%        34,975,725
stages.1.0.pw2_ttfs                                              pw2            468,043,428   491,520,000    95.22%        21,364,637
stages.1.1.pw1_ttfs                                              pw1            838,866,862 1,966,080,000    42.67%        36,199,544
stages.1.1.pw2_ttfs                                              pw2            443,241,774   491,520,000    90.18%        21,642,492
stages.2.0.pw1_ttfs                                              pw1            423,677,894   983,040,000    43.10%        34,784,573
stages.2.0.pw2_ttfs                                              pw2            237,362,453   245,760,000    96.58%        21,479,505
stages.2.1.pw1_ttfs                                              pw1            467,546,020   983,040,000    47.56%        34,798,536
stages.2.1.pw2_ttfs                                              pw2            238,248,915   245,760,000    96.94%        19,794,969
stages.2.2.pw1_ttfs                                              pw1            468,418,227   983,040,000    47.65%        35,432,452
stages.2.2.pw2_ttfs                                              pw2            241,958,766   245,760,000    98.45%        19,761,476
stages.2.3.pw1_ttfs                                              pw1            481,338,811   983,040,000    48.96%        35,792,065
stages.2.3.pw2_ttfs                                              pw2            241,975,323   245,760,000    98.46%        19,265,326
stages.2.4.pw1_ttfs                                              pw1            485,830,576   983,040,000    49.42%        35,139,413
stages.2.4.pw2_ttfs                                              pw2            231,916,455   245,760,000    94.37%        19,092,842
stages.2.5.pw1_ttfs                                              pw1            466,328,610   983,040,000    47.44%        35,997,781
stages.2.5.pw2_ttfs                                              pw2            243,064,515   245,760,000    98.90%        19,841,717
stages.3.0.pw1_ttfs                                              pw1            260,948,956   491,520,000    53.09%        33,918,764
stages.3.0.pw2_ttfs                                              pw2            107,143,557   122,880,000    87.19%        17,707,856
stages.3.1.pw1_ttfs                                              pw1            250,222,931   491,520,000    50.91%        34,415,208
stages.3.1.pw2_ttfs                                              pw2             92,039,698   122,880,000    74.90%        18,531,615
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     50.45%
===================================================================================================================

Classification accuracy: 73.79%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 668,010,014
Layerwise SynOps total:    6,680,100,138,912

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  44.49% | silent=8,308,914,726 | total=18,677,760,000
pw2             12 layers | sparsity=  74.31% | silent=3,469,639,982 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   43.10% | TTFS points= 4 | silent=4,237,042,463 | total=9,830,400,000
Stage 1:   52.97% | TTFS points= 4 | silent=2,603,490,538 | total=4,915,200,000
Stage 2:   57.34% | TTFS points=12 | silent=4,227,666,565 | total=7,372,800,000
Stage 3:   57.81% | TTFS points= 4 | silent=710,355,142 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.79%
Activation sparsity:  50.45%
Theoretical SynOps:   668,010,014 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_6543\activation_sparsity.md
```
