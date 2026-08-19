# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_7777\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 73.40%
Batch   40/  79 | samples=  5120 | accuracy= 72.70%
Batch   60/  79 | samples=  7680 | accuracy= 72.90%
Batch   79/  79 | samples= 10000 | accuracy= 73.12%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,621,376,779 3,932,160,000    41.23%        36,872,583
stages.0.0.pw2_ttfs                                              pw2            447,657,874   983,040,000    45.54%        22,183,519
stages.0.1.pw1_ttfs                                              pw1          1,707,028,274 3,932,160,000    43.41%        36,836,461
stages.0.1.pw2_ttfs                                              pw2            493,325,101   983,040,000    50.18%        21,361,265
stages.1.0.pw1_ttfs                                              pw1            903,845,947 1,966,080,000    45.97%        35,027,763
stages.1.0.pw2_ttfs                                              pw2            476,830,442   491,520,000    97.01%        20,394,894
stages.1.1.pw1_ttfs                                              pw1            892,750,833 1,966,080,000    45.41%        36,154,849
stages.1.1.pw2_ttfs                                              pw2            464,939,814   491,520,000    94.59%        20,607,920
stages.2.0.pw1_ttfs                                              pw1            409,281,861   983,040,000    41.63%        34,783,731
stages.2.0.pw2_ttfs                                              pw2            230,736,325   245,760,000    93.89%        22,032,313
stages.2.1.pw1_ttfs                                              pw1            451,162,770   983,040,000    45.89%        35,671,737
stages.2.1.pw2_ttfs                                              pw2            240,228,958   245,760,000    97.75%        20,424,086
stages.2.2.pw1_ttfs                                              pw1            459,919,376   983,040,000    46.79%        35,404,427
stages.2.2.pw2_ttfs                                              pw2            237,190,088   245,760,000    96.51%        20,087,832
stages.2.3.pw1_ttfs                                              pw1            475,251,146   983,040,000    48.35%        35,943,830
stages.2.3.pw2_ttfs                                              pw2            242,441,777   245,760,000    98.65%        19,499,092
stages.2.4.pw1_ttfs                                              pw1            466,620,122   983,040,000    47.47%        36,004,146
stages.2.4.pw2_ttfs                                              pw2            240,501,683   245,760,000    97.86%        19,830,523
stages.2.5.pw1_ttfs                                              pw1            483,132,794   983,040,000    49.15%        35,844,375
stages.2.5.pw2_ttfs                                              pw2            237,688,557   245,760,000    96.72%        19,196,437
stages.3.0.pw1_ttfs                                              pw1            260,289,452   491,520,000    52.96%        34,036,769
stages.3.0.pw2_ttfs                                              pw2            105,584,943   122,880,000    85.93%        17,758,506
stages.3.1.pw1_ttfs                                              pw1            257,530,473   491,520,000    52.39%        34,394,345
stages.3.1.pw2_ttfs                                              pw2             95,016,161   122,880,000    77.32%        17,970,396
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     50.97%
===================================================================================================================

Classification accuracy: 73.12%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 668,321,797
Layerwise SynOps total:    6,683,217,968,160

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  44.91% | silent=8,388,189,827 | total=18,677,760,000
pw2             12 layers | sparsity=  75.22% | silent=3,512,141,723 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   43.43% | TTFS points= 4 | silent=4,269,388,028 | total=9,830,400,000
Stage 1:   55.71% | TTFS points= 4 | silent=2,738,367,036 | total=4,915,200,000
Stage 2:   56.62% | TTFS points=12 | silent=4,174,155,457 | total=7,372,800,000
Stage 3:   58.47% | TTFS points= 4 | silent=718,421,029 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.12%
Activation sparsity:  50.97%
Theoretical SynOps:   668,321,797 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p01\seed_7777\activation_sparsity.md
```
