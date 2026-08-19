# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 73.44%
Batch   20/  79 | samples=  2560 | accuracy= 74.02%
Batch   40/  79 | samples=  5120 | accuracy= 73.50%
Batch   60/  79 | samples=  7680 | accuracy= 73.74%
Batch   79/  79 | samples= 10000 | accuracy= 74.31%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,894,154,063 3,932,160,000    48.17%        37,106,274
stages.0.0.pw2_ttfs                                              pw2            459,812,511   983,040,000    46.77%        19,564,857
stages.0.1.pw1_ttfs                                              pw1          1,933,765,539 3,932,160,000    49.18%        36,939,220
stages.0.1.pw2_ttfs                                              pw2            502,259,834   983,040,000    51.09%        19,184,587
stages.1.0.pw1_ttfs                                              pw1            889,208,393 1,966,080,000    45.23%        35,245,492
stages.1.0.pw2_ttfs                                              pw2            473,982,661   491,520,000    96.43%        20,675,935
stages.1.1.pw1_ttfs                                              pw1            816,895,037 1,966,080,000    41.55%        35,732,454
stages.1.1.pw2_ttfs                                              pw2            476,723,135   491,520,000    96.99%        22,064,351
stages.2.0.pw1_ttfs                                              pw1            441,419,962   983,040,000    44.90%        34,896,407
stages.2.0.pw2_ttfs                                              pw2            234,059,006   245,760,000    95.24%        20,798,209
stages.2.1.pw1_ttfs                                              pw1            466,566,566   983,040,000    47.46%        35,444,661
stages.2.1.pw2_ttfs                                              pw2            240,234,465   245,760,000    97.75%        19,832,580
stages.2.2.pw1_ttfs                                              pw1            477,338,419   983,040,000    48.56%        35,016,204
stages.2.2.pw2_ttfs                                              pw2            233,924,859   245,760,000    95.18%        19,418,941
stages.2.3.pw1_ttfs                                              pw1            482,197,791   983,040,000    49.05%        36,008,063
stages.2.3.pw2_ttfs                                              pw2            242,908,744   245,760,000    98.84%        19,232,341
stages.2.4.pw1_ttfs                                              pw1            491,595,718   983,040,000    50.01%        35,735,148
stages.2.4.pw2_ttfs                                              pw2            237,890,938   245,760,000    96.80%        18,871,460
stages.2.5.pw1_ttfs                                              pw1            479,675,233   983,040,000    48.80%        35,794,699
stages.2.5.pw2_ttfs                                              pw2            238,872,693   245,760,000    97.20%        19,329,207
stages.3.0.pw1_ttfs                                              pw1            260,580,136   491,520,000    53.02%        34,015,256
stages.3.0.pw2_ttfs                                              pw2            104,727,226   122,880,000    85.23%        17,736,182
stages.3.1.pw1_ttfs                                              pw1            260,303,400   491,520,000    52.96%        34,439,231
stages.3.1.pw2_ttfs                                              pw2             95,754,233   122,880,000    77.92%        17,757,435
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     53.26%
===================================================================================================================

Classification accuracy: 74.31%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 660,839,195
Layerwise SynOps total:    6,608,391,949,632

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  47.62% | silent=8,893,700,257 | total=18,677,760,000
pw2             12 layers | sparsity=  75.84% | silent=3,541,150,305 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   48.73% | TTFS points= 4 | silent=4,789,991,947 | total=9,830,400,000
Stage 1:   54.05% | TTFS points= 4 | silent=2,656,809,226 | total=4,915,200,000
Stage 2:   57.87% | TTFS points=12 | silent=4,266,684,394 | total=7,372,800,000
Stage 3:   58.70% | TTFS points= 4 | silent=721,364,995 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.31%
Activation sparsity:  53.26%
Theoretical SynOps:   660,839,195 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_7777\activation_sparsity.md
```
