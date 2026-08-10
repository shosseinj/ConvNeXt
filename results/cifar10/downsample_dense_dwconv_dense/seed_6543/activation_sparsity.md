# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar10\downsample_dense_dwconv_dense\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar10
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
Batch    1/  79 | samples=   128 | accuracy= 94.53%
Batch   20/  79 | samples=  2560 | accuracy= 94.34%
Batch   40/  79 | samples=  5120 | accuracy= 94.08%
Batch   60/  79 | samples=  7680 | accuracy= 94.34%
Batch   79/  79 | samples= 10000 | accuracy= 94.29%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1             27,508,392 3,932,160,000     0.70%
stages.0.0.pw2_ttfs                                              pw2            212,237,782   983,040,000    21.59%
stages.0.1.pw1_ttfs                                              pw1          1,667,175,134 3,932,160,000    42.40%
stages.0.1.pw2_ttfs                                              pw2            231,451,115   983,040,000    23.54%
stages.1.0.pw1_ttfs                                              pw1            909,755,530 1,966,080,000    46.27%
stages.1.0.pw2_ttfs                                              pw2            478,399,601   491,520,000    97.33%
stages.1.1.pw1_ttfs                                              pw1            806,287,177 1,966,080,000    41.01%
stages.1.1.pw2_ttfs                                              pw2            473,306,802   491,520,000    96.29%
stages.2.0.pw1_ttfs                                              pw1            469,353,663   983,040,000    47.75%
stages.2.0.pw2_ttfs                                              pw2            238,401,450   245,760,000    97.01%
stages.2.1.pw1_ttfs                                              pw1            498,215,004   983,040,000    50.68%
stages.2.1.pw2_ttfs                                              pw2            238,266,420   245,760,000    96.95%
stages.2.2.pw1_ttfs                                              pw1            556,241,638   983,040,000    56.58%
stages.2.2.pw2_ttfs                                              pw2            241,321,801   245,760,000    98.19%
stages.2.3.pw1_ttfs                                              pw1            517,984,873   983,040,000    52.69%
stages.2.3.pw2_ttfs                                              pw2            236,207,006   245,760,000    96.11%
stages.2.4.pw1_ttfs                                              pw1            490,795,112   983,040,000    49.93%
stages.2.4.pw2_ttfs                                              pw2            236,480,332   245,760,000    96.22%
stages.2.5.pw1_ttfs                                              pw1            538,670,836   983,040,000    54.80%
stages.2.5.pw2_ttfs                                              pw2            238,570,859   245,760,000    97.07%
stages.3.0.pw1_ttfs                                              pw1            250,532,807   491,520,000    50.97%
stages.3.0.pw2_ttfs                                              pw2            109,713,770   122,880,000    89.29%
stages.3.1.pw1_ttfs                                              pw1            259,193,815   491,520,000    52.73%
stages.3.1.pw2_ttfs                                              pw2            108,021,329   122,880,000    87.91%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     42.98%
===================================================================================================================

Classification accuracy: 94.29%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  37.43% | silent=6,991,713,981 | total=18,677,760,000
pw2             12 layers | sparsity=  65.16% | silent=3,042,378,267 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   21.75% | TTFS points= 4 | silent=2,138,372,423 | total=9,830,400,000
Stage 1:   54.28% | TTFS points= 4 | silent=2,667,749,110 | total=4,915,200,000
Stage 2:   61.04% | TTFS points=12 | silent=4,500,508,994 | total=7,372,800,000
Stage 3:   59.20% | TTFS points= 4 | silent=727,461,721 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             94.29%
Activation sparsity:  42.98%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar10\downsample_dense_dwconv_dense\seed_6543\activation_sparsity.md
```
