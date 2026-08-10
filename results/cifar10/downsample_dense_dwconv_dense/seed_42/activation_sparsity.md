# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar10\downsample_dense_dwconv_dense\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar10
Evaluation samples: 10000

Detected depthwise convolution mode: dense (legacy state-dict inference)
Detected downsampling convolution mode: dense (legacy state-dict inference)

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
Batch    1/  79 | samples=   128 | accuracy= 92.19%
Batch   20/  79 | samples=  2560 | accuracy= 93.24%
Batch   40/  79 | samples=  5120 | accuracy= 93.52%
Batch   60/  79 | samples=  7680 | accuracy= 93.70%
Batch   79/  79 | samples= 10000 | accuracy= 93.83%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1             29,476,785 3,932,160,000     0.75%
stages.0.0.pw2_ttfs                                              pw2            284,349,200   983,040,000    28.93%
stages.0.1.pw1_ttfs                                              pw1            113,301,643 3,932,160,000     2.88%
stages.0.1.pw2_ttfs                                              pw2            241,124,490   983,040,000    24.53%
stages.1.0.pw1_ttfs                                              pw1            923,735,648 1,966,080,000    46.98%
stages.1.0.pw2_ttfs                                              pw2            481,996,958   491,520,000    98.06%
stages.1.1.pw1_ttfs                                              pw1            886,662,219 1,966,080,000    45.10%
stages.1.1.pw2_ttfs                                              pw2            478,674,562   491,520,000    97.39%
stages.2.0.pw1_ttfs                                              pw1            469,929,480   983,040,000    47.80%
stages.2.0.pw2_ttfs                                              pw2            237,216,269   245,760,000    96.52%
stages.2.1.pw1_ttfs                                              pw1            515,471,928   983,040,000    52.44%
stages.2.1.pw2_ttfs                                              pw2            238,841,358   245,760,000    97.18%
stages.2.2.pw1_ttfs                                              pw1            508,935,237   983,040,000    51.77%
stages.2.2.pw2_ttfs                                              pw2            242,196,981   245,760,000    98.55%
stages.2.3.pw1_ttfs                                              pw1            490,632,138   983,040,000    49.91%
stages.2.3.pw2_ttfs                                              pw2            242,933,464   245,760,000    98.85%
stages.2.4.pw1_ttfs                                              pw1            541,221,965   983,040,000    55.06%
stages.2.4.pw2_ttfs                                              pw2            239,984,290   245,760,000    97.65%
stages.2.5.pw1_ttfs                                              pw1            476,804,711   983,040,000    48.50%
stages.2.5.pw2_ttfs                                              pw2            234,187,951   245,760,000    95.29%
stages.3.0.pw1_ttfs                                              pw1            244,155,202   491,520,000    49.67%
stages.3.0.pw2_ttfs                                              pw2            110,492,494   122,880,000    89.92%
stages.3.1.pw1_ttfs                                              pw1            272,462,749   491,520,000    55.43%
stages.3.1.pw2_ttfs                                              pw2            108,655,988   122,880,000    88.42%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.89%
===================================================================================================================

Classification accuracy: 93.83%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  29.30% | silent=5,472,789,705 | total=18,677,760,000
pw2             12 layers | sparsity=  67.26% | silent=3,140,654,005 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    6.80% | TTFS points= 4 | silent=668,252,118 | total=9,830,400,000
Stage 1:   56.38% | TTFS points= 4 | silent=2,771,069,387 | total=4,915,200,000
Stage 2:   60.20% | TTFS points=12 | silent=4,438,355,772 | total=7,372,800,000
Stage 3:   59.88% | TTFS points= 4 | silent=735,766,433 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.83%
Activation sparsity:  36.89%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar10\downsample_dense_dwconv_dense\seed_42\activation_sparsity.md
```
