# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\downsample_dense_dwconv_dense\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 75.78%
Batch   20/  79 | samples=  2560 | accuracy= 73.55%
Batch   40/  79 | samples=  5120 | accuracy= 73.20%
Batch   60/  79 | samples=  7680 | accuracy= 73.53%
Batch   79/  79 | samples= 10000 | accuracy= 74.00%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                916,099 3,932,160,000     0.02%
stages.0.0.pw2_ttfs                                              pw2            117,591,420   983,040,000    11.96%
stages.0.1.pw1_ttfs                                              pw1                999,567 3,932,160,000     0.03%
stages.0.1.pw2_ttfs                                              pw2            231,682,737   983,040,000    23.57%
stages.1.0.pw1_ttfs                                              pw1            915,602,255 1,966,080,000    46.57%
stages.1.0.pw2_ttfs                                              pw2            473,831,007   491,520,000    96.40%
stages.1.1.pw1_ttfs                                              pw1            839,331,471 1,966,080,000    42.69%
stages.1.1.pw2_ttfs                                              pw2            466,972,858   491,520,000    95.01%
stages.2.0.pw1_ttfs                                              pw1            394,364,289   983,040,000    40.12%
stages.2.0.pw2_ttfs                                              pw2            230,437,284   245,760,000    93.77%
stages.2.1.pw1_ttfs                                              pw1            473,079,069   983,040,000    48.12%
stages.2.1.pw2_ttfs                                              pw2            239,301,697   245,760,000    97.37%
stages.2.2.pw1_ttfs                                              pw1            475,569,297   983,040,000    48.38%
stages.2.2.pw2_ttfs                                              pw2            234,158,326   245,760,000    95.28%
stages.2.3.pw1_ttfs                                              pw1            484,286,227   983,040,000    49.26%
stages.2.3.pw2_ttfs                                              pw2            243,290,748   245,760,000    99.00%
stages.2.4.pw1_ttfs                                              pw1            485,735,677   983,040,000    49.41%
stages.2.4.pw2_ttfs                                              pw2            238,251,367   245,760,000    96.94%
stages.2.5.pw1_ttfs                                              pw1            496,429,774   983,040,000    50.50%
stages.2.5.pw2_ttfs                                              pw2            239,996,963   245,760,000    97.66%
stages.3.0.pw1_ttfs                                              pw1            258,766,383   491,520,000    52.65%
stages.3.0.pw2_ttfs                                              pw2            104,569,588   122,880,000    85.10%
stages.3.1.pw1_ttfs                                              pw1            264,649,868   491,520,000    53.84%
stages.3.1.pw2_ttfs                                              pw2             93,869,790   122,880,000    76.39%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     34.28%
===================================================================================================================

Classification accuracy: 74.00%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  27.25% | silent=5,089,729,976 | total=18,677,760,000
pw2             12 layers | sparsity=  62.40% | silent=2,913,953,785 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    3.57% | TTFS points= 4 | silent=351,189,823 | total=9,830,400,000
Stage 1:   54.84% | TTFS points= 4 | silent=2,695,737,591 | total=4,915,200,000
Stage 2:   57.44% | TTFS points=12 | silent=4,234,900,718 | total=7,372,800,000
Stage 3:   58.74% | TTFS points= 4 | silent=721,855,629 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.00%
Activation sparsity:  34.28%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\downsample_dense_dwconv_dense\seed_7777\activation_sparsity.md
```
