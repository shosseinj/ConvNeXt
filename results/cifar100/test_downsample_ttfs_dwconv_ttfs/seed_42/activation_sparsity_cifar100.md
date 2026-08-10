# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\test_downsample_ttfs_dwconv_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs
Detected downsampling convolution mode: ttfs

Missing keys:    0
Unexpected keys: 0

==========================================================================================
MODEL SPARSITY STRUCTURE
==========================================================================================
Spiking blocks:              12
ContinuousTTFSConv2d:        15
PW1 TTFS outputs:            12
PW2 TTFS outputs:            12
Expected total TTFS points:  39
==========================================================================================

TTFS Conv modules:
  downsample_layers.1.0
  downsample_layers.2.0
  downsample_layers.3.0
  stages.0.0.dwconv
  stages.0.1.dwconv
  stages.1.0.dwconv
  stages.1.1.dwconv
  stages.2.0.dwconv
  stages.2.1.dwconv
  stages.2.2.dwconv
  stages.2.3.dwconv
  stages.2.4.dwconv
  stages.2.5.dwconv
  stages.3.0.dwconv
  stages.3.1.dwconv
Batch    1/  79 | samples=   128 | accuracy= 76.56%
Batch   20/  79 | samples=  2560 | accuracy= 73.87%
Batch   40/  79 | samples=  5120 | accuracy= 72.91%
Batch   60/  79 | samples=  7680 | accuracy= 73.02%
Batch   79/  79 | samples= 10000 | accuracy= 73.43%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     177,427,917   491,520,000    36.10%
downsample_layers.2.0                                            downsample     170,862,136   245,760,000    69.52%
downsample_layers.3.0                                            downsample      98,652,041   122,880,000    80.28%
stages.0.0.dwconv                                                dwconv                   0   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1            317,920,800 3,932,160,000     8.09%
stages.0.0.pw2_ttfs                                              pw2            364,428,119   983,040,000    37.07%
stages.0.1.dwconv                                                dwconv             310,000   983,040,000     0.03%
stages.0.1.pw1_ttfs                                              pw1          1,428,907,249 3,932,160,000    36.34%
stages.0.1.pw2_ttfs                                              pw2            363,755,433   983,040,000    37.00%
stages.1.0.dwconv                                                dwconv          98,285,748   491,520,000    20.00%
stages.1.0.pw1_ttfs                                              pw1            243,439,708 1,966,080,000    12.38%
stages.1.0.pw2_ttfs                                              pw2            387,214,499   491,520,000    78.78%
stages.1.1.dwconv                                                dwconv          74,515,176   491,520,000    15.16%
stages.1.1.pw1_ttfs                                              pw1            246,948,942 1,966,080,000    12.56%
stages.1.1.pw2_ttfs                                              pw2            372,404,581   491,520,000    75.77%
stages.2.0.dwconv                                                dwconv          96,589,908   245,760,000    39.30%
stages.2.0.pw1_ttfs                                              pw1            192,545,356   983,040,000    19.59%
stages.2.0.pw2_ttfs                                              pw2            233,108,472   245,760,000    94.85%
stages.2.1.dwconv                                                dwconv          84,232,620   245,760,000    34.27%
stages.2.1.pw1_ttfs                                              pw1            146,426,659   983,040,000    14.90%
stages.2.1.pw2_ttfs                                              pw2            226,388,698   245,760,000    92.12%
stages.2.2.dwconv                                                dwconv          72,763,301   245,760,000    29.61%
stages.2.2.pw1_ttfs                                              pw1            181,775,231   983,040,000    18.49%
stages.2.2.pw2_ttfs                                              pw2            229,366,163   245,760,000    93.33%
stages.2.3.dwconv                                                dwconv          75,631,735   245,760,000    30.77%
stages.2.3.pw1_ttfs                                              pw1            235,060,909   983,040,000    23.91%
stages.2.3.pw2_ttfs                                              pw2            222,172,992   245,760,000    90.40%
stages.2.4.dwconv                                                dwconv          78,937,333   245,760,000    32.12%
stages.2.4.pw1_ttfs                                              pw1            274,052,723   983,040,000    27.88%
stages.2.4.pw2_ttfs                                              pw2            227,132,925   245,760,000    92.42%
stages.2.5.dwconv                                                dwconv          82,153,200   245,760,000    33.43%
stages.2.5.pw1_ttfs                                              pw1            324,329,884   983,040,000    32.99%
stages.2.5.pw2_ttfs                                              pw2            224,861,968   245,760,000    91.50%
stages.3.0.dwconv                                                dwconv          58,366,392   122,880,000    47.50%
stages.3.0.pw1_ttfs                                              pw1            213,676,664   491,520,000    43.47%
stages.3.0.pw2_ttfs                                              pw2             95,678,209   122,880,000    77.86%
stages.3.1.dwconv                                                dwconv          57,479,996   122,880,000    46.78%
stages.3.1.pw1_ttfs                                              pw1            224,032,761   491,520,000    45.58%
stages.3.1.pw2_ttfs                                              pw2             92,875,998   122,880,000    75.58%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     28.72%
===================================================================================================================

Classification accuracy: 73.43%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  16.69% | silent=779,265,409 | total=4,669,440,000
pw1             12 layers | sparsity=  21.57% | silent=4,029,116,886 | total=18,677,760,000
pw2             12 layers | sparsity=  65.09% | silent=3,039,388,057 | total=4,669,440,000
downsample       3 layers | sparsity=  51.96% | silent=446,942,094 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   20.98% | TTFS points= 6 | silent=2,475,321,601 | total=11,796,480,000
Stage 1:   25.04% | TTFS points= 7 | silent=1,600,236,571 | total=6,389,760,000
Stage 2:   37.15% | TTFS points=19 | silent=3,378,392,213 | total=9,093,120,000
Stage 3:   52.63% | TTFS points= 7 | silent=840,762,061 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.43%
Activation sparsity:  28.72%
TTFS layers/points:   39
================================================================================

Markdown report saved to: results\cifar100\test_downsample_ttfs_dwconv_ttfs\seed_42\activation_sparsity.md
```
