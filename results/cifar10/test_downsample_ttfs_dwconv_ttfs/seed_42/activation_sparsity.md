# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar10\test_downsample_ttfs_dwconv_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar10
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
Batch    1/  79 | samples=   128 | accuracy= 93.75%
Batch   20/  79 | samples=  2560 | accuracy= 93.24%
Batch   40/  79 | samples=  5120 | accuracy= 93.36%
Batch   60/  79 | samples=  7680 | accuracy= 93.42%
Batch   79/  79 | samples= 10000 | accuracy= 93.59%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     174,334,280   491,520,000    35.47%
downsample_layers.2.0                                            downsample     178,493,278   245,760,000    72.63%
downsample_layers.3.0                                            downsample      96,618,484   122,880,000    78.63%
stages.0.0.dwconv                                                dwconv             301,212   983,040,000     0.03%
stages.0.0.pw1_ttfs                                              pw1            265,192,030 3,932,160,000     6.74%
stages.0.0.pw2_ttfs                                              pw2            335,485,608   983,040,000    34.13%
stages.0.1.dwconv                                                dwconv                   9   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1            196,943,273 3,932,160,000     5.01%
stages.0.1.pw2_ttfs                                              pw2            321,379,796   983,040,000    32.69%
stages.1.0.dwconv                                                dwconv         117,684,712   491,520,000    23.94%
stages.1.0.pw1_ttfs                                              pw1            346,089,097 1,966,080,000    17.60%
stages.1.0.pw2_ttfs                                              pw2            398,016,647   491,520,000    80.98%
stages.1.1.dwconv                                                dwconv          63,187,690   491,520,000    12.86%
stages.1.1.pw1_ttfs                                              pw1            335,105,317 1,966,080,000    17.04%
stages.1.1.pw2_ttfs                                              pw2            375,489,027   491,520,000    76.39%
stages.2.0.dwconv                                                dwconv          92,941,164   245,760,000    37.82%
stages.2.0.pw1_ttfs                                              pw1            185,647,275   983,040,000    18.89%
stages.2.0.pw2_ttfs                                              pw2            231,691,760   245,760,000    94.28%
stages.2.1.dwconv                                                dwconv          81,654,500   245,760,000    33.23%
stages.2.1.pw1_ttfs                                              pw1            229,850,703   983,040,000    23.38%
stages.2.1.pw2_ttfs                                              pw2            232,443,780   245,760,000    94.58%
stages.2.2.dwconv                                                dwconv          75,919,496   245,760,000    30.89%
stages.2.2.pw1_ttfs                                              pw1            256,524,468   983,040,000    26.10%
stages.2.2.pw2_ttfs                                              pw2            234,135,331   245,760,000    95.27%
stages.2.3.dwconv                                                dwconv          70,490,363   245,760,000    28.68%
stages.2.3.pw1_ttfs                                              pw1            256,397,530   983,040,000    26.08%
stages.2.3.pw2_ttfs                                              pw2            229,802,900   245,760,000    93.51%
stages.2.4.dwconv                                                dwconv          75,671,054   245,760,000    30.79%
stages.2.4.pw1_ttfs                                              pw1            279,949,084   983,040,000    28.48%
stages.2.4.pw2_ttfs                                              pw2            232,335,387   245,760,000    94.54%
stages.2.5.dwconv                                                dwconv          77,353,029   245,760,000    31.48%
stages.2.5.pw1_ttfs                                              pw1            290,893,794   983,040,000    29.59%
stages.2.5.pw2_ttfs                                              pw2            221,620,767   245,760,000    90.18%
stages.3.0.dwconv                                                dwconv          47,627,205   122,880,000    38.76%
stages.3.0.pw1_ttfs                                              pw1            220,085,548   491,520,000    44.78%
stages.3.0.pw2_ttfs                                              pw2            102,168,132   122,880,000    83.14%
stages.3.1.dwconv                                                dwconv          55,186,812   122,880,000    44.91%
stages.3.1.pw1_ttfs                                              pw1            240,177,155   491,520,000    48.86%
stages.3.1.pw2_ttfs                                              pw2             98,304,783   122,880,000    80.00%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     25.36%
===================================================================================================================

Classification accuracy: 93.59%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  16.23% | silent=758,017,246 | total=4,669,440,000
pw1             12 layers | sparsity=  16.61% | silent=3,102,855,274 | total=18,677,760,000
pw2             12 layers | sparsity=  64.52% | silent=3,012,873,918 | total=4,669,440,000
downsample       3 layers | sparsity=  52.25% | silent=449,446,042 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    9.49% | TTFS points= 6 | silent=1,119,301,928 | total=11,796,480,000
Stage 1:   28.33% | TTFS points= 7 | silent=1,809,906,770 | total=6,389,760,000
Stage 2:   38.86% | TTFS points=19 | silent=3,533,815,663 | total=9,093,120,000
Stage 3:   53.85% | TTFS points= 7 | silent=860,168,119 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.59%
Activation sparsity:  25.36%
TTFS layers/points:   39
================================================================================

Markdown report saved to: results\cifar10\test_downsample_ttfs_dwconv_ttfs\seed_42\activation_sparsity.md
```
