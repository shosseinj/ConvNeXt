# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar10
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)

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
Batch    1/  79 | samples=   128 | accuracy= 95.31%
Batch   20/  79 | samples=  2560 | accuracy= 93.79%
Batch   40/  79 | samples=  5120 | accuracy= 93.36%
Batch   60/  79 | samples=  7680 | accuracy= 93.79%
Batch   79/  79 | samples= 10000 | accuracy= 93.86%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR10
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      33,068,662   491,520,000     6.73%
downsample_layers.2.0                                            downsample      62,092,675   245,760,000    25.27%
downsample_layers.3.0                                            downsample      77,578,116   122,880,000    63.13%
stages.0.0.dwconv                                                dwconv             161,555   983,040,000     0.02%
stages.0.0.pw1_ttfs                                              pw1                549,345 3,932,160,000     0.01%
stages.0.0.pw2_ttfs                                              pw2            120,540,790   983,040,000    12.26%
stages.0.1.dwconv                                                dwconv          11,169,039   983,040,000     1.14%
stages.0.1.pw1_ttfs                                              pw1          1,642,408,071 3,932,160,000    41.77%
stages.0.1.pw2_ttfs                                              pw2            187,573,536   983,040,000    19.08%
stages.1.0.dwconv                                                dwconv          47,720,973   491,520,000     9.71%
stages.1.0.pw1_ttfs                                              pw1            883,444,256 1,966,080,000    44.93%
stages.1.0.pw2_ttfs                                              pw2            476,708,420   491,520,000    96.99%
stages.1.1.dwconv                                                dwconv          47,150,704   491,520,000     9.59%
stages.1.1.pw1_ttfs                                              pw1            774,036,045 1,966,080,000    39.37%
stages.1.1.pw2_ttfs                                              pw2            475,868,478   491,520,000    96.82%
stages.2.0.dwconv                                                dwconv          69,939,824   245,760,000    28.46%
stages.2.0.pw1_ttfs                                              pw1            464,674,860   983,040,000    47.27%
stages.2.0.pw2_ttfs                                              pw2            238,062,813   245,760,000    96.87%
stages.2.1.dwconv                                                dwconv          81,471,865   245,760,000    33.15%
stages.2.1.pw1_ttfs                                              pw1            482,729,064   983,040,000    49.11%
stages.2.1.pw2_ttfs                                              pw2            237,774,367   245,760,000    96.75%
stages.2.2.dwconv                                                dwconv          82,292,282   245,760,000    33.48%
stages.2.2.pw1_ttfs                                              pw1            527,896,286   983,040,000    53.70%
stages.2.2.pw2_ttfs                                              pw2            241,090,196   245,760,000    98.10%
stages.2.3.dwconv                                                dwconv          79,601,447   245,760,000    32.39%
stages.2.3.pw1_ttfs                                              pw1            503,887,244   983,040,000    51.26%
stages.2.3.pw2_ttfs                                              pw2            235,033,224   245,760,000    95.64%
stages.2.4.dwconv                                                dwconv          92,562,783   245,760,000    37.66%
stages.2.4.pw1_ttfs                                              pw1            469,163,865   983,040,000    47.73%
stages.2.4.pw2_ttfs                                              pw2            235,938,869   245,760,000    96.00%
stages.2.5.dwconv                                                dwconv          92,483,923   245,760,000    37.63%
stages.2.5.pw1_ttfs                                              pw1            524,626,720   983,040,000    53.37%
stages.2.5.pw2_ttfs                                              pw2            237,696,653   245,760,000    96.72%
stages.3.0.dwconv                                                dwconv          35,116,063   122,880,000    28.58%
stages.3.0.pw1_ttfs                                              pw1            246,623,443   491,520,000    50.18%
stages.3.0.pw2_ttfs                                              pw2            108,797,272   122,880,000    88.54%
stages.3.1.dwconv                                                dwconv          47,454,010   122,880,000    38.62%
stages.3.1.pw1_ttfs                                              pw1            255,030,369   491,520,000    51.89%
stages.3.1.pw2_ttfs                                              pw2            105,418,287   122,880,000    85.79%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     36.48%
===================================================================================================================

Classification accuracy: 93.86%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.72% | silent=687,124,468 | total=4,669,440,000
pw1             12 layers | sparsity=  36.27% | silent=6,775,069,568 | total=18,677,760,000
pw2             12 layers | sparsity=  62.12% | silent=2,900,502,905 | total=4,669,440,000
downsample       3 layers | sparsity=  20.08% | silent=172,739,453 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   16.64% | TTFS points= 6 | silent=1,962,402,336 | total=11,796,480,000
Stage 1:   42.85% | TTFS points= 7 | silent=2,737,997,538 | total=6,389,760,000
Stage 2:   54.54% | TTFS points=19 | silent=4,959,018,960 | total=9,093,120,000
Stage 3:   54.84% | TTFS points= 7 | silent=876,017,560 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar10
Accuracy:             93.86%
Activation sparsity:  36.48%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar10\fully_ttfs\seed_6543\activation_sparsity.md
```
