# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_7777\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: mean (metadata)
Detected non-negative effective pointwise weights: False (metadata)

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
Batch   20/  79 | samples=  2560 | accuracy= 72.11%
Batch   40/  79 | samples=  5120 | accuracy= 70.96%
Batch   60/  79 | samples=  7680 | accuracy= 71.26%
Batch   79/  79 | samples= 10000 | accuracy= 71.79%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     169,990,853   491,520,000    34.58%        40,695,842
downsample_layers.2.0                                            downsample     121,919,111   245,760,000    49.61%        28,347,836
downsample_layers.3.0                                            downsample      64,439,433   122,880,000    52.44%        33,743,289
stages.0.0.dwconv                                                dwconv           2,224,251   983,040,000     0.23%           847,689
stages.0.0.pw1_ttfs                                              pw1            920,197,298 3,932,160,000    23.40%        36,488,286
stages.0.0.pw2_ttfs                                              pw2            445,062,211   983,040,000    45.27%        28,914,842
stages.0.1.dwconv                                                dwconv          31,123,584   983,040,000     3.17%           847,689
stages.0.1.pw1_ttfs                                              pw1             99,286,742 3,932,160,000     2.52%        36,071,579
stages.0.1.pw2_ttfs                                              pw2            550,301,383   983,040,000    55.98%        36,795,583
stages.1.0.dwconv                                                dwconv         228,572,957   491,520,000    46.50%           265,205
stages.1.0.pw1_ttfs                                              pw1            728,528,611 1,966,080,000    37.05%        37,748,736
stages.1.0.pw2_ttfs                                              pw2            449,015,848   491,520,000    91.35%        23,760,987
stages.1.1.dwconv                                                dwconv         171,131,888   491,520,000    34.82%           285,762
stages.1.1.pw1_ttfs                                              pw1            490,274,189 1,966,080,000    24.94%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            446,893,379   491,520,000    90.92%        28,335,472
stages.2.0.dwconv                                                dwconv         120,427,305   245,760,000    49.00%            93,628
stages.2.0.pw1_ttfs                                              pw1            308,656,089   983,040,000    31.40%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            220,264,357   245,760,000    89.63%        25,896,342
stages.2.1.dwconv                                                dwconv         110,617,436   245,760,000    45.01%           104,192
stages.2.1.pw1_ttfs                                              pw1            261,690,283   983,040,000    26.62%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            207,225,791   245,760,000    84.32%        27,699,829
stages.2.2.dwconv                                                dwconv         103,652,086   245,760,000    42.18%           120,147
stages.2.2.pw1_ttfs                                              pw1            274,903,920   983,040,000    27.96%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            172,516,441   245,760,000    70.20%        27,192,425
stages.2.3.dwconv                                                dwconv          99,580,892   245,760,000    40.52%           143,821
stages.2.3.pw1_ttfs                                              pw1            215,738,877   983,040,000    21.95%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            223,858,548   245,760,000    91.09%        29,464,363
stages.2.4.dwconv                                                dwconv          97,639,405   245,760,000    39.73%           149,567
stages.2.4.pw1_ttfs                                              pw1            107,038,306   983,040,000    10.89%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            123,209,280   245,760,000    50.13%        33,638,465
stages.2.5.dwconv                                                dwconv          95,069,205   245,760,000    38.68%           170,654
stages.2.5.pw1_ttfs                                              pw1            215,151,519   983,040,000    21.89%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            201,946,154   245,760,000    82.17%        29,486,918
stages.3.0.dwconv                                                dwconv          39,498,316   122,880,000    32.14%            36,889
stages.3.0.pw1_ttfs                                              pw1            254,697,176   491,520,000    51.82%        37,748,734
stages.3.0.pw2_ttfs                                              pw2             97,949,015   122,880,000    79.71%        18,187,993
stages.3.1.dwconv                                                dwconv          41,618,669   122,880,000    33.87%            46,836
stages.3.1.pw1_ttfs                                              pw1            269,121,453   491,520,000    54.75%        37,725,106
stages.3.1.pw2_ttfs                                              pw2             99,293,673   122,880,000    80.81%        17,080,208
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     30.75%
===================================================================================================================

Classification accuracy: 71.79%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 882,376,066
Layerwise SynOps total:    8,823,760,663,019

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  24.44% | silent=1,141,155,994 | total=4,669,440,000
pw1             12 layers | sparsity=  22.19% | silent=4,145,284,463 | total=18,677,760,000
pw2             12 layers | sparsity=  69.33% | silent=3,237,536,080 | total=4,669,440,000
downsample       3 layers | sparsity=  41.43% | silent=356,349,397 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   17.36% | TTFS points= 6 | silent=2,048,195,469 | total=11,796,480,000
Stage 1:   42.01% | TTFS points= 7 | silent=2,684,407,725 | total=6,389,760,000
Stage 2:   36.08% | TTFS points=19 | silent=3,281,105,005 | total=9,093,120,000
Stage 3:   54.25% | TTFS points= 7 | silent=866,617,735 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             71.79%
Activation sparsity:  30.75%
Theoretical SynOps:   882,376,066 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_7777\activation_sparsity.md
```
