# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v4\cifar100\ann_to_fully_ttfs\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)
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
Batch    1/  79 | samples=   128 | accuracy= 73.44%
Batch   20/  79 | samples=  2560 | accuracy= 69.14%
Batch   40/  79 | samples=  5120 | accuracy= 68.20%
Batch   60/  79 | samples=  7680 | accuracy= 68.68%
Batch   79/  79 | samples= 10000 | accuracy= 68.80%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     277,888,226   491,520,000    56.54%        40,159,522
downsample_layers.2.0                                            downsample     181,420,376   245,760,000    73.82%        29,601,304
downsample_layers.3.0                                            downsample      97,503,683   122,880,000    79.35%        28,376,397
stages.0.0.dwconv                                                dwconv         162,755,812   983,040,000    16.56%           811,107
stages.0.0.pw1_ttfs                                              pw1          1,265,587,308 3,932,160,000    32.19%        35,355,378
stages.0.0.pw2_ttfs                                              pw2            626,806,659   983,040,000    63.76%        25,599,098
stages.0.1.dwconv                                                dwconv         178,688,993   983,040,000    18.18%           828,962
stages.0.1.pw1_ttfs                                              pw1          1,414,301,412 3,932,160,000    35.97%        33,617,010
stages.0.1.pw2_ttfs                                              pw2            527,955,712   983,040,000    53.71%        24,171,442
stages.1.0.dwconv                                                dwconv         229,205,427   491,520,000    46.63%           174,459
stages.1.0.pw1_ttfs                                              pw1            643,978,690 1,966,080,000    32.75%        37,146,850
stages.1.0.pw2_ttfs                                              pw2            344,283,041   491,520,000    70.04%        25,384,345
stages.1.1.dwconv                                                dwconv         238,425,105   491,520,000    48.51%           254,578
stages.1.1.pw1_ttfs                                              pw1            598,862,671 1,966,080,000    30.46%        36,570,924
stages.1.1.pw2_ttfs                                              pw2            336,592,215   491,520,000    68.48%        26,250,573
stages.2.0.dwconv                                                dwconv         120,332,658   245,760,000    48.96%            47,909
stages.2.0.pw1_ttfs                                              pw1            263,711,296   983,040,000    26.83%        37,650,501
stages.2.0.pw2_ttfs                                              pw2            209,271,669   245,760,000    85.15%        27,622,222
stages.2.1.dwconv                                                dwconv         120,876,026   245,760,000    49.18%            67,246
stages.2.1.pw1_ttfs                                              pw1            303,753,234   983,040,000    30.90%        37,704,284
stages.2.1.pw2_ttfs                                              pw2            211,776,990   245,760,000    86.17%        26,084,612
stages.2.2.dwconv                                                dwconv         122,023,822   245,760,000    49.65%            82,425
stages.2.2.pw1_ttfs                                              pw1            308,740,129   983,040,000    31.41%        37,748,721
stages.2.2.pw2_ttfs                                              pw2            203,529,982   245,760,000    82.82%        25,893,115
stages.2.3.dwconv                                                dwconv         118,320,163   245,760,000    48.14%            99,002
stages.2.3.pw1_ttfs                                              pw1            315,655,694   983,040,000    32.11%        37,719,270
stages.2.3.pw2_ttfs                                              pw2            194,983,862   245,760,000    79.34%        25,627,557
stages.2.4.dwconv                                                dwconv         120,436,606   245,760,000    49.01%           116,173
stages.2.4.pw1_ttfs                                              pw1            327,925,763   983,040,000    33.36%        37,735,196
stages.2.4.pw2_ttfs                                              pw2            190,545,652   245,760,000    77.53%        25,156,387
stages.2.5.dwconv                                                dwconv         116,130,011   245,760,000    47.25%           132,270
stages.2.5.pw1_ttfs                                              pw1            324,594,058   983,040,000    33.02%        37,604,255
stages.2.5.pw2_ttfs                                              pw2            178,314,793   245,760,000    72.56%        25,284,324
stages.3.0.dwconv                                                dwconv          63,491,860   122,880,000    51.67%            16,242
stages.3.0.pw1_ttfs                                              pw1            290,758,397   491,520,000    59.15%        37,553,635
stages.3.0.pw2_ttfs                                              pw2             77,315,363   122,880,000    62.92%        15,418,491
stages.3.1.dwconv                                                dwconv          59,936,806   122,880,000    48.78%            39,825
stages.3.1.pw1_ttfs                                              pw1            318,277,977   491,520,000    64.75%        35,307,534
stages.3.1.pw2_ttfs                                              pw2             54,468,016   122,880,000    44.33%        13,304,987
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     40.65%
===================================================================================================================

Classification accuracy: 68.80%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 828,318,135
Layerwise SynOps total:    8,283,181,354,886

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  35.35% | silent=1,650,623,289 | total=4,669,440,000
pw1             12 layers | sparsity=  34.14% | silent=6,376,146,629 | total=18,677,760,000
pw2             12 layers | sparsity=  67.59% | silent=3,155,843,954 | total=4,669,440,000
downsample       3 layers | sparsity=  64.73% | silent=556,812,285 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   35.40% | TTFS points= 6 | silent=4,176,095,896 | total=11,796,480,000
Stage 1:   41.77% | TTFS points= 7 | silent=2,669,235,375 | total=6,389,760,000
Stage 2:   43.25% | TTFS points=19 | silent=3,932,342,784 | total=9,093,120,000
Stage 3:   60.21% | TTFS points= 7 | silent=961,752,102 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             68.80%
Activation sparsity:  40.65%
Theoretical SynOps:   828,318,135 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v4\cifar100\ann_to_fully_ttfs\seed_42\activation_sparsity.md
```
