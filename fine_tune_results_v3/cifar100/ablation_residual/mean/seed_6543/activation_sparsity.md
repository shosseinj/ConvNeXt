# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_6543\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 69.53%
Batch   20/  79 | samples=  2560 | accuracy= 70.51%
Batch   40/  79 | samples=  5120 | accuracy= 70.33%
Batch   60/  79 | samples=  7680 | accuracy= 69.82%
Batch   79/  79 | samples= 10000 | accuracy= 69.96%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     195,164,157   491,520,000    39.71%        40,716,288
downsample_layers.2.0                                            downsample     133,140,995   245,760,000    54.18%        25,907,865
downsample_layers.3.0                                            downsample      61,800,100   122,880,000    50.29%        31,977,127
stages.0.0.dwconv                                                dwconv           5,027,735   983,040,000     0.51%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,569,594,037 3,932,160,000    39.92%        36,109,882
stages.0.0.pw2_ttfs                                              pw2            326,198,844   983,040,000    33.18%        22,680,633
stages.0.1.dwconv                                                dwconv          58,992,223   983,040,000     6.00%           848,256
stages.0.1.pw1_ttfs                                              pw1          1,001,597,608 3,932,160,000    25.47%        35,135,443
stages.0.1.pw2_ttfs                                              pw2            625,384,580   983,040,000    63.62%        28,133,399
stages.1.0.dwconv                                                dwconv         212,550,228   491,520,000    43.24%           244,861
stages.1.0.pw1_ttfs                                              pw1            832,066,774 1,966,080,000    42.32%        37,748,591
stages.1.0.pw2_ttfs                                              pw2            460,962,122   491,520,000    93.78%        21,773,054
stages.1.1.dwconv                                                dwconv         202,320,487   491,520,000    41.16%           251,758
stages.1.1.pw1_ttfs                                              pw1            425,297,695 1,966,080,000    21.63%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            440,222,512   491,520,000    89.56%        29,583,020
stages.2.0.dwconv                                                dwconv         108,556,070   245,760,000    44.17%            85,178
stages.2.0.pw1_ttfs                                              pw1            419,286,974   983,040,000    42.65%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            232,331,711   245,760,000    94.54%        21,648,116
stages.2.1.dwconv                                                dwconv         112,986,001   245,760,000    45.97%            93,629
stages.2.1.pw1_ttfs                                              pw1            317,712,225   983,040,000    32.32%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            210,385,524   245,760,000    85.61%        25,548,587
stages.2.2.dwconv                                                dwconv         104,271,528   245,760,000    42.43%           112,644
stages.2.2.pw1_ttfs                                              pw1            366,075,543   983,040,000    37.24%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            237,947,844   245,760,000    96.82%        23,691,435
stages.2.3.dwconv                                                dwconv         101,208,929   245,760,000    41.18%           115,651
stages.2.3.pw1_ttfs                                              pw1            238,771,653   983,040,000    24.29%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            204,699,986   245,760,000    83.29%        28,579,905
stages.2.4.dwconv                                                dwconv          95,097,823   245,760,000    38.70%           129,350
stages.2.4.pw1_ttfs                                              pw1            170,350,222   983,040,000    17.33%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            101,509,551   245,760,000    41.30%        31,207,287
stages.2.5.dwconv                                                dwconv         102,645,057   245,760,000    41.77%           165,282
stages.2.5.pw1_ttfs                                              pw1            372,099,814   983,040,000    37.85%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            234,525,683   245,760,000    95.43%        23,460,103
stages.3.0.dwconv                                                dwconv          43,814,289   122,880,000    35.66%            38,424
stages.3.0.pw1_ttfs                                              pw1            259,704,390   491,520,000    52.84%        37,748,736
stages.3.0.pw2_ttfs                                              pw2            101,942,417   122,880,000    82.96%        17,803,439
stages.3.1.dwconv                                                dwconv          43,474,147   122,880,000    35.38%            46,666
stages.3.1.pw1_ttfs                                              pw1            257,566,800   491,520,000    52.40%        37,746,387
stages.3.1.pw2_ttfs                                              pw2             98,877,035   122,880,000    80.47%        17,967,606
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     38.39%
===================================================================================================================

Classification accuracy: 69.96%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 842,388,008
Layerwise SynOps total:    8,423,880,078,802

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  25.51% | silent=1,190,944,517 | total=4,669,440,000
pw1             12 layers | sparsity=  33.36% | silent=6,230,123,735 | total=18,677,760,000
pw2             12 layers | sparsity=  70.14% | silent=3,274,987,809 | total=4,669,440,000
downsample       3 layers | sparsity=  45.35% | silent=390,105,252 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   30.41% | TTFS points= 6 | silent=3,586,795,027 | total=11,796,480,000
Stage 1:   43.33% | TTFS points= 7 | silent=2,768,583,975 | total=6,389,760,000
Stage 2:   42.49% | TTFS points=19 | silent=3,863,603,133 | total=9,093,120,000
Stage 3:   54.29% | TTFS points= 7 | silent=867,179,178 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             69.96%
Activation sparsity:  38.39%
Theoretical SynOps:   842,388,008 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_6543\activation_sparsity.md
```
