# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: learnable_gate (metadata)
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
Batch   20/  79 | samples=  2560 | accuracy= 71.17%
Batch   40/  79 | samples=  5120 | accuracy= 71.13%
Batch   60/  79 | samples=  7680 | accuracy= 70.89%
Batch   79/  79 | samples= 10000 | accuracy= 70.96%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     162,452,566   491,520,000    33.05%        40,716,288
downsample_layers.2.0                                            downsample     129,980,194   245,760,000    52.89%        27,493,267
downsample_layers.3.0                                            downsample      63,165,328   122,880,000    51.40%        31,951,231
stages.0.0.dwconv                                                dwconv           6,921,483   983,040,000     0.70%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,317,180,698 3,932,160,000    33.50%        36,260,692
stages.0.0.pw2_ttfs                                              pw2            283,904,116   983,040,000    28.88%        25,103,801
stages.0.1.dwconv                                                dwconv          42,003,397   983,040,000     4.27%           848,256
stages.0.1.pw1_ttfs                                              pw1          1,495,182,602 3,932,160,000    38.02%        35,216,338
stages.0.1.pw2_ttfs                                              pw2            644,849,936   983,040,000    65.60%        23,394,983
stages.1.0.dwconv                                                dwconv         177,622,607   491,520,000    36.14%           271,711
stages.1.0.pw1_ttfs                                              pw1            886,376,372 1,966,080,000    45.08%        37,738,260
stages.1.0.pw2_ttfs                                              pw2            466,109,304   491,520,000    94.83%        20,730,310
stages.1.1.dwconv                                                dwconv         185,286,826   491,520,000    37.70%           278,038
stages.1.1.pw1_ttfs                                              pw1            640,544,348 1,966,080,000    32.58%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            462,757,777   491,520,000    94.15%        25,450,285
stages.2.0.dwconv                                                dwconv         108,107,025   245,760,000    43.99%            87,564
stages.2.0.pw1_ttfs                                              pw1            433,401,682   983,040,000    44.09%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            231,623,134   245,760,000    94.25%        21,106,111
stages.2.1.dwconv                                                dwconv         108,762,980   245,760,000    44.26%            96,010
stages.2.1.pw1_ttfs                                              pw1            309,843,882   983,040,000    31.52%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            201,858,254   245,760,000    82.14%        25,850,731
stages.2.2.dwconv                                                dwconv         109,569,403   245,760,000    44.58%           117,413
stages.2.2.pw1_ttfs                                              pw1            363,595,300   983,040,000    36.99%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            238,142,410   245,760,000    96.90%        23,786,676
stages.2.3.dwconv                                                dwconv          97,806,081   245,760,000    39.80%           120,261
stages.2.3.pw1_ttfs                                              pw1            246,456,231   983,040,000    25.07%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            203,044,155   245,760,000    82.62%        28,284,817
stages.2.4.dwconv                                                dwconv          83,468,937   245,760,000    33.96%           130,696
stages.2.4.pw1_ttfs                                              pw1            190,646,172   983,040,000    19.39%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            106,947,671   245,760,000    43.52%        30,427,923
stages.2.5.dwconv                                                dwconv         106,975,081   245,760,000    43.53%           164,668
stages.2.5.pw1_ttfs                                              pw1            358,301,078   983,040,000    36.45%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            231,002,156   245,760,000    94.00%        23,989,975
stages.3.0.dwconv                                                dwconv          43,360,962   122,880,000    35.29%            37,630
stages.3.0.pw1_ttfs                                              pw1            259,014,105   491,520,000    52.70%        37,748,736
stages.3.0.pw2_ttfs                                              pw2            101,382,403   122,880,000    82.51%        17,856,453
stages.3.1.dwconv                                                dwconv          43,223,194   122,880,000    35.18%            46,202
stages.3.1.pw1_ttfs                                              pw1            259,421,065   491,520,000    52.78%        37,747,025
stages.3.1.pw2_ttfs                                              pw2             99,330,505   122,880,000    80.84%        17,825,198
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     39.82%
===================================================================================================================

Classification accuracy: 70.96%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 835,966,957
Layerwise SynOps total:    8,359,669,565,654

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  23.84% | silent=1,113,107,976 | total=4,669,440,000
pw1             12 layers | sparsity=  36.19% | silent=6,759,963,535 | total=18,677,760,000
pw2             12 layers | sparsity=  70.05% | silent=3,270,951,821 | total=4,669,440,000
downsample       3 layers | sparsity=  41.34% | silent=355,598,088 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   32.13% | TTFS points= 6 | silent=3,790,042,232 | total=11,796,480,000
Stage 1:   46.66% | TTFS points= 7 | silent=2,981,149,800 | total=6,389,760,000
Stage 2:   42.44% | TTFS points=19 | silent=3,859,531,826 | total=9,093,120,000
Stage 3:   54.39% | TTFS points= 7 | silent=868,897,562 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             70.96%
Activation sparsity:  39.82%
Theoretical SynOps:   835,966,957 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_6543\activation_sparsity.md
```
