# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 77.34%
Batch   20/  79 | samples=  2560 | accuracy= 71.88%
Batch   40/  79 | samples=  5120 | accuracy= 71.02%
Batch   60/  79 | samples=  7680 | accuracy= 71.09%
Batch   79/  79 | samples= 10000 | accuracy= 71.45%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      95,561,276   491,520,000    19.44%        40,716,288
downsample_layers.2.0                                            downsample      96,208,370   245,760,000    39.15%        32,902,973
downsample_layers.3.0                                            downsample      62,290,178   122,880,000    50.69%        34,032,494
stages.0.0.dwconv                                                dwconv          18,167,368   983,040,000     1.85%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,056,105,499 3,932,160,000    26.86%        35,853,944
stages.0.0.pw2_ttfs                                              pw2            305,763,064   983,040,000    31.10%        27,610,123
stages.0.1.dwconv                                                dwconv          68,412,087   983,040,000     6.96%           848,256
stages.0.1.pw1_ttfs                                              pw1            165,212,880 3,932,160,000     4.20%        35,329,428
stages.0.1.pw2_ttfs                                              pw2            577,673,342   983,040,000    58.76%        36,162,692
stages.1.0.dwconv                                                dwconv         193,768,830   491,520,000    39.42%           326,172
stages.1.0.pw1_ttfs                                              pw1            828,037,253 1,966,080,000    42.12%        37,748,684
stages.1.0.pw2_ttfs                                              pw2            462,191,099   491,520,000    94.03%        21,850,421
stages.1.1.dwconv                                                dwconv         174,754,626   491,520,000    35.55%           337,273
stages.1.1.pw1_ttfs                                              pw1            542,229,883 1,966,080,000    27.58%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            465,444,231   491,520,000    94.69%        27,337,922
stages.2.0.dwconv                                                dwconv         111,513,692   245,760,000    45.38%           112,960
stages.2.0.pw1_ttfs                                              pw1            368,775,883   983,040,000    37.51%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            211,588,975   245,760,000    86.10%        23,587,742
stages.2.1.dwconv                                                dwconv         104,495,819   245,760,000    42.52%           125,400
stages.2.1.pw1_ttfs                                              pw1            244,169,056   983,040,000    24.84%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            165,125,395   245,760,000    67.19%        28,372,644
stages.2.2.dwconv                                                dwconv          97,515,310   245,760,000    39.68%           147,917
stages.2.2.pw1_ttfs                                              pw1            185,603,004   983,040,000    18.88%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            163,574,013   245,760,000    66.56%        30,621,581
stages.2.3.dwconv                                                dwconv         104,009,293   245,760,000    42.32%           163,754
stages.2.3.pw1_ttfs                                              pw1            175,561,763   983,040,000    17.86%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            190,506,031   245,760,000    77.52%        31,007,164
stages.2.4.dwconv                                                dwconv          96,357,007   245,760,000    39.21%           168,984
stages.2.4.pw1_ttfs                                              pw1            219,005,949   983,040,000    22.28%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            198,363,508   245,760,000    80.71%        29,338,908
stages.2.5.dwconv                                                dwconv          88,236,266   245,760,000    35.90%           173,254
stages.2.5.pw1_ttfs                                              pw1            227,802,141   983,040,000    23.17%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            197,829,531   245,760,000    80.50%        29,001,134
stages.3.0.dwconv                                                dwconv          41,506,861   122,880,000    33.78%            38,180
stages.3.0.pw1_ttfs                                              pw1            259,874,996   491,520,000    52.87%        37,748,722
stages.3.0.pw2_ttfs                                              pw2            100,099,846   122,880,000    81.46%        17,790,336
stages.3.1.dwconv                                                dwconv          40,985,322   122,880,000    33.35%            47,077
stages.3.1.pw1_ttfs                                              pw1            263,851,120   491,520,000    53.68%        37,735,606
stages.3.1.pw2_ttfs                                              pw2            100,857,288   122,880,000    82.08%        17,484,970
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     31.41%
===================================================================================================================

Classification accuracy: 71.45%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 879,812,414
Layerwise SynOps total:    8,798,124,137,338

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  24.41% | silent=1,139,722,481 | total=4,669,440,000
pw1             12 layers | sparsity=  24.29% | silent=4,536,229,427 | total=18,677,760,000
pw2             12 layers | sparsity=  67.22% | silent=3,139,016,323 | total=4,669,440,000
downsample       3 layers | sparsity=  29.54% | silent=254,059,824 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   18.58% | TTFS points= 6 | silent=2,191,334,240 | total=11,796,480,000
Stage 1:   43.23% | TTFS points= 7 | silent=2,761,987,198 | total=6,389,760,000
Stage 2:   35.70% | TTFS points=19 | silent=3,246,241,006 | total=9,093,120,000
Stage 3:   54.43% | TTFS points= 7 | silent=869,465,611 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             71.45%
Activation sparsity:  31.41%
Theoretical SynOps:   879,812,414 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_42\activation_sparsity.md
```
