# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_7777\best_checkpoint.pth

Device: cuda
Dataset: cifar100
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
Batch    1/  79 | samples=   128 | accuracy= 78.12%
Batch   20/  79 | samples=  2560 | accuracy= 73.05%
Batch   40/  79 | samples=  5120 | accuracy= 72.07%
Batch   60/  79 | samples=  7680 | accuracy= 71.99%
Batch   79/  79 | samples= 10000 | accuracy= 72.43%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      26,552,331   491,520,000     5.40%
downsample_layers.2.0                                            downsample      70,590,303   245,760,000    28.72%
downsample_layers.3.0                                            downsample      77,441,930   122,880,000    63.02%
stages.0.0.dwconv                                                dwconv              23,888   983,040,000     0.00%
stages.0.0.pw1_ttfs                                              pw1             12,105,788 3,932,160,000     0.31%
stages.0.0.pw2_ttfs                                              pw2            112,653,136   983,040,000    11.46%
stages.0.1.dwconv                                                dwconv              10,000   983,040,000     0.00%
stages.0.1.pw1_ttfs                                              pw1                691,469 3,932,160,000     0.02%
stages.0.1.pw2_ttfs                                              pw2            162,542,717   983,040,000    16.53%
stages.1.0.dwconv                                                dwconv          42,614,634   491,520,000     8.67%
stages.1.0.pw1_ttfs                                              pw1            911,643,955 1,966,080,000    46.37%
stages.1.0.pw2_ttfs                                              pw2            476,840,431   491,520,000    97.01%
stages.1.1.dwconv                                                dwconv          47,118,456   491,520,000     9.59%
stages.1.1.pw1_ttfs                                              pw1            805,381,592 1,966,080,000    40.96%
stages.1.1.pw2_ttfs                                              pw2            486,886,594   491,520,000    99.06%
stages.2.0.dwconv                                                dwconv          58,485,255   245,760,000    23.80%
stages.2.0.pw1_ttfs                                              pw1            371,172,823   983,040,000    37.76%
stages.2.0.pw2_ttfs                                              pw2            228,740,318   245,760,000    93.07%
stages.2.1.dwconv                                                dwconv          78,400,239   245,760,000    31.90%
stages.2.1.pw1_ttfs                                              pw1            436,080,535   983,040,000    44.36%
stages.2.1.pw2_ttfs                                              pw2            239,101,082   245,760,000    97.29%
stages.2.2.dwconv                                                dwconv          79,787,650   245,760,000    32.47%
stages.2.2.pw1_ttfs                                              pw1            453,339,189   983,040,000    46.12%
stages.2.2.pw2_ttfs                                              pw2            232,618,337   245,760,000    94.65%
stages.2.3.dwconv                                                dwconv          97,899,751   245,760,000    39.84%
stages.2.3.pw1_ttfs                                              pw1            449,871,450   983,040,000    45.76%
stages.2.3.pw2_ttfs                                              pw2            243,004,906   245,760,000    98.88%
stages.2.4.dwconv                                                dwconv          95,925,295   245,760,000    39.03%
stages.2.4.pw1_ttfs                                              pw1            442,029,931   983,040,000    44.97%
stages.2.4.pw2_ttfs                                              pw2            237,087,648   245,760,000    96.47%
stages.2.5.dwconv                                                dwconv          94,880,227   245,760,000    38.61%
stages.2.5.pw1_ttfs                                              pw1            470,284,477   983,040,000    47.84%
stages.2.5.pw2_ttfs                                              pw2            239,436,170   245,760,000    97.43%
stages.3.0.dwconv                                                dwconv          32,780,360   122,880,000    26.68%
stages.3.0.pw1_ttfs                                              pw1            254,443,329   491,520,000    51.77%
stages.3.0.pw2_ttfs                                              pw2            102,239,433   122,880,000    83.20%
stages.3.1.dwconv                                                dwconv          45,328,385   122,880,000    36.89%
stages.3.1.pw1_ttfs                                              pw1            262,329,872   491,520,000    53.37%
stages.3.1.pw2_ttfs                                              pw2             90,079,276   122,880,000    73.31%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     29.67%
===================================================================================================================

Classification accuracy: 72.43%
Measured TTFS points:     39
Expected TTFS points:     39

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.42% | silent=673,254,140 | total=4,669,440,000
pw1             12 layers | sparsity=  26.07% | silent=4,869,374,410 | total=18,677,760,000
pw2             12 layers | sparsity=  61.06% | silent=2,851,230,048 | total=4,669,440,000
downsample       3 layers | sparsity=  20.30% | silent=174,584,564 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    2.44% | TTFS points= 6 | silent=288,026,998 | total=11,796,480,000
Stage 1:   43.77% | TTFS points= 7 | silent=2,797,037,993 | total=6,389,760,000
Stage 2:   50.79% | TTFS points=19 | silent=4,618,735,586 | total=9,093,120,000
Stage 3:   54.13% | TTFS points= 7 | silent=864,642,585 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.43%
Activation sparsity:  29.67%
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_7777\activation_sparsity.md
```
