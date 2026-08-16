# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_42\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 75.78%
Batch   20/  79 | samples=  2560 | accuracy= 73.24%
Batch   40/  79 | samples=  5120 | accuracy= 71.80%
Batch   60/  79 | samples=  7680 | accuracy= 71.89%
Batch   79/  79 | samples= 10000 | accuracy= 72.36%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      89,848,152   491,520,000    18.28%        40,716,288
downsample_layers.2.0                                            downsample      91,147,017   245,760,000    37.09%        33,486,065
downsample_layers.3.0                                            downsample      61,601,924   122,880,000    50.13%        34,660,831
stages.0.0.dwconv                                                dwconv          19,902,724   983,040,000     2.02%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,095,973,089 3,932,160,000    27.87%        35,747,021
stages.0.0.pw2_ttfs                                              pw2            311,222,435   983,040,000    31.66%        27,227,394
stages.0.1.dwconv                                                dwconv         105,619,365   983,040,000    10.74%           848,256
stages.0.1.pw1_ttfs                                              pw1            126,932,989 3,932,160,000     3.23%        36,289,819
stages.0.1.pw2_ttfs                                              pw2            623,885,311   983,040,000    63.46%        36,530,179
stages.1.0.dwconv                                                dwconv         204,383,247   491,520,000    41.58%           331,293
stages.1.0.pw1_ttfs                                              pw1            892,044,495 1,966,080,000    45.37%        37,748,477
stages.1.0.pw2_ttfs                                              pw2            462,598,059   491,520,000    94.12%        20,621,482
stages.1.1.dwconv                                                dwconv         203,602,393   491,520,000    41.42%           341,959
stages.1.1.pw1_ttfs                                              pw1            577,785,963 1,966,080,000    29.39%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            471,127,215   491,520,000    95.85%        26,655,246
stages.2.0.dwconv                                                dwconv         101,080,334   245,760,000    41.13%           116,836
stages.2.0.pw1_ttfs                                              pw1            368,839,943   983,040,000    37.52%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            207,912,377   245,760,000    84.60%        23,585,282
stages.2.1.dwconv                                                dwconv         105,659,721   245,760,000    42.99%           131,288
stages.2.1.pw1_ttfs                                              pw1            362,581,950   983,040,000    36.88%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            203,372,000   245,760,000    82.75%        23,825,589
stages.2.2.dwconv                                                dwconv          83,537,707   245,760,000    33.99%           144,464
stages.2.2.pw1_ttfs                                              pw1             94,900,155   983,040,000     9.65%        37,748,736
stages.2.2.pw2_ttfs                                              pw2             69,990,800   245,760,000    28.48%        34,104,570
stages.2.3.dwconv                                                dwconv         113,469,078   245,760,000    46.17%           174,527
stages.2.3.pw1_ttfs                                              pw1            271,360,790   983,040,000    27.60%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            207,106,101   245,760,000    84.27%        27,328,482
stages.2.4.dwconv                                                dwconv         105,920,339   245,760,000    43.10%           177,094
stages.2.4.pw1_ttfs                                              pw1            326,613,237   983,040,000    33.22%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            219,087,639   245,760,000    89.15%        25,206,788
stages.2.5.dwconv                                                dwconv          91,580,940   245,760,000    37.26%           178,589
stages.2.5.pw1_ttfs                                              pw1            273,137,062   983,040,000    27.78%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            205,327,766   245,760,000    83.55%        27,260,273
stages.3.0.dwconv                                                dwconv          39,743,831   122,880,000    32.34%            38,588
stages.3.0.pw1_ttfs                                              pw1            260,579,155   491,520,000    53.01%        37,748,462
stages.3.0.pw2_ttfs                                              pw2             99,846,615   122,880,000    81.26%        17,736,257
stages.3.1.dwconv                                                dwconv          40,145,898   122,880,000    32.67%            47,529
stages.3.1.pw1_ttfs                                              pw1            263,887,448   491,520,000    53.69%        37,719,142
stages.3.1.pw2_ttfs                                              pw2            100,464,339   122,880,000    81.76%        17,482,180
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     33.08%
===================================================================================================================

Classification accuracy: 72.36%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 869,299,657
Layerwise SynOps total:    8,692,996,568,302

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  26.01% | silent=1,214,645,577 | total=4,669,440,000
pw1             12 layers | sparsity=  26.31% | silent=4,914,636,276 | total=18,677,760,000
pw2             12 layers | sparsity=  68.14% | silent=3,181,940,657 | total=4,669,440,000
downsample       3 layers | sparsity=  28.20% | silent=242,597,093 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   19.36% | TTFS points= 6 | silent=2,283,535,913 | total=11,796,480,000
Stage 1:   45.41% | TTFS points= 7 | silent=2,901,389,524 | total=6,389,760,000
Stage 2:   38.52% | TTFS points=19 | silent=3,502,624,956 | total=9,093,120,000
Stage 3:   54.23% | TTFS points= 7 | silent=866,269,210 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.36%
Activation sparsity:  33.08%
Theoretical SynOps:   869,299,657 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\mean\seed_42\activation_sparsity.md
```
