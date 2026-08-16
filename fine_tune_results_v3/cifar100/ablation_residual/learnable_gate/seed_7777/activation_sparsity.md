# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 78.12%
Batch   20/  79 | samples=  2560 | accuracy= 71.95%
Batch   40/  79 | samples=  5120 | accuracy= 71.04%
Batch   60/  79 | samples=  7680 | accuracy= 70.87%
Batch   79/  79 | samples= 10000 | accuracy= 71.58%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample     173,804,773   491,520,000    35.36%        40,544,639
downsample_layers.2.0                                            downsample     117,475,397   245,760,000    47.80%        27,770,884
downsample_layers.3.0                                            downsample      63,864,770   122,880,000    51.97%        33,597,510
stages.0.0.dwconv                                                dwconv           2,520,125   983,040,000     0.26%           842,379
stages.0.0.pw1_ttfs                                              pw1            929,267,346 3,932,160,000    23.63%        36,485,360
stages.0.0.pw2_ttfs                                              pw2            423,748,772   983,040,000    43.11%        28,827,769
stages.0.1.dwconv                                                dwconv          40,253,838   983,040,000     4.09%           843,994
stages.0.1.pw1_ttfs                                              pw1             84,698,492 3,932,160,000     2.15%        35,626,834
stages.0.1.pw2_ttfs                                              pw2            530,612,575   983,040,000    53.98%        36,935,630
stages.1.0.dwconv                                                dwconv         212,536,876   491,520,000    43.24%           262,368
stages.1.0.pw1_ttfs                                              pw1            748,204,489 1,966,080,000    38.06%        37,748,736
stages.1.0.pw2_ttfs                                              pw2            447,866,746   491,520,000    91.12%        23,383,210
stages.1.1.dwconv                                                dwconv         193,923,064   491,520,000    39.45%           286,161
stages.1.1.pw1_ttfs                                              pw1            854,713,201 1,966,080,000    43.47%        37,748,736
stages.1.1.pw2_ttfs                                              pw2            483,939,128   491,520,000    98.46%        21,338,243
stages.2.0.dwconv                                                dwconv         113,475,574   245,760,000    46.17%            96,984
stages.2.0.pw1_ttfs                                              pw1            312,873,094   983,040,000    31.83%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            220,341,819   245,760,000    89.66%        25,734,409
stages.2.1.dwconv                                                dwconv         114,200,825   245,760,000    46.47%           109,451
stages.2.1.pw1_ttfs                                              pw1            289,495,821   983,040,000    29.45%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            209,376,620   245,760,000    85.20%        26,632,096
stages.2.2.dwconv                                                dwconv         102,759,902   245,760,000    41.81%           123,354
stages.2.2.pw1_ttfs                                              pw1            249,370,700   983,040,000    25.37%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            173,821,141   245,760,000    70.73%        28,172,901
stages.2.3.dwconv                                                dwconv         111,298,522   245,760,000    45.29%           143,695
stages.2.3.pw1_ttfs                                              pw1            222,208,403   983,040,000    22.60%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            227,913,599   245,760,000    92.74%        29,215,933
stages.2.4.dwconv                                                dwconv          98,192,755   245,760,000    39.95%           147,454
stages.2.4.pw1_ttfs                                              pw1            118,431,524   983,040,000    12.05%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            137,580,109   245,760,000    55.98%        33,200,965
stages.2.5.dwconv                                                dwconv          99,143,406   245,760,000    40.34%           170,570
stages.2.5.pw1_ttfs                                              pw1            214,435,238   983,040,000    21.81%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            200,538,218   245,760,000    81.60%        29,514,423
stages.3.0.dwconv                                                dwconv          40,683,717   122,880,000    33.11%            37,241
stages.3.0.pw1_ttfs                                              pw1            255,823,755   491,520,000    52.05%        37,748,736
stages.3.0.pw2_ttfs                                              pw2             98,437,352   122,880,000    80.11%        18,101,472
stages.3.1.dwconv                                                dwconv          42,138,455   122,880,000    34.29%            46,868
stages.3.1.pw1_ttfs                                              pw1            270,798,678   491,520,000    55.09%        37,733,884
stages.3.1.pw2_ttfs                                              pw2             99,966,761   122,880,000    81.35%        16,951,398
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     32.31%
===================================================================================================================

Classification accuracy: 71.58%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 872,616,702
Layerwise SynOps total:    8,726,167,023,858

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  25.08% | silent=1,171,127,059 | total=4,669,440,000
pw1             12 layers | sparsity=  24.36% | silent=4,550,320,741 | total=18,677,760,000
pw2             12 layers | sparsity=  69.69% | silent=3,254,142,840 | total=4,669,440,000
downsample       3 layers | sparsity=  41.29% | silent=355,144,940 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   17.05% | TTFS points= 6 | silent=2,011,101,148 | total=11,796,480,000
Stage 1:   48.75% | TTFS points= 7 | silent=3,114,988,277 | total=6,389,760,000
Stage 2:   36.65% | TTFS points=19 | silent=3,332,932,667 | total=9,093,120,000
Stage 3:   54.57% | TTFS points= 7 | silent=871,713,488 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             71.58%
Activation sparsity:  32.31%
Theoretical SynOps:   872,616,702 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_residual\learnable_gate\seed_7777\activation_sparsity.md
```
