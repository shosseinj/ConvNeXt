# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)
Detected pointwise weight parameterization: softplus (metadata)

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
Batch    1/  79 | samples=   128 | accuracy= 57.03%
Batch   20/  79 | samples=  2560 | accuracy= 46.52%
Batch   40/  79 | samples=  5120 | accuracy= 45.84%
Batch   60/  79 | samples=  7680 | accuracy= 45.31%
Batch   79/  79 | samples= 10000 | accuracy= 45.17%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      60,109,836   491,520,000    12.23%        40,716,288
downsample_layers.2.0                                            downsample     153,842,972   245,760,000    62.60%        34,331,664
downsample_layers.3.0                                            downsample     113,153,719   122,880,000    92.08%        13,377,334
stages.0.0.dwconv                                                dwconv             359,297   983,040,000     0.04%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,178,684,955 3,932,160,000    29.98%        34,886,881
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        26,433,360
stages.0.1.dwconv                                                dwconv             417,994   983,040,000     0.04%           848,256
stages.0.1.pw1_ttfs                                              pw1          1,236,837,257 3,932,160,000    31.45%        34,665,718
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        25,875,098
stages.1.0.dwconv                                                dwconv          57,217,128   491,520,000    11.64%           357,763
stages.1.0.pw1_ttfs                                              pw1            859,909,570 1,966,080,000    43.74%        34,906,276
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,238,472
stages.1.1.dwconv                                                dwconv          51,479,978   491,520,000    10.47%           357,763
stages.1.1.pw1_ttfs                                              pw1            859,494,660 1,966,080,000    43.72%        35,702,997
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,246,439
stages.2.0.dwconv                                                dwconv          78,321,171   245,760,000    31.87%            69,604
stages.2.0.pw1_ttfs                                              pw1            467,584,470   983,040,000    47.57%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,793,492
stages.2.1.dwconv                                                dwconv          84,050,532   245,760,000    34.20%            69,604
stages.2.1.pw1_ttfs                                              pw1            471,134,793   983,040,000    47.93%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,657,160
stages.2.2.dwconv                                                dwconv          84,768,374   245,760,000    34.49%            69,604
stages.2.2.pw1_ttfs                                              pw1            472,352,404   983,040,000    48.05%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,610,404
stages.2.3.dwconv                                                dwconv          87,623,330   245,760,000    35.65%            69,604
stages.2.3.pw1_ttfs                                              pw1            468,070,999   983,040,000    47.61%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,774,810
stages.2.4.dwconv                                                dwconv          77,201,137   245,760,000    31.41%            69,604
stages.2.4.pw1_ttfs                                              pw1            475,616,499   983,040,000    48.38%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,485,062
stages.2.5.dwconv                                                dwconv          76,642,510   245,760,000    31.19%            69,604
stages.2.5.pw1_ttfs                                              pw1            474,405,738   983,040,000    48.26%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,531,556
stages.3.0.dwconv                                                dwconv          44,023,735   122,880,000    35.83%             6,390
stages.3.0.pw1_ttfs                                              pw1            238,358,819   491,520,000    48.49%        37,748,736
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,442,779
stages.3.1.dwconv                                                dwconv          42,615,652   122,880,000    34.68%             6,390
stages.3.1.pw1_ttfs                                              pw1            233,319,573   491,520,000    47.47%        37,748,736
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,829,793
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     45.42%
===================================================================================================================

Classification accuracy: 45.17%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 785,337,914
Layerwise SynOps total:    7,853,379,141,846

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  14.66% | silent=684,720,838 | total=4,669,440,000
pw1             12 layers | sparsity=  39.81% | silent=7,435,769,737 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  38.03% | silent=327,106,527 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   37.15% | TTFS points= 6 | silent=4,382,379,503 | total=11,796,480,000
Stage 1:   44.94% | TTFS points= 7 | silent=2,871,251,172 | total=6,389,760,000
Stage 2:   54.39% | TTFS points=19 | silent=4,946,174,929 | total=9,093,120,000
Stage 3:   57.42% | TTFS points= 7 | silent=917,231,498 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             45.17%
Activation sparsity:  45.42%
Theoretical SynOps:   785,337,914 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_6543\activation_sparsity.md
```
