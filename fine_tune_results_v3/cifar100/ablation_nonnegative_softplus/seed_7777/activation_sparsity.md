# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_7777\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 53.12%
Batch   20/  79 | samples=  2560 | accuracy= 49.26%
Batch   40/  79 | samples=  5120 | accuracy= 48.67%
Batch   60/  79 | samples=  7680 | accuracy= 47.90%
Batch   79/  79 | samples= 10000 | accuracy= 47.81%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      76,014,159   491,520,000    15.47%        40,716,288
downsample_layers.2.0                                            downsample     155,952,695   245,760,000    63.46%        33,045,848
downsample_layers.3.0                                            downsample     113,276,857   122,880,000    92.18%        13,038,847
stages.0.0.dwconv                                                dwconv              18,905   983,040,000     0.00%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,298,652,755 3,932,160,000    33.03%        34,916,348
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        25,281,670
stages.0.1.dwconv                                                dwconv              10,000   983,040,000     0.00%           848,256
stages.0.1.pw1_ttfs                                              pw1          1,163,291,352 3,932,160,000    29.58%        35,421,458
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        26,581,139
stages.1.0.dwconv                                                dwconv          43,646,781   491,520,000     8.88%           344,343
stages.1.0.pw1_ttfs                                              pw1            863,349,197 1,966,080,000    43.91%        35,007,881
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,172,431
stages.1.1.dwconv                                                dwconv          43,722,614   491,520,000     8.90%           344,343
stages.1.1.pw1_ttfs                                              pw1            845,744,856 1,966,080,000    43.02%        35,085,083
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,510,435
stages.2.0.dwconv                                                dwconv          66,508,912   245,760,000    27.06%            67,840
stages.2.0.pw1_ttfs                                              pw1            459,657,934   983,040,000    46.76%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        20,097,871
stages.2.1.dwconv                                                dwconv          74,865,107   245,760,000    30.46%            67,840
stages.2.1.pw1_ttfs                                              pw1            469,762,292   983,040,000    47.79%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,709,864
stages.2.2.dwconv                                                dwconv          69,108,843   245,760,000    28.12%            67,840
stages.2.2.pw1_ttfs                                              pw1            468,523,254   983,040,000    47.66%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,757,443
stages.2.3.dwconv                                                dwconv          70,551,987   245,760,000    28.71%            67,840
stages.2.3.pw1_ttfs                                              pw1            461,440,686   983,040,000    46.94%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        20,029,414
stages.2.4.dwconv                                                dwconv          72,312,826   245,760,000    29.42%            67,840
stages.2.4.pw1_ttfs                                              pw1            466,378,211   983,040,000    47.44%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,839,813
stages.2.5.dwconv                                                dwconv          72,300,386   245,760,000    29.42%            67,840
stages.2.5.pw1_ttfs                                              pw1            458,103,730   983,040,000    46.60%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        20,157,553
stages.3.0.dwconv                                                dwconv          37,465,894   122,880,000    30.49%             6,327
stages.3.0.pw1_ttfs                                              pw1            238,835,021   491,520,000    48.59%        37,748,736
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,406,206
stages.3.1.dwconv                                                dwconv          34,192,783   122,880,000    27.83%             6,327
stages.3.1.pw1_ttfs                                              pw1            232,937,974   491,520,000    47.39%        37,748,736
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,859,100
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     45.11%
===================================================================================================================

Classification accuracy: 47.81%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 785,429,472
Layerwise SynOps total:    7,854,294,721,604

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  12.52% | silent=584,705,038 | total=4,669,440,000
pw1             12 layers | sparsity=  39.76% | silent=7,426,677,262 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  40.14% | silent=345,243,711 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   37.54% | TTFS points= 6 | silent=4,428,053,012 | total=11,796,480,000
Stage 1:   44.69% | TTFS points= 7 | silent=2,855,517,607 | total=6,389,760,000
Stage 2:   53.23% | TTFS points=19 | silent=4,840,026,863 | total=9,093,120,000
Stage 3:   56.49% | TTFS points= 7 | silent=902,468,529 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             47.81%
Activation sparsity:  45.11%
Theoretical SynOps:   785,429,472 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_7777\activation_sparsity.md
```
