# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_42\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 48.48%
Batch   40/  79 | samples=  5120 | accuracy= 47.56%
Batch   60/  79 | samples=  7680 | accuracy= 47.47%
Batch   79/  79 | samples= 10000 | accuracy= 47.50%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      66,490,378   491,520,000    13.53%        40,716,288
downsample_layers.2.0                                            downsample     154,599,663   245,760,000    62.91%        33,801,598
downsample_layers.3.0                                            downsample     113,620,864   122,880,000    92.46%        13,251,977
stages.0.0.dwconv                                                dwconv              29,548   983,040,000     0.00%           848,256
stages.0.0.pw1_ttfs                                              pw1          1,227,187,394 3,932,160,000    31.21%        34,497,711
stages.0.0.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        25,967,737
stages.0.1.dwconv                                                dwconv                   7   983,040,000     0.00%           848,256
stages.0.1.pw1_ttfs                                              pw1          1,213,383,265 3,932,160,000    30.86%        33,863,210
stages.0.1.pw2_ttfs                                              pw2            983,040,000   983,040,000   100.00%        26,100,257
stages.1.0.dwconv                                                dwconv          35,454,982   491,520,000     7.21%           352,221
stages.1.0.pw1_ttfs                                              pw1            845,777,837 1,966,080,000    43.02%        34,634,708
stages.1.0.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,509,802
stages.1.1.dwconv                                                dwconv          45,674,484   491,520,000     9.29%           352,221
stages.1.1.pw1_ttfs                                              pw1            848,721,532 1,966,080,000    43.17%        34,983,286
stages.1.1.pw2_ttfs                                              pw2            491,520,000   491,520,000   100.00%        21,453,283
stages.2.0.dwconv                                                dwconv          77,163,666   245,760,000    31.40%            68,955
stages.2.0.pw1_ttfs                                              pw1            474,563,754   983,040,000    48.28%        37,748,736
stages.2.0.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,525,488
stages.2.1.dwconv                                                dwconv          73,214,379   245,760,000    29.79%            68,955
stages.2.1.pw1_ttfs                                              pw1            462,092,129   983,040,000    47.01%        37,748,736
stages.2.1.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        20,004,398
stages.2.2.dwconv                                                dwconv          72,690,048   245,760,000    29.58%            68,955
stages.2.2.pw1_ttfs                                              pw1            468,997,598   983,040,000    47.71%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,739,228
stages.2.3.dwconv                                                dwconv          68,239,534   245,760,000    27.77%            68,955
stages.2.3.pw1_ttfs                                              pw1            462,389,236   983,040,000    47.04%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,992,989
stages.2.4.dwconv                                                dwconv          74,903,957   245,760,000    30.48%            68,955
stages.2.4.pw1_ttfs                                              pw1            464,424,005   983,040,000    47.24%        37,748,736
stages.2.4.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        19,914,854
stages.2.5.dwconv                                                dwconv          76,158,795   245,760,000    30.99%            68,955
stages.2.5.pw1_ttfs                                              pw1            461,831,710   983,040,000    46.98%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            245,760,000   245,760,000   100.00%        20,014,398
stages.3.0.dwconv                                                dwconv          36,370,069   122,880,000    29.60%             6,094
stages.3.0.pw1_ttfs                                              pw1            237,976,789   491,520,000    48.42%        37,748,736
stages.3.0.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,472,119
stages.3.1.dwconv                                                dwconv          38,618,337   122,880,000    31.43%             6,094
stages.3.1.pw1_ttfs                                              pw1            234,762,921   491,520,000    47.76%        37,748,736
stages.3.1.pw2_ttfs                                              pw2            122,880,000   122,880,000   100.00%        19,718,944
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     45.04%
===================================================================================================================

Classification accuracy: 47.50%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 783,979,035
Layerwise SynOps total:    7,839,790,351,206

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  12.82% | silent=598,517,806 | total=4,669,440,000
pw1             12 layers | sparsity=  39.63% | silent=7,402,108,170 | total=18,677,760,000
pw2             12 layers | sparsity= 100.00% | silent=4,669,440,000 | total=4,669,440,000
downsample       3 layers | sparsity=  38.91% | silent=334,710,905 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   37.36% | TTFS points= 6 | silent=4,406,680,214 | total=11,796,480,000
Stage 1:   44.21% | TTFS points= 7 | silent=2,825,159,213 | total=6,389,760,000
Stage 2:   53.51% | TTFS points=19 | silent=4,865,828,474 | total=9,093,120,000
Stage 3:   56.79% | TTFS points= 7 | silent=907,108,980 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             47.50%
Activation sparsity:  45.04%
Theoretical SynOps:   783,979,035 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_42\activation_sparsity.md
```
