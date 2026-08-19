# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_7777\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: dense (metadata)
Detected downsampling convolution mode: dense (metadata)
Detected residual operator: min (metadata)
Detected non-negative effective pointwise weights: False (metadata)

Missing keys:    0
Unexpected keys: 0

==========================================================================================
MODEL SPARSITY STRUCTURE
==========================================================================================
Spiking blocks:              12
ContinuousTTFSConv2d:        0
PW1 TTFS outputs:            12
PW2 TTFS outputs:            12
Expected total TTFS points:  24
==========================================================================================

TTFS Conv modules:
Batch    1/  79 | samples=   128 | accuracy= 78.12%
Batch   20/  79 | samples=  2560 | accuracy= 75.08%
Batch   40/  79 | samples=  5120 | accuracy= 74.06%
Batch   60/  79 | samples=  7680 | accuracy= 74.28%
Batch   79/  79 | samples= 10000 | accuracy= 74.60%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1             24,735,598 3,932,160,000     0.63%        37,150,558
stages.0.0.pw2_ttfs                                              pw2            207,950,997   983,040,000    21.15%        37,511,274
stages.0.1.pw1_ttfs                                              pw1             23,827,572 3,932,160,000     0.61%        37,026,955
stages.0.1.pw2_ttfs                                              pw2            350,808,676   983,040,000    35.69%        37,519,991
stages.1.0.pw1_ttfs                                              pw1            901,589,959 1,966,080,000    45.86%        35,102,939
stages.1.0.pw2_ttfs                                              pw2            474,649,849   491,520,000    96.57%        20,438,209
stages.1.1.pw1_ttfs                                              pw1            874,429,872 1,966,080,000    44.48%        35,778,808
stages.1.1.pw2_ttfs                                              pw2            458,123,064   491,520,000    93.21%        20,959,682
stages.2.0.pw1_ttfs                                              pw1            451,164,648   983,040,000    45.89%        35,048,490
stages.2.0.pw2_ttfs                                              pw2            237,491,336   245,760,000    96.64%        20,424,014
stages.2.1.pw1_ttfs                                              pw1            481,901,194   983,040,000    49.02%        35,674,812
stages.2.1.pw2_ttfs                                              pw2            240,744,314   245,760,000    97.96%        19,243,730
stages.2.2.pw1_ttfs                                              pw1            469,249,991   983,040,000    47.73%        35,148,923
stages.2.2.pw2_ttfs                                              pw2            232,901,504   245,760,000    94.77%        19,729,536
stages.2.3.pw1_ttfs                                              pw1            489,665,324   983,040,000    49.81%        36,011,726
stages.2.3.pw2_ttfs                                              pw2            242,374,950   245,760,000    98.62%        18,945,588
stages.2.4.pw1_ttfs                                              pw1            465,449,507   983,040,000    47.35%        35,993,908
stages.2.4.pw2_ttfs                                              pw2            239,611,193   245,760,000    97.50%        19,875,475
stages.2.5.pw1_ttfs                                              pw1            489,454,127   983,040,000    49.79%        35,873,743
stages.2.5.pw2_ttfs                                              pw2            237,537,487   245,760,000    96.65%        18,953,698
stages.3.0.pw1_ttfs                                              pw1            261,632,781   491,520,000    53.23%        34,173,046
stages.3.0.pw2_ttfs                                              pw2            106,154,486   122,880,000    86.39%        17,655,338
stages.3.1.pw1_ttfs                                              pw1            262,582,567   491,520,000    53.42%        34,276,791
stages.3.1.pw2_ttfs                                              pw2             94,400,494   122,880,000    76.82%        17,582,395
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     35.63%
===================================================================================================================

Classification accuracy: 74.60%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 696,099,629
Layerwise SynOps total:    6,960,996,288,768

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  27.82% | silent=5,195,683,140 | total=18,677,760,000
pw2             12 layers | sparsity=  66.88% | silent=3,122,748,350 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    6.18% | TTFS points= 4 | silent=607,322,843 | total=9,830,400,000
Stage 1:   55.11% | TTFS points= 4 | silent=2,708,792,744 | total=4,915,200,000
Stage 2:   58.02% | TTFS points=12 | silent=4,277,545,575 | total=7,372,800,000
Stage 3:   58.98% | TTFS points= 4 | silent=724,770,328 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             74.60%
Activation sparsity:  35.63%
Theoretical SynOps:   696,099,629 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\low\seed_7777\activation_sparsity.md
```
