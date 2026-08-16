# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_6543\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: ttfs (metadata)
Detected downsampling convolution mode: ttfs (metadata)
Detected residual operator: min (metadata)
Detected non-negative effective pointwise weights: False (legacy default)

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
Batch    1/  79 | samples=   128 | accuracy= 71.88%
Batch   20/  79 | samples=  2560 | accuracy= 72.85%
Batch   40/  79 | samples=  5120 | accuracy= 72.75%
Batch   60/  79 | samples=  7680 | accuracy= 72.34%
Batch   79/  79 | samples= 10000 | accuracy= 72.46%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
downsample_layers.1.0                                            downsample      34,438,862   491,520,000     7.01%        40,716,288
downsample_layers.2.0                                            downsample      70,765,051   245,760,000    28.79%        37,685,613
downsample_layers.3.0                                            downsample      77,346,858   122,880,000    62.95%        29,072,938
stages.0.0.dwconv                                                dwconv              20,668   983,040,000     0.00%           848,256
stages.0.0.pw1_ttfs                                              pw1             71,377,434 3,932,160,000     1.82%        36,309,948
stages.0.0.pw2_ttfs                                              pw2            191,850,819   983,040,000    19.52%        37,063,513
stages.0.1.dwconv                                                dwconv             310,139   983,040,000     0.03%           848,256
stages.0.1.pw1_ttfs                                              pw1             19,408,104 3,932,160,000     0.49%        36,478,777
stages.0.1.pw2_ttfs                                              pw2            172,401,813   983,040,000    17.54%        37,562,418
stages.1.0.dwconv                                                dwconv          26,947,225   491,520,000     5.48%           379,546
stages.1.0.pw1_ttfs                                              pw1            892,006,441 1,966,080,000    45.37%        34,526,682
stages.1.0.pw2_ttfs                                              pw2            476,565,341   491,520,000    96.96%        20,622,212
stages.1.1.dwconv                                                dwconv          67,049,426   491,520,000    13.64%           389,666
stages.1.1.pw1_ttfs                                              pw1            903,722,520 1,966,080,000    45.97%        37,677,285
stages.1.1.pw2_ttfs                                              pw2            476,352,823   491,520,000    96.91%        20,397,264
stages.2.0.dwconv                                                dwconv          61,962,362   245,760,000    25.21%           132,132
stages.2.0.pw1_ttfs                                              pw1            447,929,486   983,040,000    45.57%        37,528,824
stages.2.0.pw2_ttfs                                              pw2            236,130,222   245,760,000    96.08%        20,548,244
stages.2.1.dwconv                                                dwconv          70,652,063   245,760,000    28.75%           137,562
stages.2.1.pw1_ttfs                                              pw1            457,649,292   983,040,000    46.55%        37,748,296
stages.2.1.pw2_ttfs                                              pw2            233,922,018   245,760,000    95.18%        20,175,003
stages.2.2.dwconv                                                dwconv          80,043,116   245,760,000    32.57%           142,858
stages.2.2.pw1_ttfs                                              pw1            456,932,739   983,040,000    46.48%        37,748,736
stages.2.2.pw2_ttfs                                              pw2            240,925,915   245,760,000    98.03%        20,202,519
stages.2.3.dwconv                                                dwconv          91,902,936   245,760,000    37.40%           144,803
stages.2.3.pw1_ttfs                                              pw1            459,842,913   983,040,000    46.78%        37,748,736
stages.2.3.pw2_ttfs                                              pw2            240,798,329   245,760,000    97.98%        20,090,768
stages.2.4.dwconv                                                dwconv          75,311,723   245,760,000    30.64%           146,349
stages.2.4.pw1_ttfs                                              pw1            454,375,970   983,040,000    46.22%        37,748,559
stages.2.4.pw2_ttfs                                              pw2            227,841,396   245,760,000    92.71%        20,300,699
stages.2.5.dwconv                                                dwconv          94,245,422   245,760,000    38.35%           150,618
stages.2.5.pw1_ttfs                                              pw1            470,410,281   983,040,000    47.85%        37,748,736
stages.2.5.pw2_ttfs                                              pw2            242,110,657   245,760,000    98.52%        19,684,981
stages.3.0.dwconv                                                dwconv          35,041,882   122,880,000    28.52%            28,569
stages.3.0.pw1_ttfs                                              pw1            262,477,735   491,520,000    53.40%        37,730,488
stages.3.0.pw2_ttfs                                              pw2            104,365,970   122,880,000    84.93%        17,590,446
stages.3.1.dwconv                                                dwconv          46,074,204   122,880,000    37.50%            36,596
stages.3.1.pw1_ttfs                                              pw1            253,323,987   491,520,000    51.54%        37,745,627
stages.3.1.pw2_ttfs                                              pw2             90,398,389   122,880,000    73.57%        18,293,454
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     30.87%
===================================================================================================================

Classification accuracy: 72.46%
Measured TTFS points:     39
Expected TTFS points:     39
Theoretical SynOps/sample: 830,132,265
Layerwise SynOps total:    8,301,322,651,080

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv          12 layers | sparsity=  13.91% | silent=649,561,166 | total=4,669,440,000
pw1             12 layers | sparsity=  27.57% | silent=5,149,456,902 | total=18,677,760,000
pw2             12 layers | sparsity=  62.83% | silent=2,933,663,692 | total=4,669,440,000
downsample       3 layers | sparsity=  21.22% | silent=182,550,771 | total=860,160,000
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    3.86% | TTFS points= 6 | silent=455,368,977 | total=11,796,480,000
Stage 1:   45.03% | TTFS points= 7 | silent=2,877,082,638 | total=6,389,760,000
Stage 2:   51.84% | TTFS points=19 | silent=4,713,751,891 | total=9,093,120,000
Stage 3:   54.40% | TTFS points= 7 | silent=869,029,025 | total=1,597,440,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.46%
Activation sparsity:  30.87%
Theoretical SynOps:   830,132,265 per sample
TTFS layers/points:   39
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\fine_tune_results_v3\cifar100\fully_ttfs\seed_6543\activation_sparsity.md
```
