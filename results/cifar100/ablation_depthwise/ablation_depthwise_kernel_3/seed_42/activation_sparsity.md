# Activation Sparsity Evaluation

```text
Using checkpoint: results\cifar100\downsample_dense_dwconv_dense\seed_42\best_checkpoint.pth

Device: cuda
Dataset: cifar100
Evaluation samples: 10000

Detected depthwise convolution mode: dense (metadata)
Detected downsampling convolution mode: dense (metadata)

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
Batch    1/  79 | samples=   128 | accuracy= 74.22%
Batch   20/  79 | samples=  2560 | accuracy= 73.83%
Batch   40/  79 | samples=  5120 | accuracy= 73.20%
Batch   60/  79 | samples=  7680 | accuracy= 73.26%
Batch   79/  79 | samples= 10000 | accuracy= 73.46%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          1,705,697,766 3,932,160,000    43.38%
stages.0.0.pw2_ttfs                                              pw2            282,577,857   983,040,000    28.75%
stages.0.1.pw1_ttfs                                              pw1            242,636,084 3,932,160,000     6.17%
stages.0.1.pw2_ttfs                                              pw2            254,529,154   983,040,000    25.89%
stages.1.0.pw1_ttfs                                              pw1            913,142,305 1,966,080,000    46.44%
stages.1.0.pw2_ttfs                                              pw2            478,004,974   491,520,000    97.25%
stages.1.1.pw1_ttfs                                              pw1            739,561,127 1,966,080,000    37.62%
stages.1.1.pw2_ttfs                                              pw2            448,431,977   491,520,000    91.23%
stages.2.0.pw1_ttfs                                              pw1            390,225,531   983,040,000    39.70%
stages.2.0.pw2_ttfs                                              pw2            229,481,777   245,760,000    93.38%
stages.2.1.pw1_ttfs                                              pw1            450,939,944   983,040,000    45.87%
stages.2.1.pw2_ttfs                                              pw2            239,469,114   245,760,000    97.44%
stages.2.2.pw1_ttfs                                              pw1            464,430,476   983,040,000    47.24%
stages.2.2.pw2_ttfs                                              pw2            241,024,143   245,760,000    98.07%
stages.2.3.pw1_ttfs                                              pw1            468,048,469   983,040,000    47.61%
stages.2.3.pw2_ttfs                                              pw2            241,794,753   245,760,000    98.39%
stages.2.4.pw1_ttfs                                              pw1            496,079,923   983,040,000    50.46%
stages.2.4.pw2_ttfs                                              pw2            241,493,334   245,760,000    98.26%
stages.2.5.pw1_ttfs                                              pw1            467,941,406   983,040,000    47.60%
stages.2.5.pw2_ttfs                                              pw2            239,079,768   245,760,000    97.28%
stages.3.0.pw1_ttfs                                              pw1            264,154,716   491,520,000    53.74%
stages.3.0.pw2_ttfs                                              pw2            104,118,059   122,880,000    84.73%
stages.3.1.pw1_ttfs                                              pw1            249,210,680   491,520,000    50.70%
stages.3.1.pw2_ttfs                                              pw2             95,114,358   122,880,000    77.40%
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     42.61%
===================================================================================================================

Classification accuracy: 73.46%
Measured TTFS points:     24
Expected TTFS points:     24

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  36.69% | silent=6,852,068,427 | total=18,677,760,000
pw2             12 layers | sparsity=  66.28% | silent=3,095,119,268 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   25.28% | TTFS points= 4 | silent=2,485,440,861 | total=9,830,400,000
Stage 1:   52.47% | TTFS points= 4 | silent=2,579,140,383 | total=4,915,200,000
Stage 2:   56.56% | TTFS points=12 | silent=4,170,008,638 | total=7,372,800,000
Stage 3:   57.99% | TTFS points= 4 | silent=712,597,813 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             73.46%
Activation sparsity:  42.61%
TTFS layers/points:   24
================================================================================

Markdown report saved to: results\cifar100\downsample_dense_dwconv_dense\seed_42\activation_sparsity.md
```
