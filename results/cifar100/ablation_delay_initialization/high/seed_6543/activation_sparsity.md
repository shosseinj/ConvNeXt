# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_6543\best_checkpoint.pth

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
Batch    1/  79 | samples=   128 | accuracy= 75.00%
Batch   20/  79 | samples=  2560 | accuracy= 72.70%
Batch   40/  79 | samples=  5120 | accuracy= 72.54%
Batch   60/  79 | samples=  7680 | accuracy= 72.54%
Batch   79/  79 | samples= 10000 | accuracy= 72.80%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1                 84,877 3,932,160,000     0.00%        37,211,060
stages.0.0.pw2_ttfs                                              pw2            124,188,799   983,040,000    12.63%        37,747,921
stages.0.1.pw1_ttfs                                              pw1                850,512 3,932,160,000     0.02%        37,220,167
stages.0.1.pw2_ttfs                                              pw2            126,428,355   983,040,000    12.86%        37,740,571
stages.1.0.pw1_ttfs                                              pw1            842,333,032 1,966,080,000    42.84%        35,210,110
stages.1.0.pw2_ttfs                                              pw2            467,196,062   491,520,000    95.05%        21,575,942
stages.1.1.pw1_ttfs                                              pw1            898,182,720 1,966,080,000    45.68%        35,664,739
stages.1.1.pw2_ttfs                                              pw2            481,438,630   491,520,000    97.95%        20,503,628
stages.2.0.pw1_ttfs                                              pw1            415,425,396   983,040,000    42.26%        35,101,433
stages.2.0.pw2_ttfs                                              pw2            237,457,205   245,760,000    96.62%        21,796,401
stages.2.1.pw1_ttfs                                              pw1            452,922,329   983,040,000    46.07%        35,053,062
stages.2.1.pw2_ttfs                                              pw2            234,877,082   245,760,000    95.57%        20,356,519
stages.2.2.pw1_ttfs                                              pw1            466,139,495   983,040,000    47.42%        35,647,863
stages.2.2.pw2_ttfs                                              pw2            241,498,404   245,760,000    98.27%        19,848,979
stages.2.3.pw1_ttfs                                              pw1            480,363,783   983,040,000    48.87%        35,714,681
stages.2.3.pw2_ttfs                                              pw2            240,420,787   245,760,000    97.83%        19,302,767
stages.2.4.pw1_ttfs                                              pw1            476,749,913   983,040,000    48.50%        35,331,176
stages.2.4.pw2_ttfs                                              pw2            231,486,463   245,760,000    94.19%        19,441,539
stages.2.5.pw1_ttfs                                              pw1            488,791,291   983,040,000    49.72%        35,967,041
stages.2.5.pw2_ttfs                                              pw2            242,653,627   245,760,000    98.74%        18,979,150
stages.3.0.pw1_ttfs                                              pw1            260,008,108   491,520,000    52.90%        34,018,976
stages.3.0.pw2_ttfs                                              pw2            104,532,697   122,880,000    85.07%        17,780,113
stages.3.1.pw1_ttfs                                              pw1            245,021,247   491,520,000    49.85%        34,374,480
stages.3.1.pw2_ttfs                                              pw2             93,164,339   122,880,000    75.82%        18,931,104
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     33.63%
===================================================================================================================

Classification accuracy: 72.80%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 700,519,422
Layerwise SynOps total:    7,005,194,218,272

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  26.91% | silent=5,026,872,703 | total=18,677,760,000
pw2             12 layers | sparsity=  60.51% | silent=2,825,342,450 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:    2.56% | TTFS points= 4 | silent=251,552,543 | total=9,830,400,000
Stage 1:   54.71% | TTFS points= 4 | silent=2,689,150,444 | total=4,915,200,000
Stage 2:   57.09% | TTFS points=12 | silent=4,208,785,775 | total=7,372,800,000
Stage 3:   57.19% | TTFS points= 4 | silent=702,726,391 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.80%
Activation sparsity:  33.63%
Theoretical SynOps:   700,519,422 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_initialization\high\seed_6543\activation_sparsity.md
```
