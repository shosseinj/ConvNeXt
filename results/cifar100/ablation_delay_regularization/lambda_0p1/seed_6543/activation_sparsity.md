# Activation Sparsity Evaluation

```text
Using checkpoint: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_6543\best_checkpoint.pth

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
Batch   20/  79 | samples=  2560 | accuracy= 72.62%
Batch   40/  79 | samples=  5120 | accuracy= 72.48%
Batch   60/  79 | samples=  7680 | accuracy= 72.50%
Batch   79/  79 | samples= 10000 | accuracy= 72.75%


===================================================================================================================
TTFS ACTIVATION SPARSITY REPORT - CIFAR100
===================================================================================================================
Layer                                                            Type                Silent         Total  Sparsity     SynOps/sample
-------------------------------------------------------------------------------------------------------------------
stages.0.0.pw1_ttfs                                              pw1          2,086,272,535 3,932,160,000    53.06%        37,549,343
stages.0.0.pw2_ttfs                                              pw2            479,697,523   983,040,000    48.80%        17,720,520
stages.0.1.pw1_ttfs                                              pw1          1,876,143,271 3,932,160,000    47.71%        37,197,233
stages.0.1.pw2_ttfs                                              pw2            463,945,374   983,040,000    47.19%        19,737,761
stages.1.0.pw1_ttfs                                              pw1            878,713,853 1,966,080,000    44.69%        35,037,138
stages.1.0.pw2_ttfs                                              pw2            470,587,207   491,520,000    95.74%        20,877,430
stages.1.1.pw1_ttfs                                              pw1            798,045,264 1,966,080,000    40.59%        36,212,095
stages.1.1.pw2_ttfs                                              pw2            456,624,614   491,520,000    92.90%        22,426,267
stages.2.0.pw1_ttfs                                              pw1            441,294,587   983,040,000    44.89%        34,992,917
stages.2.0.pw2_ttfs                                              pw2            236,610,584   245,760,000    96.28%        20,803,024
stages.2.1.pw1_ttfs                                              pw1            485,520,897   983,040,000    49.39%        35,016,248
stages.2.1.pw2_ttfs                                              pw2            235,996,060   245,760,000    96.03%        19,104,734
stages.2.2.pw1_ttfs                                              pw1            481,904,439   983,040,000    49.02%        35,449,965
stages.2.2.pw2_ttfs                                              pw2            240,078,145   245,760,000    97.69%        19,243,606
stages.2.3.pw1_ttfs                                              pw1            500,909,757   983,040,000    50.96%        35,913,611
stages.2.3.pw2_ttfs                                              pw2            242,436,672   245,760,000    98.65%        18,513,801
stages.2.4.pw1_ttfs                                              pw1            491,568,394   983,040,000    50.00%        35,256,003
stages.2.4.pw2_ttfs                                              pw2            231,162,813   245,760,000    94.06%        18,872,510
stages.2.5.pw1_ttfs                                              pw1            500,406,976   983,040,000    50.90%        35,930,345
stages.2.5.pw2_ttfs                                              pw2            242,775,785   245,760,000    98.79%        18,533,108
stages.3.0.pw1_ttfs                                              pw1            267,219,277   491,520,000    54.37%        33,884,224
stages.3.0.pw2_ttfs                                              pw2            105,084,157   122,880,000    85.52%        17,226,296
stages.3.1.pw1_ttfs                                              pw1            242,236,105   491,520,000    49.28%        34,303,628
stages.3.1.pw2_ttfs                                              pw2             88,537,136   122,880,000    72.05%        19,145,003
-------------------------------------------------------------------------------------------------------------------
GLOBAL WEIGHTED SPARSITY                                                                                     53.73%
===================================================================================================================

Classification accuracy: 72.75%
Measured TTFS points:     24
Expected TTFS points:     24
Theoretical SynOps/sample: 658,946,808
Layerwise SynOps total:    6,589,468,081,920

================================================================================
SPARSITY BY TTFS OPERATION TYPE
================================================================================
dwconv           0 layers | sparsity=   0.00% | silent=0 | total=0
pw1             12 layers | sparsity=  48.45% | silent=9,050,235,355 | total=18,677,760,000
pw2             12 layers | sparsity=  74.82% | silent=3,493,536,070 | total=4,669,440,000
downsample       0 layers | sparsity=   0.00% | silent=0 | total=0
================================================================================

================================================================================
STAGE-WISE WEIGHTED SPARSITY
================================================================================
Stage 0:   49.91% | TTFS points= 4 | silent=4,906,058,703 | total=9,830,400,000
Stage 1:   52.98% | TTFS points= 4 | silent=2,603,970,938 | total=4,915,200,000
Stage 2:   58.74% | TTFS points=12 | silent=4,330,665,109 | total=7,372,800,000
Stage 3:   57.22% | TTFS points= 4 | silent=703,076,675 | total=1,228,800,000
================================================================================

================================================================================
PAPER-READY SUMMARY
================================================================================
Dataset:              cifar100
Accuracy:             72.75%
Activation sparsity:  53.73%
Theoretical SynOps:   658,946,808 per sample
TTFS layers/points:   24
================================================================================

Markdown report saved to: C:\Users\jafari.h\Desktop\ai_project\ConvNeXt\results\cifar100\ablation_delay_regularization\lambda_0p1\seed_6543\activation_sparsity.md
```
