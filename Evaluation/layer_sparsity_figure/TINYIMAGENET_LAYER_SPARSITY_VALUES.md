# Tiny ImageNet layer-wise activation sparsity

Values are the arithmetic mean +/- sample standard deviation (`n - 1`) across seeds 42, 2344, and 5435. Stage and block numbers below are one-based. PW1 and PW2 are the first and second pointwise transformations in each ConvNeXt block.

| Stage | Block | PW1 sparsity mean +/- SD (%) | PW2 sparsity mean +/- SD (%) |
|---:|---:|---:|---:|
| 1 | 1 | 0.00 +/- 0.00 | 7.99 +/- 1.21 |
| 1 | 2 | 28.87 +/- 6.67 | 3.12 +/- 2.59 |
| 2 | 1 | 23.84 +/- 3.01 | 80.15 +/- 7.96 |
| 2 | 2 | 17.13 +/- 6.17 | 78.03 +/- 4.81 |
| 3 | 1 | 18.54 +/- 5.03 | 92.32 +/- 3.52 |
| 3 | 2 | 14.41 +/- 3.31 | 94.01 +/- 5.00 |
| 3 | 3 | 11.41 +/- 5.91 | 97.28 +/- 1.02 |
| 3 | 4 | 8.75 +/- 3.31 | 96.30 +/- 1.31 |
| 3 | 5 | 10.60 +/- 6.46 | 98.14 +/- 0.59 |
| 3 | 6 | 10.05 +/- 2.09 | 96.75 +/- 0.98 |
| 4 | 1 | 48.40 +/- 4.29 | 90.12 +/- 1.63 |
| 4 | 2 | 44.15 +/- 2.41 | 81.64 +/- 4.12 |

## Overall results

| Metric | Mean +/- SD (%) |
|---|---:|
| Classification accuracy | 63.11 +/- 0.32 |
| Global weighted activation sparsity | 24.12 +/- 1.51 |

## Evaluation configuration

- Dataset: Tiny ImageNet validation set, 10,000 images.
- Seeds: 42, 2344, and 5435.
- Measured TTFS points per seed: 24.
- Depthwise convolution: dense (not included as a TTFS sparsity point).
- Downsampling convolution: dense (not included as a TTFS sparsity point).
- Source reports: `results/tinyimagenet/test_downsample_ttfs_dwconv_dense/seed_<seed>/activation_sparsity.md`.

