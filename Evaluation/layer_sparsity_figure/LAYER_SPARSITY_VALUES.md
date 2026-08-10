# Layer-wise activation sparsity values

Values are reported as mean +/- sample standard deviation (%) across seeds 42, 6543, and 7777. Stage and block numbers are one-based. PW1 and PW2 denote the first and second pointwise transformations within each ConvNeXt block.

| Layer | CIFAR-10 PW1 (%) | CIFAR-10 PW2 (%) | CIFAR-100 PW1 (%) | CIFAR-100 PW2 (%) |
|---|---:|---:|---:|---:|
| S1-B1 | 0.49 +/- 0.41 | 23.18 +/- 5.14 | 14.75 +/- 24.80 | 19.94 +/- 8.43 |
| S1-B2 | 15.36 +/- 23.44 | 22.62 +/- 2.50 | 2.94 +/- 3.08 | 24.45 +/- 1.26 |
| S2-B1 | 47.04 +/- 0.81 | 97.92 +/- 0.54 | 46.01 +/- 0.85 | 96.76 +/- 0.44 |
| S2-B2 | 44.18 +/- 2.82 | 96.61 +/- 0.68 | 41.19 +/- 3.10 | 93.00 +/- 1.90 |
| S3-B1 | 47.81 +/- 0.06 | 96.65 +/- 0.31 | 42.08 +/- 3.76 | 94.69 +/- 1.94 |
| S3-B2 | 51.44 +/- 0.90 | 97.10 +/- 0.13 | 46.82 +/- 1.17 | 97.05 +/- 0.61 |
| S3-B3 | 54.81 +/- 2.65 | 98.41 +/- 0.20 | 47.51 +/- 0.77 | 97.29 +/- 1.75 |
| S3-B4 | 52.33 +/- 2.26 | 97.28 +/- 1.41 | 48.20 +/- 0.92 | 98.63 +/- 0.33 |
| S3-B5 | 51.67 +/- 2.94 | 97.06 +/- 0.75 | 49.19 +/- 1.39 | 96.37 +/- 2.24 |
| S3-B6 | 50.95 +/- 3.38 | 96.27 +/- 0.90 | 48.74 +/- 1.55 | 97.93 +/- 0.82 |
| S4-B1 | 50.56 +/- 0.77 | 89.42 +/- 0.45 | 53.35 +/- 0.61 | 85.53 +/- 1.08 |
| S4-B2 | 52.95 +/- 2.38 | 87.87 +/- 0.57 | 51.60 +/- 1.96 | 76.41 +/- 0.98 |

## Aggregation details

- Statistic: arithmetic mean +/- sample standard deviation (`n - 1`).
- Number of runs per dataset: 3.
- Architecture: dense depthwise convolution and dense downsampling, with TTFS PW1 and PW2 outputs.
- TTFS measurement points per model: 24.
- Source: each seed's `activation_sparsity.md` report.
