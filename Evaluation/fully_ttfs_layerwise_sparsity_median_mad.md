# Fully TTFS Layer-wise Activation Sparsity

Values are reported as **median ± median absolute deviation (MAD)** across three seeds.

- CIFAR-10 seeds: 42, 6543, 7777
- CIFAR-100 seeds: 42, 6543, 7777
- Tiny ImageNet seeds: 42, 2344, 5435
- Each best-validation checkpoint contains 39 measured TTFS sparsity points.

Downsampling layers are positioned at their architectural stage transitions.

| TTFS layer | CIFAR-10 | CIFAR-100 | Tiny ImageNet |
|---|---:|---:|---:|
| stages.0.0.dwconv | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.98 ± 0.13% |
| stages.0.0.pw1_ttfs | 0.01 ± 0.01% | 1.82 ± 1.51% | 0.00 ± 0.00% |
| stages.0.0.pw2_ttfs | 14.60 ± 2.34% | 19.52 ± 4.72% | 12.50 ± 3.00% |
| stages.0.1.dwconv | 0.08 ± 0.08% | 0.03 ± 0.00% | 0.70 ± 0.19% |
| stages.0.1.pw1_ttfs | 0.33 ± 0.33% | 0.41 ± 0.08% | 32.28 ± 0.78% |
| stages.0.1.pw2_ttfs | 19.08 ± 1.54% | 17.54 ± 1.01% | 8.23 ± 6.66% |
| **downsample_layers.1.0** | **5.89 ± 0.34%** | **5.40 ± 1.61%** | **8.65 ± 0.49%** |
| stages.1.0.dwconv | 7.81 ± 1.90% | 6.97 ± 1.49% | 6.60 ± 1.75% |
| stages.1.0.pw1_ttfs | 45.86 ± 0.93% | 46.37 ± 0.51% | 22.58 ± 0.05% |
| stages.1.0.pw2_ttfs | 98.57 ± 0.48% | 97.01 ± 0.05% | 88.97 ± 5.34% |
| stages.1.1.dwconv | 9.39 ± 0.20% | 13.20 ± 0.44% | 6.38 ± 0.53% |
| stages.1.1.pw1_ttfs | 45.83 ± 0.44% | 41.09 ± 0.13% | 17.70 ± 0.80% |
| stages.1.1.pw2_ttfs | 96.82 ± 0.52% | 96.91 ± 0.96% | 86.09 ± 3.55% |
| **downsample_layers.2.0** | **26.47 ± 1.20%** | **28.72 ± 0.07%** | **31.56 ± 1.79%** |
| stages.2.0.dwconv | 28.74 ± 0.25% | 24.26 ± 0.46% | 19.51 ± 1.36% |
| stages.2.0.pw1_ttfs | 47.27 ± 0.01% | 38.92 ± 1.16% | 12.74 ± 0.15% |
| stages.2.0.pw2_ttfs | 96.23 ± 0.11% | 93.07 ± 1.46% | 89.31 ± 3.80% |
| stages.2.1.dwconv | 33.15 ± 0.22% | 31.90 ± 2.73% | 31.78 ± 3.19% |
| stages.2.1.pw1_ttfs | 50.10 ± 0.12% | 45.03 ± 0.67% | 10.95 ± 3.00% |
| stages.2.1.pw2_ttfs | 96.92 ± 0.01% | 96.98 ± 0.31% | 95.82 ± 0.06% |
| stages.2.2.dwconv | 33.48 ± 0.20% | 32.57 ± 0.10% | 34.72 ± 2.78% |
| stages.2.2.pw1_ttfs | 53.70 ± 0.74% | 46.12 ± 0.23% | 4.12 ± 0.96% |
| stages.2.2.pw2_ttfs | 98.17 ± 0.07% | 97.73 ± 0.30% | 95.89 ± 1.29% |
| stages.2.3.dwconv | 34.95 ± 1.81% | 37.40 ± 2.03% | 37.08 ± 1.47% |
| stages.2.3.pw1_ttfs | 51.26 ± 0.64% | 46.78 ± 1.02% | 3.76 ± 0.40% |
| stages.2.3.pw2_ttfs | 96.46 ± 0.82% | 98.01 ± 0.03% | 94.45 ± 0.92% |
| stages.2.4.dwconv | 37.62 ± 0.04% | 36.83 ± 2.20% | 37.05 ± 2.06% |
| stages.2.4.pw1_ttfs | 47.73 ± 0.04% | 46.22 ± 1.25% | 6.59 ± 3.18% |
| stages.2.4.pw2_ttfs | 96.81 ± 0.45% | 96.47 ± 0.95% | 97.11 ± 0.17% |
| stages.2.5.dwconv | 38.02 ± 0.39% | 38.35 ± 0.26% | 37.88 ± 3.03% |
| stages.2.5.pw1_ttfs | 47.19 ± 1.78% | 47.85 ± 0.01% | 5.97 ± 0.48% |
| stages.2.5.pw2_ttfs | 95.94 ± 0.78% | 97.43 ± 0.95% | 95.31 ± 0.57% |
| **downsample_layers.3.0** | **63.13 ± 2.06%** | **62.95 ± 0.07%** | **79.90 ± 0.52%** |
| stages.3.0.dwconv | 28.58 ± 1.14% | 27.47 ± 0.79% | 19.47 ± 4.86% |
| stages.3.0.pw1_ttfs | 50.04 ± 0.14% | 53.40 ± 0.59% | 52.07 ± 1.03% |
| stages.3.0.pw2_ttfs | 88.54 ± 0.38% | 83.20 ± 0.13% | 87.55 ± 0.68% |
| stages.3.1.dwconv | 38.92 ± 0.10% | 36.89 ± 0.55% | 39.21 ± 0.88% |
| stages.3.1.pw1_ttfs | 51.89 ± 2.22% | 52.07 ± 0.53% | 47.61 ± 2.14% |
| stages.3.1.pw2_ttfs | 85.79 ± 0.08% | 73.57 ± 0.26% | 77.65 ± 3.65% |

## Global weighted activation sparsity

The global values below are reported separately as **mean ± sample standard deviation**, following the requested summary statistic.

| Dataset | Global sparsity mean ± std |
|---|---:|
| CIFAR-10 | **33.00 ± 3.02%** |
| CIFAR-100 | **32.38 ± 3.70%** |
| Tiny ImageNet | **21.70 ± 1.29%** |
