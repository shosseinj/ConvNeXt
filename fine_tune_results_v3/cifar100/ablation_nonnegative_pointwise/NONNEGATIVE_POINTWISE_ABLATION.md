# CIFAR-100 Non-Negative Pointwise Weight Ablation

PW1/PW2 use non-negative effective weights via ReLU. DWConv, downsampling, stem, and classifier weights remain unconstrained.

Values are mean ± sample standard deviation across seeds 42, 6543, and 7777.

## Overall comparison

| Condition | Best validation accuracy | Test accuracy | Weighted global sparsity |
|---|---:|---:|---:|
| Unconstrained fully TTFS | 72.87 ± 1.16% | 72.44 ± 0.02% | 32.38 ± 3.70% |
| Constraint before adaptation | — | 1.27 ± 0.39% | 33.55 ± 3.71% |
| Constraint after adaptation | 20.46 ± 1.19% | 20.75 ± 0.80% | 33.47 ± 3.00% |

## Per-seed results

### Unconstrained fully TTFS

| Seed | Best validation accuracy | Test accuracy | Weighted global sparsity |
|---:|---:|---:|---:|
| 42 | 72.84% | 72.44% | 36.60% |
| 6543 | 71.72% | 72.46% | 30.87% |
| 7777 | 74.04% | 72.43% | 29.67% |

### Constraint before adaptation

| Seed | Best validation accuracy | Test accuracy | Weighted global sparsity |
|---:|---:|---:|---:|
| 42 | — | 1.31% | 37.67% |
| 6543 | — | 0.86% | 32.53% |
| 7777 | — | 1.64% | 30.46% |

### Constraint after adaptation

| Seed | Best validation accuracy | Test accuracy | Weighted global sparsity |
|---:|---:|---:|---:|
| 42 | 21.78% | 21.66% | 36.92% |
| 6543 | 19.48% | 20.40% | 32.02% |
| 7777 | 20.12% | 20.18% | 31.48% |

## Layer-wise activation sparsity

| Layer | Unconstrained fully TTFS | Constraint before adaptation | Constraint after adaptation |
|---|---:|---:|---:|
| stages.0.0.dwconv | 0.64 ± 1.11% | 0.64 ± 1.11% | 0.62 ± 1.08% |
| stages.0.0.pw1_ttfs | 14.84 ± 23.88% | 11.98 ± 20.74% | 11.80 ± 20.44% |
| stages.0.0.pw2_ttfs | 18.41 ± 6.46% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.0.1.dwconv | 0.02 ± 0.02% | 0.02 ± 0.02% | 0.01 ± 0.02% |
| stages.0.1.pw1_ttfs | 0.31 ± 0.25% | 0.00 ± 0.00% | 0.00 ± 0.00% |
| stages.0.1.pw2_ttfs | 19.34 ± 4.03% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| downsample_layers.1.0 | 5.29 ± 1.77% | 5.28 ± 1.79% | 5.33 ± 1.88% |
| stages.1.0.dwconv | 7.04 ± 1.60% | 6.95 ± 1.61% | 8.19 ± 2.34% |
| stages.1.0.pw1_ttfs | 46.21 ± 0.77% | 32.51 ± 2.76% | 33.64 ± 1.54% |
| stages.1.0.pw2_ttfs | 97.47 ± 0.84% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.1.1.dwconv | 12.14 ± 2.22% | 13.54 ± 2.08% | 15.94 ± 3.09% |
| stages.1.1.pw1_ttfs | 42.67 ± 2.86% | 26.80 ± 5.00% | 15.59 ± 5.07% |
| stages.1.1.pw2_ttfs | 97.31 ± 1.59% | 100.00 ± 0.00% | 99.76 ± 0.22% |
| downsample_layers.2.0 | 27.64 ± 1.94% | 19.98 ± 3.60% | 23.31 ± 2.30% |
| stages.2.0.dwconv | 24.42 ± 0.72% | 28.81 ± 1.51% | 27.68 ± 0.35% |
| stages.2.0.pw1_ttfs | 40.75 ± 4.21% | 27.89 ± 5.08% | 29.61 ± 4.20% |
| stages.2.0.pw2_ttfs | 93.59 ± 2.28% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.2.1.dwconv | 31.76 ± 2.94% | 33.68 ± 1.33% | 34.71 ± 1.91% |
| stages.2.1.pw1_ttfs | 45.31 ± 1.12% | 36.41 ± 1.77% | 37.56 ± 2.40% |
| stages.2.1.pw2_ttfs | 96.48 ± 1.14% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.2.2.dwconv | 33.63 ± 1.92% | 35.23 ± 3.03% | 34.89 ± 2.21% |
| stages.2.2.pw1_ttfs | 46.16 ± 0.30% | 35.42 ± 3.37% | 38.35 ± 1.77% |
| stages.2.2.pw2_ttfs | 96.80 ± 1.87% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.2.3.dwconv | 37.54 ± 2.24% | 37.81 ± 0.90% | 37.57 ± 1.95% |
| stages.2.3.pw1_ttfs | 47.01 ± 1.38% | 37.07 ± 8.38% | 37.17 ± 3.28% |
| stages.2.3.pw2_ttfs | 98.29 ± 0.51% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.2.4.dwconv | 35.50 ± 4.35% | 34.15 ± 3.33% | 35.31 ± 3.42% |
| stages.2.4.pw1_ttfs | 47.79 ± 3.86% | 40.94 ± 11.92% | 47.09 ± 8.26% |
| stages.2.4.pw2_ttfs | 95.53 ± 2.49% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.2.5.dwconv | 37.38 ± 1.90% | 36.48 ± 2.34% | 37.94 ± 2.09% |
| stages.2.5.pw1_ttfs | 47.98 ± 0.23% | 32.52 ± 3.01% | 35.58 ± 4.27% |
| stages.2.5.pw2_ttfs | 97.48 ± 1.02% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| downsample_layers.3.0 | 62.73 ± 0.45% | 36.59 ± 4.03% | 78.93 ± 2.17% |
| stages.3.0.dwconv | 27.56 ± 0.92% | 32.54 ± 5.71% | 15.69 ± 1.46% |
| stages.3.0.pw1_ttfs | 53.05 ± 1.15% | 48.16 ± 2.52% | 42.85 ± 6.37% |
| stages.3.0.pw2_ttfs | 83.73 ± 1.04% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| stages.3.1.dwconv | 36.91 ± 0.58% | 37.63 ± 1.79% | 41.57 ± 1.59% |
| stages.3.1.pw1_ttfs | 52.33 ± 0.94% | 52.60 ± 0.62% | 51.39 ± 3.90% |
| stages.3.1.pw2_ttfs | 74.03 ± 1.04% | 100.00 ± 0.00% | 100.00 ± 0.00% |
