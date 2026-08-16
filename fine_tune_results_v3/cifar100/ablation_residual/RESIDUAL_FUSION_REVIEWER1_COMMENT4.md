# CIFAR-100 Fully-TTFS Residual Fusion Ablation

All values are mean ± sample standard deviation across seeds 42, 6543, and 7777. The 10-view TTA accuracy uses the predeclared flip_shift protocol on the checkpoint selected exclusively by best validation accuracy. Activation sparsity and SynOps are measured on the standard single-view test set.

## Main results

| Residual fusion | Best validation accuracy | 10-view TTA test accuracy | Weighted activation sparsity |
|---|---:|---:|---:|
| Minimum | 72.87 ± 1.16% | 73.72 ± 0.25% | 32.38 ± 3.70% |
| Normalized Sum (Mean) | 71.69 ± 1.90% | 72.57 ± 1.38% | 34.07 ± 3.92% |
| Learnable Gate | 72.00 ± 1.35% | 72.58 ± 0.61% | 34.51 ± 4.62% |

## Per-seed results

### Minimum

| Seed | Best validation accuracy | Standard test accuracy | 10-view TTA test accuracy | TTA gain | Weighted sparsity | SynOps/sample |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 72.84% | 72.42% | 73.91% | +1.49 pp | 36.60% | 817,885,116 |
| 6543 | 71.72% | 72.46% | 73.43% | +0.97 pp | 30.87% | 830,132,265 |
| 7777 | 74.04% | 72.39% | 73.81% | +1.42 pp | 29.67% | 836,937,821 |

### Normalized Sum (Mean)

| Seed | Best validation accuracy | Standard test accuracy | 10-view TTA test accuracy | TTA gain | Weighted sparsity | SynOps/sample |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 72.72% | 72.33% | 73.42% | +1.09 pp | 33.08% | 869,299,657 |
| 6543 | 69.50% | 70.00% | 70.98% | +0.98 pp | 38.39% | 842,388,008 |
| 7777 | 72.86% | 71.83% | 73.31% | +1.48 pp | 30.75% | 882,376,066 |

### Learnable Gate

| Seed | Best validation accuracy | Standard test accuracy | 10-view TTA test accuracy | TTA gain | Weighted sparsity | SynOps/sample |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 72.54% | 71.55% | 73.20% | +1.65 pp | 31.41% | 879,812,414 |
| 6543 | 70.46% | 70.84% | 71.98% | +1.14 pp | 39.82% | 835,966,957 |
| 7777 | 73.00% | 71.64% | 72.55% | +0.91 pp | 32.31% | 872,616,702 |

## Layerwise supplementary results

| Layer | Type | Minimum sparsity | Normalized Sum (Mean) sparsity | Learnable Gate sparsity | Minimum SynOps | Normalized Sum (Mean) SynOps | Learnable Gate SynOps |
|---|---|---:|---:|---:|---:|---:|---:|
| downsample_layers.1.0 | downsample | 5.29 ± 1.77% | 30.86 ± 11.19% | 29.28 ± 8.60% | 40716288 ± 0 | 40709473 ± 11805 | 40659072 ± 99102 |
| downsample_layers.2.0 | downsample | 27.64 ± 1.94% | 46.96 ± 8.85% | 46.61 ± 6.95% | 38188348 ± 519736 | 29247255 ± 3868332 | 29389041 ± 3046318 |
| downsample_layers.3.0 | downsample | 62.73 ± 0.45% | 50.95 ± 1.29% | 51.35 ± 0.64% | 29298234 ± 237491 | 33460416 ± 1364031 | 33193745 ± 1097809 |
| stages.0.0.dwconv | dwconv | 0.64 ± 1.11% | 0.92 ± 0.96% | 0.94 ± 0.82% | 848256 ± 0 | 848067 ± 327 | 846297 ± 3393 |
| stages.0.0.pw1_ttfs | pw1 | 14.84 ± 23.88% | 30.40 ± 8.54% | 28.00 ± 5.03% | 36096882 ± 248783 | 36115063 ± 370660 | 36199999 ± 320054 |
| stages.0.0.pw2_ttfs | pw2 | 18.41 ± 6.46% | 36.70 ± 7.46% | 34.36 ± 7.66% | 32146812 ± 9013242 | 26274290 ± 3224538 | 27180564 ± 1898782 |
| stages.0.1.dwconv | dwconv | 0.02 ± 0.02% | 6.64 ± 3.82% | 5.11 ± 1.61% | 848256 ± 0 | 848067 ± 327 | 846835 ± 2461 |
| stages.0.1.pw1_ttfs | pw1 | 0.31 ± 0.25% | 10.41 ± 13.05% | 14.79 ± 20.14% | 36629659 ± 549678 | 35832280 ± 613265 | 35390867 ± 212032 |
| stages.0.1.pw2_ttfs | pw2 | 19.34 ± 4.03% | 61.02 ± 4.37% | 59.45 ± 5.84% | 37633428 ± 95577 | 33819720 ± 4926286 | 32164435 ± 7604395 |
| stages.1.0.dwconv | dwconv | 7.04 ± 1.60% | 43.77 ± 2.50% | 39.60 ± 3.55% | 386108 ± 6815 | 280453 ± 45188 | 286750 ± 34458 |
| stages.1.0.pw1_ttfs | pw1 | 46.21 ± 0.77% | 41.58 ± 4.21% | 41.75 ± 3.52% | 34864302 ± 300513 | 37748601 ± 130 | 37745227 ± 6033 |
| stages.1.0.pw2_ttfs | pw2 | 97.47 ± 0.84% | 93.08 ± 1.51% | 93.33 ± 1.95% | 20306341 ± 290163 | 22051841 ± 1588211 | 21987980 ± 1331789 |
| stages.1.1.dwconv | dwconv | 12.14 ± 2.22% | 39.13 ± 3.74% | 37.57 ± 1.95% | 394827 ± 4849 | 293160 ± 45553 | 300491 ± 32112 |
| stages.1.1.pw1_ttfs | pw1 | 42.67 ± 2.86% | 25.32 ± 3.89% | 34.54 ± 8.12% | 37198626 ± 434939 | 37748736 ± 0 | 37748736 ± 0 |
| stages.1.1.pw2_ttfs | pw2 | 97.31 ± 1.59% | 92.11 ± 3.31% | 95.77 ± 2.35% | 21639541 ± 1076127 | 28191246 ± 1469206 | 24708817 ± 3067795 |
| stages.2.0.dwconv | dwconv | 24.42 ± 0.72% | 44.77 ± 3.97% | 45.18 ± 1.10% | 134489 ± 3725 | 98547 ± 16392 | 99169 ± 12838 |
| stages.2.0.pw1_ttfs | pw1 | 40.75 ± 4.21% | 37.19 ± 5.63% | 37.81 ± 6.14% | 37624670 ± 97505 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.0.pw2_ttfs | pw2 | 93.59 ± 2.28% | 89.59 ± 4.97% | 90.00 ± 4.09% | 22366989 ± 1590278 | 23709913 ± 2126853 | 23476087 ± 2316168 |
| stages.2.1.dwconv | dwconv | 31.76 ± 2.94% | 44.66 ± 1.52% | 44.42 ± 1.98% | 141055 ± 3642 | 109703 ± 19425 | 110287 ± 14713 |
| stages.2.1.pw1_ttfs | pw1 | 45.31 ± 1.12% | 31.94 ± 5.14% | 28.60 ± 3.42% | 37748589 ± 254 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.1.pw2_ttfs | pw2 | 96.48 ± 1.14% | 84.23 ± 1.43% | 78.18 ± 9.64% | 20643307 ± 424621 | 25691335 ± 1941061 | 26951824 ± 1291000 |
| stages.2.2.dwconv | dwconv | 33.63 ± 1.92% | 39.53 ± 4.80% | 42.02 ± 2.46% | 144732 ± 2753 | 125752 ± 16634 | 129561 ± 16172 |
| stages.2.2.pw1_ttfs | pw1 | 46.16 ± 0.30% | 24.95 ± 14.04% | 27.08 ± 9.18% | 37748736 ± 0 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.2.pw2_ttfs | pw2 | 96.80 ± 1.87% | 65.17 ± 34.45% | 78.06 ± 16.45% | 20323498 ± 113433 | 28329477 ± 5298869 | 27527053 ± 3462921 |
| stages.2.3.dwconv | dwconv | 37.54 ± 2.24% | 42.62 ± 3.09% | 42.47 ± 2.75% | 147302 ± 2341 | 144666 ± 29447 | 142570 ± 21768 |
| stages.2.3.pw1_ttfs | pw1 | 47.01 ± 1.38% | 24.61 ± 2.84% | 21.84 ± 3.66% | 37748736 ± 0 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.3.pw2_ttfs | pw2 | 98.29 ± 0.51% | 86.22 ± 4.25% | 84.29 ± 7.75% | 20001189 ± 523057 | 28457583 ± 1073182 | 29502638 ± 1383634 |
| stages.2.4.dwconv | dwconv | 35.50 ± 4.35% | 40.51 ± 2.30% | 37.71 ± 3.27% | 148664 ± 2382 | 152004 ± 23965 | 149045 ± 19193 |
| stages.2.4.pw1_ttfs | pw1 | 47.79 ± 3.86% | 20.48 ± 11.49% | 17.91 ± 5.27% | 37748677 ± 102 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.4.pw2_ttfs | pw2 | 95.53 ± 2.49% | 60.19 ± 25.46% | 60.07 ± 18.93% | 19707626 ± 1457215 | 30017513 ± 4339927 | 30989265 ± 1991281 |
| stages.2.5.dwconv | dwconv | 37.38 ± 1.90% | 39.24 ± 2.31% | 39.92 ± 3.83% | 151272 ± 988 | 171508 ± 6695 | 169497 ± 4392 |
| stages.2.5.pw1_ttfs | pw1 | 47.98 ± 0.23% | 29.17 ± 8.07% | 27.14 ± 8.09% | 37748736 ± 0 | 37748736 ± 0 | 37748736 ± 0 |
| stages.2.5.pw2_ttfs | pw2 | 97.48 ± 1.02% | 87.05 ± 7.29% | 85.37 ± 7.50% | 19636315 ± 88509 | 26735765 ± 3047451 | 27501844 ± 3052177 |
| stages.3.0.dwconv | dwconv | 27.56 ± 0.92% | 33.38 ± 1.98% | 34.06 ± 1.12% | 28756 ± 393 | 37967 ± 937 | 37684 ± 472 |
| stages.3.0.pw1_ttfs | pw1 | 53.05 ± 1.15% | 52.56 ± 0.64% | 52.54 ± 0.43% | 37683805 ± 42858 | 37748644 ± 158 | 37748731 ± 8 |
| stages.3.0.pw2_ttfs | pw2 | 83.73 ± 1.04% | 81.31 ± 1.63% | 81.36 ± 1.20% | 17721758 ± 435195 | 17909230 ± 243742 | 17916087 ± 163916 |
| stages.3.1.dwconv | dwconv | 36.91 ± 0.58% | 33.97 ± 1.36% | 34.27 ± 0.92% | 37316 ± 766 | 47010 ± 457 | 46716 ± 457 |
| stages.3.1.pw1_ttfs | pw1 | 52.33 ± 0.94% | 53.61 ± 1.18% | 53.85 ± 1.16% | 37740280 ± 7827 | 37730212 ± 14322 | 37738838 ± 7142 |
| stages.3.1.pw2_ttfs | pw2 | 74.03 ± 1.04% | 81.01 ± 0.67% | 81.42 ± 0.62% | 17995997 ± 355829 | 17509998 ± 444353 | 17420522 ± 440451 |

## Learnable-gate supplementary results

Gate values are `sigmoid(raw_gate)`; 0.5 gives normalized-sum behavior.

| Layer | Seed 42 mean ± channel SD | Seed 6543 mean ± channel SD | Seed 7777 mean ± channel SD |
|---|---:|---:|---:|
| stages.0.0 | 0.4855 ± 0.0093 | 0.4891 ± 0.0104 | 0.4895 ± 0.0153 |
| stages.0.1 | 0.5088 ± 0.0073 | 0.5042 ± 0.0085 | 0.5081 ± 0.0117 |
| stages.1.0 | 0.5103 ± 0.0086 | 0.5084 ± 0.0119 | 0.5081 ± 0.0113 |
| stages.1.1 | 0.5115 ± 0.0082 | 0.5125 ± 0.0130 | 0.5113 ± 0.0096 |
| stages.2.0 | 0.5026 ± 0.0087 | 0.5042 ± 0.0085 | 0.5026 ± 0.0076 |
| stages.2.1 | 0.5059 ± 0.0107 | 0.5076 ± 0.0124 | 0.5052 ± 0.0091 |
| stages.2.2 | 0.5115 ± 0.0131 | 0.5107 ± 0.0123 | 0.5077 ± 0.0147 |
| stages.2.3 | 0.5150 ± 0.0131 | 0.5145 ± 0.0173 | 0.5140 ± 0.0135 |
| stages.2.4 | 0.5166 ± 0.0131 | 0.5006 ± 0.0101 | 0.5119 ± 0.0146 |
| stages.2.5 | 0.5175 ± 0.0146 | 0.5090 ± 0.0087 | 0.5167 ± 0.0145 |
| stages.3.0 | 0.4894 ± 0.0061 | 0.4876 ± 0.0071 | 0.4864 ± 0.0074 |
| stages.3.1 | 0.4852 ± 0.0096 | 0.4795 ± 0.0110 | 0.4746 ± 0.0109 |

## Response to Reviewer 1, Comment 4

We added a controlled residual-fusion ablation on CIFAR-100 using the same Fully-TTFS architecture, matched dense initialization, data splits, training settings, and three random seeds. Normalized sum is the arithmetic mean of the identity and residual-branch spike times. The learnable gate is initialized at 0.5 and therefore begins with exactly the same behavior as normalized sum. Minimum fusion selects the earliest spike time and consequently preserves the native TTFS interpretation that an earlier event represents stronger evidence. The resulting accuracy, activation sparsity, and theoretical event-driven SynOps are reported above as mean ± sample standard deviation.
