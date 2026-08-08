# CIFAR-100 TTFS Convolution Evaluation

- Seed: 42
- Best epoch: 190
- Best validation accuracy: 74.04%
- Standard single-view test accuracy: 73.03%
- Standard single-view test loss: 1.1145
- Horizontal-flip TTA test accuracy: 74.03%
- Horizontal-flip TTA test loss: 1.0919
- Flip + shift TTA test accuracy: 74.32%
- Flip + shift TTA test loss: 1.0790
- Checkpoint weights: EMA
- Checkpoint integrity: strict load passed
- Confusion matrix: not generated

## Primary reported result

```text
Validation accuracy: 74.04%
Test accuracy:       73.03%
```

The 74.32% result uses 10-view flip-and-shift test-time augmentation and should
be reported separately from the standard single-view test accuracy.

## Comparison with previous dense-convolution baseline

| Metric | Previous baseline | TTFS depthwise + downsampling | Change |
|---|---:|---:|---:|
| Best validation accuracy | 73.02% | 74.04% | +1.02 pp |
| Standard test accuracy | 72.54% | 73.03% | +0.49 pp |
| Flip TTA test accuracy | 73.21% | 74.03% | +0.82 pp |
| Flip + shift TTA test accuracy | 73.47% | 74.32% | +0.85 pp |
