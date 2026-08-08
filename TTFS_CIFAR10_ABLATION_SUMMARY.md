# Continuous TTFS ConvNeXt CIFAR-10 Ablation Summary

## Reference model

The best reliable student configuration was:

```text
Input resolution: 32x32
Stem: Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
Downsampling: Conv2d(kernel_size=3, stride=2, padding=1)
Dimensions: 96,192,384,768
Depths: 2,2,6,2
Depthwise kernel: 3
PW1: continuous TTFS
PW2: continuous TTFS
Residual: torch.minimum(tj, t_out)
Internal normalization: score_layernorm
Final score normalization: enabled
Stage-delay initialization: 0.05,0.02,0.01,0.01
Head dropout: 0.1
Spike dropout: 0
DropPath: 0
Mixup: 0.2
CutMix: 1.0
RandAugment: 2 operations, magnitude 9
Random Erasing: 0.1
EMA: enabled
```

Best result:

```text
Validation accuracy: 94.40%
Best epoch: 234
Test accuracy without TTA: approximately 93.83%
Test accuracy with horizontal-flip TTA: 94.17%
```

Preserved checkpoint:

```text
results\cifar10\fully_ttfs\clean_finetune_from_94_36\seed_42\preserved_best_94_40\best_val_94_40_epoch_234.pth
```

---

## Changes that improved performance

### 1. Native CIFAR-10 resolution and stride-1 stem

Changed from an ImageNet-style input/stem to:

```python
Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1)
```

Spatial schedule:

```text
32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4
```

This made the model appropriate for native CIFAR-10 and avoided destroying spatial information in the stem.

**Status:** Improvement; retained in the best model.

### 2. Three-by-three downsampling

Changed downsampling from:

```python
Conv2d(..., kernel_size=2, stride=2)
```

to:

```python
Conv2d(..., kernel_size=3, stride=2, padding=1)
```

**Status:** Improvement as part of the successful CIFAR architecture; retained.

### 3. Restoring larger channel dimensions

The smaller model used:

```text
dims = 64,128,256,512
depths = 2,2,6,2
```

It underperformed. Returning to:

```text
dims = 96,192,384,768
depths = 2,2,6,2
```

provided sufficient capacity and ultimately reached 94.40% validation accuracy.

**Status:** Clear improvement; retained.

### 4. Internal score-space LayerNorm

Normalization was applied after the depthwise convolution in score space:

```python
scores = -x
scores = self.norm(scores)
x = -scores
```

It was not applied directly to the incoming spike-time tensor. Internal LayerNorm alone initially trained poorly, but it became useful as part of the final combination with final score normalization, the corrected architecture, and the successful augmentation recipe.

**Status:** Useful as part of the final combined configuration; not independently proven to account for the complete gain; retained.

### 5. Final score LayerNorm

Applied after pooled spike times were converted to scores:

```python
x_pool = self.forward_features(x_t)
scores = -x_pool
scores = self.final_norm(scores)
logits = self.head(scores)
```

The successful 94.36-94.40% model used this normalization.

**Status:** Improvement as part of the best configuration; retained.

### 6. Strong augmentation during main training

Successful configuration:

```text
Mixup alpha: 0.2
CutMix alpha: 1.0
RandAugment: enabled
RandAugment operations: 2
RandAugment magnitude: 9
Random Erasing: 0.1
```

The displayed training accuracy was around 75-78%, but that was mixed-label accuracy, not clean training accuracy.

**Status:** Clear generalization improvement; retained during main training.

### 7. EMA

Successful model validation used EMA weights:

```text
EMA decay: 0.9998
```

Faster EMA values such as `0.9995` were later used for short fine-tuning experiments, but they did not improve the best result.

**Status:** Useful in the successful training configuration; no controlled independent no-EMA comparison; retained.

### 8. ReduceLROnPlateau and early stopping

The original cosine run stopped around:

```text
Best validation accuracy: 89.30%
```

Later runs using plateau scheduling, the better architecture, normalization, and augmentation reached approximately 94.4%. Because several changes occurred together, the improvement cannot be attributed only to the scheduler.

**Status:** Useful training-control change; not independently isolated; retained.

### 9. Horizontal-flip test-time augmentation

Results from the same checkpoint:

```text
Without TTA: 93.83% test accuracy
With horizontal-flip TTA: 94.17% test accuracy
Improvement: +0.34 percentage points
```

**Status:** Confirmed improvement; inference cost approximately doubled; does not improve the trained checkpoint itself.

---

## Ablations that did not improve performance

### 1. Smaller model

Configuration:

```text
dims = 64,128,256,512
depths = 2,2,6,2
classifier input = 512
```

Observed behavior included:

```text
Training accuracy: approximately 87%
Validation accuracy: approximately 77%
```

**Conclusion:** Insufficient capacity for the target accuracy; failed.

### 2. Original or poorly adapted 32x32 configurations

Earlier native-resolution configurations produced validation results around 85-89%. They lacked the final successful combination of architecture, normalization, augmentation, and optimization.

**Conclusion:** Failed to reach the target.

### 3. Deeper model

Configuration:

```text
dims = 96,192,384,768
depths = 3,3,9,3
depthwise kernel = 3
learning rate = 3e-4
```

Result:

```text
Best validation accuracy: 94.12%
Best epoch: 168
Test accuracy: approximately 93.95%
Reference validation accuracy: 94.40%
```

**Conclusion:** More depth increased computation but reduced accuracy; failed.

Preserved ablation checkpoint:

```text
results\cifar10\fully_ttfs\score_layernorm_final_norm_aug\depths_3_3_9_3\kernel_3\seed_42\lr3e4\preserved_best_94_12\best_val_94_12_epoch_168.pth
```

### 4. Depthwise kernel size 5

Configuration:

```text
depths = 2,2,6,2
depthwise kernel = 5
```

Observed result:

```text
Best validation accuracy: approximately 93.70%
Kernel 3 reference: 94.40%
```

**Conclusion:** Kernel 5 was worse; failed.

### 5. Larger depthwise kernel consideration

Kernel size 7 was supported, but the available kernel-size results gave no evidence that increasing the receptive field beyond kernel 3 would help. Kernel 5 already underperformed.

**Conclusion:** Kernel 3 remained the best choice.

### 6. GRN module

A ConvNeXt-V2-style GRN module was added in score space between the TTFS pointwise operations.

Result:

```text
Best validation accuracy: 94.14%
Test accuracy: 93.45%
Reference validation accuracy: 94.40%
```

**Conclusion:** GRN did not improve the model; failed.

### 7. Score-bounded spatial remapping

A bounded score/time remapping experiment was attempted.

Observed behavior:

```text
Accuracy remained very low
Approximately 51% around epoch 40 in one run
```

It also produced an unsupported-argument issue in one restored trainer version:

```text
unrecognized arguments: --spatial_time_mode score_bounded
```

**Conclusion:** Strong failure; discarded.

### 8. Clean fine-tuning

After reaching approximately 94%, augmentation was removed and training continued with a small learning rate.

Observed behavior:

```text
Training accuracy: approximately 99.8-99.9%
Validation accuracy: declined or remained near 94.1-94.3%
```

**Conclusion:** Severe overfitting; failed.

### 9. Mild fine-tuning

Configuration included:

```text
Learning rate: 5e-6
Weight decay: 0.02
Mixup alpha: 0.05
CutMix: disabled
RandAugment: disabled
Random Erasing: disabled
```

Observed behavior:

```text
Training accuracy: approximately 96-97.5%
Best validation accuracy remained 94.40%
Later validation accuracy: approximately 94.24-94.28%
```

**Conclusion:** Reduced overfitting relative to completely clean training but did not improve validation accuracy; failed.

### 10. Reducing or removing Mixup and CutMix near 94%

The intent was to allow the model to refine clean classifications after reaching approximately 94%.

Observed behavior:

```text
Training accuracy increased strongly
Validation accuracy remained flat or declined
```

**Conclusion:** The model overfit instead of improving generalization; failed.

### 11. Increasing learning rate during fine-tuning

The fine-tuning learning rate was increased from approximately `1e-5` to `2e-5`.

Observed behavior:

```text
Validation loss decreased
Validation accuracy remained around 94.2-94.36%
```

**Conclusion:** The model was not simply trapped because of an excessively small learning rate; failed.

### 12. Knowledge distillation

Teacher:

```text
Architecture: WideResNet-28-10
Teacher validation accuracy: 97.0%
```

Student distillation configuration:

```text
Student initialized from 94.40% EMA checkpoint
Distillation alpha: 0.5
Temperature: 4.0
Strong augmentation
Learning rate: first 1e-5, then 2e-5 experiment
```

Result:

```text
Best student validation accuracy: 94.36%
Final validation accuracy: approximately 94.26%
Test accuracy with the configured evaluation path: approximately 94.15%
Validation loss: approximately 0.462 -> 0.440
```

**Conclusion:** Distillation improved calibration and confidence but did not correct additional classifications; failed.

### 13. Straight-through soft-min residual gradient

Training used exact hard-min forward values with soft-min surrogate gradients. Evaluation continued to use exact:

```python
torch.minimum(tj, t_out)
```

Validation confirmed that evaluation logits were bitwise identical before training and that gradients were finite.

Observed result:

```text
No validation improvement over 94.40%
```

**Conclusion:** Hard-min backward routing was not the only limiting factor; failed and discarded.

### 14. Training-only deep supervision

Added auxiliary classifiers after stages 1 and 2:

```text
Stage 1 auxiliary weight: 0.15
Stage 2 auxiliary weight: 0.10
Base-model LR: 5e-6
Auxiliary-head LR: 5e-5
```

Results:

```text
Original validation accuracy: 94.40%
Best deep-supervision validation accuracy: 94.42%
Difference: +0.02%, equivalent to one validation image
Best epoch: 7
Final validation accuracy: 94.22%
Test accuracy: 93.82%
Auxiliary loss: approximately 0.571 -> 0.478
```

**Conclusion:** The auxiliary heads learned successfully, but the final classifier did not improve. The +0.02% validation difference is statistical noise; failed.

### 15. Adding more TTFS blocks or increasing capacity

Greater depth was tested and underperformed. Additional TTFS layers would also increase the number of hard-min residual decisions and computation.

**Conclusion:** No evidence that more TTFS blocks improve accuracy; the deeper model failed.

### 16. Head-only clean refinement

Small-learning-rate refinement with reduced augmentation improved clean training accuracy but did not improve validation accuracy.

**Conclusion:** Classifier refinement was not the primary limitation; failed.

---

## Discussed but not established as successful ablations

### Positive-only weights

Potential implementation:

```python
torch.relu(weight)
```

Concerns:

```text
Negative weights become dead
Model expressiveness is reduced
ReLU-constrained weights can receive zero gradients
```

**Status:** Not recommended; no confirmed accuracy improvement.

### Replacing the minimum residual

Alternatives such as `add`, `max`, a logical OR-like operation, and soft minimum were discussed. The straight-through soft-min experiment failed. Directly replacing `minimum` would also change the intended TTFS residual semantics.

**Status:** No successful alternative found.

### Delay-only fine-tuning

The best checkpoint learned approximately:

| Stage | D_mid | D_out |
|---|---:|---:|
| 0 | 0.1553 | 0.1553 |
| 1 | 0.0992 | 0.0985 |
| 2 | 0.0696 | 0.0686 |
| 3 | 0.0683 | 0.0681 |

Initial delays were:

```text
0.05,0.02,0.01,0.01
```

This proves that the delay parameters were learning and were not saturated.

**Status:** Proposed but not yet confirmed. Changing `--stage_delays` while resuming does not work because checkpoint loading overwrites them. A proper experiment requires a separate delay optimizer group.

---

## Final conclusions

### Confirmed useful components

```text
Native 32x32 input
Stride-1 3x3 CIFAR stem
3x3 stride-2 downsampling
dims 96,192,384,768
depths 2,2,6,2
depthwise kernel 3
fully TTFS PW1 and PW2
hard minimum residual
score-space LayerNorm
final score LayerNorm
strong Mixup/CutMix/RandAugment/Random Erasing
EMA
plateau scheduling and early stopping
horizontal-flip TTA for final evaluation
```

### Confirmed unsuccessful approaches

```text
Smaller 64,128,256,512 model
Deeper 3,3,9,3 model
Depthwise kernel 5
GRN
Score-bounded time remapping
Clean fine-tuning
Mild fine-tuning
Removing augmentation near 94%
Higher fine-tuning learning rate
Knowledge distillation from a 97% teacher
Straight-through soft-min gradients
Training-only deep supervision
Additional model depth
```

### Best retained result

```text
Validation accuracy: 94.40%
Test accuracy without TTA: approximately 93.83%
Test accuracy with horizontal-flip TTA: 94.17%
```

The accumulated evidence indicates that the current approximately 94.4% ceiling is not primarily caused by model size, kernel size, augmentation strength, scheduler choice, hard-min gradients, intermediate supervision, classifier calibration, or lack of teacher knowledge. The remaining limitation is likely in the continuous TTFS transformation itself or in how spatial convolutions operate directly on spike-time tensors.
