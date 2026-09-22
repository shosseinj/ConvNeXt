# TTFS ConvNeXt Experiments [This was my idea]

This repository contains research experiments on adapting ConvNeXt-style architectures to time-to-first-spike (TTFS) computation. The code is used to study how convolutional operations, residual connections, normalization, temporal delays, and output scoring behave when activations are represented by spike timing rather than conventional dense feature values.

## Research Questions
The experiments focus on several questions:

- Which ConvNeXt operations can be expressed directly in the TTFS domain?
- How should residual fusion be implemented when earlier spike times represent stronger activations?
- How do learnable or fixed stage delays affect optimization and accuracy?
- What is the relationship between classification accuracy, spike sparsity, latency, and approximate synaptic operation counts?
- Which architectural components remain necessary when moving from dense to temporal representations?

## Experimental Scope

The current code contains CIFAR-oriented experiments with configurable stage dimensions, depths, TTFS normalization, temporal delays, data augmentation, EMA, and regularization. Several scripts are dedicated to ablation studies and evaluation of sparse/event-driven computation.

Relevant files include:

- `train_continuous_ttfs_cifar10_32x32_stem1.py` — main TTFS training path
- `TTFS_CIFAR10_ABLATION_SUMMARY.md` — ablation notes and summaries
- `evaluate_sparsity_synops.py` — spike activity and approximate SynOps analysis
- `evaluate_ttfs_cifar10_tta.py` — evaluation utilities
- `Evaluation/` and result directories — experiment outputs

## Example Training

A typical experiment configures the temporal form of depthwise convolution, downsampling, pointwise layers, residual behavior, stage delays, and the TTFS normalization strategy. Exact commands used for individual runs are kept in the repository for reproducibility.

## Metrics

The evaluation code is designed to inspect both conventional prediction quality and temporal-computation characteristics. Approximate SynOps are used as a computational proxy and should not be interpreted as direct hardware energy measurements.

## Upstream Attribution

The repository retains training utilities and backbone code derived from Meta's [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) implementation. The work here concerns TTFS adaptations and experiments built on that baseline, not authorship of the original ConvNeXt architecture.

## Status

This is an active experimental codebase rather than a general-purpose ConvNeXt implementation. Results should be read together with the corresponding experiment configuration and random seed.


## Goal

The repository provides a controlled workspace for testing analytic and continuous TTFS transformations of ConvNeXt components while tracking accuracy, latency, spike activity, and approximate operation counts.

## Installation

No pinned environment is supplied. Install a PyTorch build appropriate for the machine, followed by torchvision, timm, and the scientific dependencies imported by the selected script:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision timm numpy
python train_continuous_ttfs_cifar10_32x32_stem1.py --help
```

## Working with the Repository

Choose one training entry point and keep its dataset, dimensions, depths, temporal range, delay configuration, normalization mode, seed, and output directory fixed when comparing an ablation. Evaluation scripts consume the resulting checkpoints and write separate summaries. Do not compare metrics across result directories unless their configuration files establish the same protocol.
