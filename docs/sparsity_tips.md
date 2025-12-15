# Increasing Spike / Activation Sparsity (ConvNeXt SNN TTFS)

This repo now includes a few mechanisms to increase sparsity (percentage of silent neurons) during TTFS spiking ConvNeXt training/evaluation.

Recommended levers (trade-offs: sparsity vs accuracy):

- TTFS encoding nonlinearity: `--ttfs_power` (default=1.0)

  - Values < 1 push spike times later (more neurons will be silent), e.g. `--ttfs_power 0.7`.

- Spike-rate regularization: `--lambda_spike` (default=0.0)

  - Adds a term to the loss proportional to the average firing rate across spiking blocks; increasing it penalizes firing and yields sparser responses.

- Delay regularization (already present): `--lambda_delay` (default=1e-1)

  - Encourages larger learned per-output delays (`D_mid`, `D_out`), which shifts spikes later and increases silence.

- Enforce non-negative pointwise weights: `--ttfs_force_pos_weights true`

  - Applies ReLU to pointwise (1x1) weights used in TTFS mapping. This can induce effective zeros in weights and help pruning.

- Learned per-layer delays: implemented in `SpikingBlock` (`D_mid` / `D_out`) and used during forward pass. These were previously unused.

Practical recipe to get started:

1. Start with `--ttfs_power 0.8 --lambda_spike 1e-3 --lambda_delay 1e-1`
2. Monitor validation accuracy and sparsity (see `scripts/evaluate_sparsity.py`).
3. If accuracy drops too much, reduce `--lambda_spike` or raise `--ttfs_power` toward 1.0.

Advanced options:

- Structured pruning: after measuring per-channel silence, prune channels with consistently high silence and fine-tune.
- Iterative magnitude pruning (IMP): combine with the spike regularizer for best sparse performance.

If you'd like, I can add an automated pruning script that uses per-channel silence statistics to remove channels with high silence and then fine-tunes the model.
