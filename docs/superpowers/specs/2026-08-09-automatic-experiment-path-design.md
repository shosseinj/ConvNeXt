# Automatic Experiment Naming and Output Path

## Scope

Update `train_continuous_ttfs_cifar10_32x32_stem1.py` so a user supplies the base experiment name, dataset, and seed without needing to supply an output directory.

## Naming contract

Given:

- `--dataset cifar10`
- `--experiment_name ttfs_dwconv_downsample`
- `--seed 8888`

the trainer derives:

- Full experiment name: `cifar10_ttfs_dwconv_downsample_seed8888`
- Output directory: `results/cifar10/ttfs_dwconv_downsample/seed_8888`

The general forms are:

- Full experiment name: `{dataset}_{experiment_name}_seed{seed}`
- Output directory: `results/{dataset}/{experiment_name}/seed_{seed}`

The base experiment name remains unchanged for path construction. The derived full name is used in experiment metadata and reports.

## CLI compatibility and validation

`--output_dir` becomes optional. When explicitly supplied, it overrides only the automatically generated directory; the full experiment name is still derived from dataset, base experiment name, and seed.

The trainer rejects an empty base experiment name when no output directory can provide a legacy fallback. It also rejects names that could escape or add path hierarchy, including absolute paths, `.` or `..`, and names containing directory separators.

## Resume behavior

Resume continues to use `--resume` for the checkpoint. If `--output_dir` is omitted, resumed artifacts go to the same deterministic directory derived from dataset, experiment name, and seed. Existing architecture compatibility checks remain unchanged.

## Verification

Automated tests will verify:

1. CIFAR-10, `ttfs_dwconv_downsample`, and seed 8888 produce the specified full name and directory.
2. An explicit output directory remains an override.
3. Invalid or empty experiment names fail clearly.
4. The trainer's CLI help and argument parsing work without `--output_dir`.

No training run or model architecture behavior is changed by this feature.
