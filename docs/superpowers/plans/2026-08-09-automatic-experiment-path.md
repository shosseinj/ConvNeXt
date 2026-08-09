# Automatic Experiment Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the training experiment name and output directory from dataset, base experiment name, and seed when `--output_dir` is omitted.

**Architecture:** Add a small pure resolver near argument parsing, then normalize the parsed arguments once before the rest of training consumes them. Preserve explicit `--output_dir` as an override and test the resolver without starting dataset loading or GPU training.

**Tech Stack:** Python 3, argparse, pathlib, unittest/pytest-compatible tests.

## Global Constraints

- Full experiment name is `{dataset}_{experiment_name}_seed{seed}`.
- Automatic output directory is `results/{dataset}/{experiment_name}/seed_{seed}`.
- Explicit `--output_dir` overrides only the directory.
- Model architecture and training behavior must not change.

---

### Task 1: Resolve experiment identity and path

**Files:**
- Create: `tests/test_automatic_experiment_path.py`
- Modify: `train_continuous_ttfs_cifar10_32x32_stem1.py:80-178`

**Interfaces:**
- Consumes: normalized dataset string, raw base experiment name, integer seed, optional output directory.
- Produces: `resolve_experiment_identity(dataset, experiment_name, seed, output_dir) -> tuple[str, str]` containing the derived full name and selected directory.

- [ ] **Step 1: Write failing resolver tests**

```python
def test_derives_name_and_output_path():
    name, path = resolve_experiment_identity("cifar10", "ttfs_dwconv_downsample", 8888, "")
    assert name == "cifar10_ttfs_dwconv_downsample_seed8888"
    assert path == "results/cifar10/ttfs_dwconv_downsample/seed_8888"

def test_explicit_output_path_overrides_derived_path():
    name, path = resolve_experiment_identity("cifar10", "ttfs_dwconv_downsample", 7, "custom/run")
    assert name == "cifar10_ttfs_dwconv_downsample_seed7"
    assert path == "custom/run"
```

Add parameterized invalid-name cases for empty names, `.`, `..`, absolute paths, `/`, and `\\`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_automatic_experiment_path.py -q`
Expected: collection fails because `resolve_experiment_identity` does not exist.

- [ ] **Step 3: Implement the resolver and argument normalization**

```python
def resolve_experiment_identity(dataset, experiment_name, seed, output_dir):
    base_name = experiment_name.strip()
    # Reject empty and path-like names.
    full_name = f"{dataset}_{base_name}_seed{seed}"
    selected_output = output_dir.strip() or str(
        Path("results") / dataset / base_name / f"seed_{seed}"
    )
    return full_name, selected_output
```

Set `--output_dir` default to an empty string. After existing dataset normalization, replace `args.experiment_name` and `args.output_dir` using the resolver; convert resolver `ValueError` into `parser.error(...)`.

- [ ] **Step 4: Verify GREEN and CLI parsing**

Run: `python -m pytest tests/test_automatic_experiment_path.py -q`
Expected: all tests pass.

Run the trainer through a parse-only test with `--dataset cifar10 --experiment_name ttfs_dwconv_downsample --seed 8888` and assert the literal derived values.

- [ ] **Step 5: Run focused regression checks**

Run: `python -m py_compile train_continuous_ttfs_cifar10_32x32_stem1.py tests/test_automatic_experiment_path.py`
Expected: exit code 0.

Run: `python -m pytest tests/test_automatic_experiment_path.py -q`
Expected: all tests pass without warnings or errors.
