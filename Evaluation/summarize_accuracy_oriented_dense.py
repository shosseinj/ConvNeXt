"""Summarize the six completed fully-dense CIFAR accuracy runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "accuracy_oriented_results"
SEEDS = (42, 6543, 7777)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def formatted(values):
    mean, deviation = mean_std(values)
    return f"{mean:.2f} ± {deviation:.2f}%"


def load_run(dataset, seed):
    run = RESULTS / dataset / "fully_dense" / f"seed_{seed}"
    summary_path = run / "training_summary.json"
    tta_path = run / "evaluation_tta" / "tta_evaluation.json"
    if not summary_path.is_file() or not tta_path.is_file():
        raise FileNotFoundError(f"Incomplete run: {run}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tta = json.loads(tta_path.read_text(encoding="utf-8"))
    expected_checkpoint = (run / "best_checkpoint.pth").resolve()
    if Path(tta["checkpoint"]).resolve() != expected_checkpoint:
        raise ValueError(f"TTA did not evaluate the selected best checkpoint: {run}")
    if any(tta["integrity"].values()):
        raise ValueError(f"Checkpoint integrity failed: {run}")
    modes = {row["mode"]: row for row in tta["results"]}
    if set(modes) != {"none", "flip_shift"}:
        raise ValueError(f"Required TTA modes are missing: {run}")
    if modes["none"]["samples"] != 10000 or modes["none"]["views_per_model"] != 1:
        raise ValueError(f"Invalid single-view accounting: {run}")
    if modes["flip_shift"]["samples"] != 10000 or modes["flip_shift"]["views_per_model"] != 10:
        raise ValueError(f"Invalid 10-view accounting: {run}")
    return {
        "seed": seed, "validation": float(summary["best_validation_accuracy"]),
        "single": float(modes["none"]["accuracy"]), "tta": float(modes["flip_shift"]["accuracy"]),
        "gain": float(modes["flip_shift"]["accuracy"] - modes["none"]["accuracy"]),
        "best_epoch": int(summary["best_epoch"]),
    }


def main():
    all_rows = {dataset: [load_run(dataset, seed) for seed in SEEDS] for dataset in ("cifar10", "cifar100")}
    lines = [
        "# Fully-Dense Accuracy-Oriented ConvNeXt", "",
        "All checkpoints were selected exclusively by EMA validation accuracy. Test-time augmentation uses the predeclared 10-view flip-shift protocol.", "",
        "| Dataset | Single-view test accuracy | 10-view TTA test accuracy | TTA gain |", "|---|---:|---:|---:|",
    ]
    for dataset, rows in all_rows.items():
        label = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
        lines.append(f"| {label} | {formatted([r['single'] for r in rows])} | **{formatted([r['tta'] for r in rows])}** | {formatted([r['gain'] for r in rows])} |")
    for dataset, rows in all_rows.items():
        label = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
        lines += ["", f"## {label} per seed", "", "| Seed | Best validation | Best epoch | Single-view test | 10-view TTA | Gain |", "|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['seed']} | {row['validation']:.2f}% | {row['best_epoch']} | {row['single']:.2f}% | {row['tta']:.2f}% | {row['gain']:+.2f}% |")
    report = RESULTS / "FULLY_DENSE_ACCURACY_SUMMARY.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
