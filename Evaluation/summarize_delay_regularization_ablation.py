#!/usr/bin/env python3
"""Summarize the three-seed CIFAR-100 delay-regularization ablation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import torch


SEEDS = (42, 6543, 7777)
WEIGHTS = (0.0, 0.01, 0.1)
LABELS = {0.0: "Reference", 0.01: "lambda_0p01", 0.1: "lambda_0p1"}
STAGE_PATTERN = re.compile(r"^stages\.(\d+)\..*\.(D_mid|D_out)$")


def run_directory(project_root: Path, weight: float, seed: int) -> Path:
    if weight == 0.0:
        return project_root / "results" / "cifar100" / "downsample_dense_dwconv_dense" / f"seed_{seed}"
    return (
        project_root
        / "results"
        / "cifar100"
        / "ablation_delay_regularization"
        / LABELS[weight]
        / f"seed_{seed}"
    )


def checkpoint_delay_statistics(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {path}")
    stage_values = {stage: {"mid": [], "out": []} for stage in range(4)}
    all_values = []
    for name, raw_delay in state.items():
        match = STAGE_PATTERN.match(name)
        if match is None:
            continue
        stage = int(match.group(1))
        kind = "mid" if match.group(2) == "D_mid" else "out"
        effective = 0.9 * torch.sigmoid(raw_delay.detach().float())
        stage_values[stage][kind].extend(effective.reshape(-1).tolist())
        all_values.extend(effective.reshape(-1).tolist())
    if not all_values:
        raise ValueError(f"No D_mid/D_out parameters found in {path}")
    per_stage = []
    for stage in range(4):
        values = stage_values[stage]
        if not values["mid"] or not values["out"]:
            raise ValueError(f"Incomplete delay parameters for stage {stage} in {path}")
        per_stage.append(
            {
                "stage": stage,
                "mid": statistics.fmean(values["mid"]),
                "out": statistics.fmean(values["out"]),
            }
        )
    return {"overall_mean": statistics.fmean(all_values), "per_stage": per_stage}


def load_run(project_root: Path, weight: float, seed: int) -> dict:
    directory = run_directory(project_root, weight, seed)
    summary_path = directory / "training_summary.json"
    checkpoint_path = directory / "best_checkpoint.pth"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete delay-regularization run: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded_weight = float(summary.get("delay_regularization_weight", 0.0))
    if recorded_weight != weight:
        raise ValueError(
            f"Regularization weight mismatch in {summary_path}: "
            f"expected={weight}, recorded={recorded_weight}"
        )
    test_metrics = summary.get("test_metrics") or {}
    if "best_validation_accuracy" not in summary or "accuracy" not in test_metrics:
        raise ValueError(f"Missing accuracy metrics in {summary_path}")
    delays = checkpoint_delay_statistics(checkpoint_path)
    return {
        "weight": weight,
        "label": LABELS[weight],
        "seed": seed,
        "validation_accuracy": float(summary["best_validation_accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "overall_mean_delay": delays["overall_mean"],
        "per_stage_delays": delays["per_stage"],
        "directory": str(directory.resolve()),
    }


def aggregate(rows: list[dict]) -> dict:
    def metric(name: str) -> dict:
        values = [row[name] for row in rows]
        return {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
        }

    return {
        "weight": rows[0]["weight"],
        "label": rows[0]["label"],
        "seeds": [row["seed"] for row in rows],
        "validation_accuracy": metric("validation_accuracy"),
        "test_accuracy": metric("test_accuracy"),
        "overall_mean_delay": metric("overall_mean_delay"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "results" / "cifar100" / "ablation_delay_regularization"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(project_root, weight, seed) for weight in WEIGHTS for seed in SEEDS]
    aggregates = [aggregate([row for row in runs if row["weight"] == weight]) for weight in WEIGHTS]
    reference = aggregates[0]
    for item in aggregates:
        item["test_accuracy_delta_vs_reference"] = (
            item["test_accuracy"]["mean"] - reference["test_accuracy"]["mean"]
        )
        item["mean_delay_reduction_vs_reference"] = (
            reference["overall_mean_delay"]["mean"] - item["overall_mean_delay"]["mean"]
        )

    report = {
        "dataset": "CIFAR-100",
        "regularization_definition": "mean effective bounded D_mid/D_out delay",
        "seeds": list(SEEDS),
        "runs": runs,
        "aggregates": aggregates,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "weight", "label", "validation_mean", "validation_sample_std",
            "test_mean", "test_sample_std", "delay_mean", "delay_sample_std",
            "test_accuracy_delta_vs_reference", "mean_delay_reduction_vs_reference",
        ))
        writer.writeheader()
        for item in aggregates:
            writer.writerow({
                "weight": item["weight"], "label": item["label"],
                "validation_mean": item["validation_accuracy"]["mean"],
                "validation_sample_std": item["validation_accuracy"]["sample_std"],
                "test_mean": item["test_accuracy"]["mean"],
                "test_sample_std": item["test_accuracy"]["sample_std"],
                "delay_mean": item["overall_mean_delay"]["mean"],
                "delay_sample_std": item["overall_mean_delay"]["sample_std"],
                "test_accuracy_delta_vs_reference": item["test_accuracy_delta_vs_reference"],
                "mean_delay_reduction_vs_reference": item["mean_delay_reduction_vs_reference"],
            })

    lines = [
        "# Delay Regularization Ablation", "",
        "| Weight | Validation accuracy | Test accuracy | Mean effective delay | Test delta | Delay reduction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['weight']:g} | "
            f"{item['validation_accuracy']['mean']:.2f} ± {item['validation_accuracy']['sample_std']:.2f}% | "
            f"{item['test_accuracy']['mean']:.2f} ± {item['test_accuracy']['sample_std']:.2f}% | "
            f"{item['overall_mean_delay']['mean']:.6f} ± {item['overall_mean_delay']['sample_std']:.6f} | "
            f"{item['test_accuracy_delta_vs_reference']:+.2f} pp | "
            f"{item['mean_delay_reduction_vs_reference']:.6f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
