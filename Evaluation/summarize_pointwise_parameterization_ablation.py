#!/usr/bin/env python3
"""Summarize signed, post-hoc ReLU, and from-scratch Softplus Fully-TTFS runs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


SEEDS = (42, 6543, 7777)
CONDITIONS = (
    ("signed", "Signed Fully-TTFS", "fully_ttfs"),
    (
        "relu_posthoc",
        "Hard ReLU (post-hoc adaptation)",
        "ablation_nonnegative_pointwise",
    ),
    (
        "softplus_scratch",
        "Softplus (from scratch)",
        "ablation_nonnegative_softplus",
    ),
)


def required_float(pattern: str, text: str, path: Path, label: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Missing {label} in {path}")
    return float(match.group(1).replace(",", ""))


def load_run(root: Path, condition: tuple[str, str, str], seed: int) -> dict:
    key, label, folder = condition
    directory = root / folder / f"seed_{seed}"
    summary_path = directory / "training_summary.json"
    report_path = directory / "activation_sparsity.md"
    checkpoint_path = directory / "best_checkpoint.pth"
    missing = [
        str(path)
        for path in (summary_path, report_path, checkpoint_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Incomplete pointwise ablation: " + ", ".join(missing))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    test_metrics = summary.get("test_metrics") or {}
    return {
        "condition": key,
        "label": label,
        "seed": seed,
        "validation_accuracy": float(summary["best_validation_accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "evaluated_accuracy": required_float(
            r"Classification accuracy:\s+([\d.]+)%",
            report,
            report_path,
            "classification accuracy",
        ),
        "activation_sparsity": required_float(
            r"Activation sparsity:\s+([\d.]+)%",
            report,
            report_path,
            "activation sparsity",
        ),
        "theoretical_synops": required_float(
            r"Theoretical SynOps:\s+([\d,]+) per sample",
            report,
            report_path,
            "theoretical SynOps",
        ),
        "directory": str(directory.resolve()),
    }


def metric(rows: list[dict], name: str) -> dict:
    values = [row[name] for row in rows]
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "fine_tune_results_v3" / "cifar100"
    output_dir = root / "ablation_nonnegative_softplus"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(root, condition, seed) for condition in CONDITIONS for seed in SEEDS]
    aggregates = []
    for key, label, _ in CONDITIONS:
        rows = [row for row in runs if row["condition"] == key]
        aggregates.append({
            "condition": key,
            "label": label,
            "seeds": list(SEEDS),
            "validation_accuracy": metric(rows, "validation_accuracy"),
            "test_accuracy": metric(rows, "test_accuracy"),
            "activation_sparsity": metric(rows, "activation_sparsity"),
            "theoretical_synops": metric(rows, "theoretical_synops"),
        })

    payload = {"seeds": list(SEEDS), "runs": runs, "aggregates": aggregates}
    (output_dir / "POINTWISE_PARAMETERIZATION_ABLATION.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "POINTWISE_PARAMETERIZATION_ABLATION.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "condition", "label", "validation_mean", "validation_sample_std",
            "test_mean", "test_sample_std", "sparsity_mean", "sparsity_sample_std",
            "synops_mean", "synops_sample_std",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in aggregates:
            writer.writerow({
                "condition": item["condition"], "label": item["label"],
                "validation_mean": item["validation_accuracy"]["mean"],
                "validation_sample_std": item["validation_accuracy"]["sample_std"],
                "test_mean": item["test_accuracy"]["mean"],
                "test_sample_std": item["test_accuracy"]["sample_std"],
                "sparsity_mean": item["activation_sparsity"]["mean"],
                "sparsity_sample_std": item["activation_sparsity"]["sample_std"],
                "synops_mean": item["theoretical_synops"]["mean"],
                "synops_sample_std": item["theoretical_synops"]["sample_std"],
            })

    lines = [
        "# Fully-TTFS Pointwise Weight Parameterization Ablation", "",
        "Hard ReLU is a post-hoc constrained adaptation; Softplus is trained from scratch.", "",
        "Values are mean ± sample standard deviation over seeds 42, 6543, and 7777.", "",
        "| Condition | Validation accuracy | Test accuracy | Activation sparsity | SynOps/sample |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['label']} | "
            f"{item['validation_accuracy']['mean']:.2f} ± {item['validation_accuracy']['sample_std']:.2f}% | "
            f"{item['test_accuracy']['mean']:.2f} ± {item['test_accuracy']['sample_std']:.2f}% | "
            f"{item['activation_sparsity']['mean']:.2f} ± {item['activation_sparsity']['sample_std']:.2f}% | "
            f"{item['theoretical_synops']['mean']:,.0f} ± {item['theoretical_synops']['sample_std']:,.0f} |"
        )
    output = output_dir / "POINTWISE_PARAMETERIZATION_ABLATION.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
