#!/usr/bin/env python3
"""Summarize the CIFAR-100 non-negative pointwise-weight ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    from Evaluation.summarize_finetuned_ttfs import parse_sparsity_report
except ModuleNotFoundError:
    from summarize_finetuned_ttfs import parse_sparsity_report


SEEDS = (42, 6543, 7777)


def summarize_condition(report_paths, training_summary_paths=None):
    reports = [parse_sparsity_report(path) for path in report_paths]
    if len(reports) != 3:
        raise ValueError("Exactly three seed reports are required")
    layer_names = list(reports[0]["layers"])
    if any(list(report["layers"]) != layer_names for report in reports[1:]):
        raise ValueError("Condition reports have different layer sets")

    result = {
        "test_accuracy_values": [report["accuracy"] for report in reports],
        "global_sparsity_values": [
            report["global_sparsity"] for report in reports
        ],
        "layers": {},
    }
    for name in layer_names:
        values = [report["layers"][name]["sparsity"] for report in reports]
        result["layers"][name] = {
            "type": reports[0]["layers"][name]["type"],
            "values": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    for field in ("test_accuracy", "global_sparsity"):
        values = result[f"{field}_values"]
        result[f"{field}_mean"] = statistics.mean(values)
        result[f"{field}_sample_std"] = statistics.stdev(values)

    if training_summary_paths is not None:
        summaries = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in training_summary_paths
        ]
        values = [float(item["best_validation_accuracy"]) for item in summaries]
        result["best_validation_accuracy_values"] = values
        result["best_validation_accuracy_mean"] = statistics.mean(values)
        result["best_validation_accuracy_sample_std"] = statistics.stdev(values)
    return result


def architectural_layer_order(layer_names):
    names = set(layer_names)
    ordered = []
    for stage, depth in enumerate((2, 2, 6, 2)):
        for block in range(depth):
            for suffix in ("dwconv", "pw1_ttfs", "pw2_ttfs"):
                name = f"stages.{stage}.{block}.{suffix}"
                if name in names:
                    ordered.append(name)
        if stage < 3:
            name = f"downsample_layers.{stage + 1}.0"
            if name in names:
                ordered.append(name)
    if set(ordered) != names:
        raise ValueError("Cannot determine architectural layer order")
    return ordered


def format_markdown(conditions):
    lines = [
        "# CIFAR-100 Non-Negative Pointwise Weight Ablation",
        "",
        "PW1/PW2 use non-negative effective weights via ReLU. DWConv, "
        "downsampling, stem, and classifier weights remain unconstrained.",
        "",
        "Values are mean ± sample standard deviation across seeds "
        "42, 6543, and 7777.",
        "",
        "## Overall comparison",
        "",
        "| Condition | Best validation accuracy | Test accuracy | "
        "Weighted global sparsity |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in conditions.items():
        validation = "—"
        if "best_validation_accuracy_mean" in summary:
            validation = (
                f"{summary['best_validation_accuracy_mean']:.2f} ± "
                f"{summary['best_validation_accuracy_sample_std']:.2f}%"
            )
        lines.append(
            f"| {name} | {validation} | "
            f"{summary['test_accuracy_mean']:.2f} ± "
            f"{summary['test_accuracy_sample_std']:.2f}% | "
            f"{summary['global_sparsity_mean']:.2f} ± "
            f"{summary['global_sparsity_sample_std']:.2f}% |"
        )

    lines.extend(["", "## Per-seed results", ""])
    for name, summary in conditions.items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Seed | Best validation accuracy | Test accuracy | "
                "Weighted global sparsity |",
                "|---:|---:|---:|---:|",
            ]
        )
        validation_values = summary.get("best_validation_accuracy_values")
        for index, seed in enumerate(SEEDS):
            validation = (
                f"{validation_values[index]:.2f}%"
                if validation_values is not None else "—"
            )
            lines.append(
                f"| {seed} | {validation} | "
                f"{summary['test_accuracy_values'][index]:.2f}% | "
                f"{summary['global_sparsity_values'][index]:.2f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Layer-wise activation sparsity",
            "",
            "| Layer | " + " | ".join(conditions) + " |",
            "|---|" + "---:|" * len(conditions),
        ]
    )
    reference = next(iter(conditions.values()))
    for layer_name in architectural_layer_order(reference["layers"]):
        cells = []
        for summary in conditions.values():
            layer = summary["layers"][layer_name]
            cells.append(f"{layer['mean']:.2f} ± {layer['sample_std']:.2f}%")
        lines.append(f"| {layer_name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="fine_tune_results_v3/cifar100")
    parser.add_argument(
        "--output",
        default=(
            "fine_tune_results_v3/cifar100/ablation_nonnegative_pointwise/"
            "NONNEGATIVE_POINTWISE_ABLATION.md"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root)
    baseline_root = root / "fully_ttfs"
    constrained_root = root / "ablation_nonnegative_pointwise"

    def paths(base, relative):
        return [base / f"seed_{seed}" / relative for seed in SEEDS]

    conditions = {
        "Unconstrained fully TTFS": summarize_condition(
            paths(baseline_root, "activation_sparsity.md"),
            paths(baseline_root, "training_summary.json"),
        ),
        "Constraint before adaptation": summarize_condition(
            paths(
                constrained_root,
                "initial_constraint/activation_sparsity.md",
            )
        ),
        "Constraint after adaptation": summarize_condition(
            paths(constrained_root, "activation_sparsity.md"),
            paths(constrained_root, "training_summary.json"),
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown(conditions), encoding="utf-8")
    print(f"Saved ablation summary: {output.resolve()}")


if __name__ == "__main__":
    main()
