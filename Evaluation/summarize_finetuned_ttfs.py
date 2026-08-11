#!/usr/bin/env python3
"""Aggregate fully-TTFS activation sparsity reports across training seeds."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


LAYER_PATTERN = re.compile(
    r"^(downsample_layers\.\d+\.\d+|"
    r"stages\.\d+\.\d+\.(?:dwconv|pw[12]_ttfs))"
    r"\s+(dwconv|pw1|pw2|downsample).*?([0-9]+\.[0-9]+)%\s*$"
)


def _required_percentage(text, label, path):
    match = re.search(rf"{re.escape(label)}:\s+([0-9]+\.[0-9]+)%", text)
    if match is None:
        raise ValueError(f"Missing {label} in {path}")
    return float(match.group(1))


def parse_sparsity_report(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    depthwise = re.search(
        r"Detected depthwise convolution mode:\s+(\w+)", text, re.IGNORECASE
    )
    downsample = re.search(
        r"Detected downsampling convolution mode:\s+(\w+)", text, re.IGNORECASE
    )
    if depthwise is None or depthwise.group(1).lower() != "ttfs":
        raise ValueError(f"Report depthwise convolution mode is not TTFS: {path}")
    if downsample is None or downsample.group(1).lower() != "ttfs":
        raise ValueError(f"Report does not use TTFS downsampling convolution: {path}")

    layers = {}
    for line in text.splitlines():
        match = LAYER_PATTERN.match(line.strip())
        if match:
            name, operation_type, sparsity = match.groups()
            if name in layers:
                raise ValueError(f"Duplicate TTFS layer {name} in {path}")
            layers[name] = {
                "type": operation_type,
                "sparsity": float(sparsity),
            }
    if len(layers) != 39:
        raise ValueError(
            f"Expected exactly 39 TTFS points in {path}, found {len(layers)}"
        )
    measured = re.search(r"Measured TTFS points:\s+(\d+)", text)
    expected = re.search(r"Expected TTFS points:\s+(\d+)", text)
    if measured is None or expected is None or measured.group(1) != "39" or expected.group(1) != "39":
        raise ValueError(f"Report TTFS point counters are not both 39: {path}")
    return {
        "layers": layers,
        "accuracy": _required_percentage(text, "Classification accuracy", path),
        "global_sparsity": _required_percentage(text, "Activation sparsity", path),
    }


def aggregate_dataset(report_paths):
    reports = [parse_sparsity_report(path) for path in report_paths]
    if len(reports) < 2:
        raise ValueError("At least two seed reports are required for sample std")
    layer_names = list(reports[0]["layers"])
    for report in reports[1:]:
        if list(report["layers"]) != layer_names:
            raise ValueError("Seed reports contain different TTFS layer sets or ordering")

    layers = {}
    for name in layer_names:
        values = [report["layers"][name]["sparsity"] for report in reports]
        layers[name] = {
            "type": reports[0]["layers"][name]["type"],
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    accuracies = [report["accuracy"] for report in reports]
    global_sparsities = [report["global_sparsity"] for report in reports]
    return {
        "layers": layers,
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_sample_std": statistics.stdev(accuracies),
        "global_sparsity_mean": statistics.mean(global_sparsities),
        "global_sparsity_sample_std": statistics.stdev(global_sparsities),
    }


def format_markdown(summaries, seeds_by_dataset):
    lines = [
        "# Dense-pretrained fully-TTFS fine-tuning summary",
        "",
        "Values are arithmetic mean +/- sample standard deviation across three seeds.",
        "",
        "## Overall results",
        "",
        "| Dataset | Seeds | Accuracy mean +/- SD (%) | Global sparsity mean +/- SD (%) |",
        "|---|---|---:|---:|",
    ]
    for dataset, summary in summaries.items():
        seeds = ", ".join(str(seed) for seed in seeds_by_dataset[dataset])
        lines.append(
            f"| {dataset} | {seeds} | "
            f"{summary['accuracy_mean']:.2f} +/- {summary['accuracy_sample_std']:.2f} | "
            f"{summary['global_sparsity_mean']:.2f} +/- "
            f"{summary['global_sparsity_sample_std']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Layer-wise activation sparsity",
            "",
            "| Layer | Type | "
            + " | ".join(f"{dataset} mean +/- SD (%)" for dataset in summaries)
            + " |",
            "|---|---|" + "---:|" * len(summaries),
        ]
    )
    reference = next(iter(summaries.values()))
    for layer_name in reference["layers"]:
        row = [layer_name, reference["layers"][layer_name]["type"]]
        for summary in summaries.values():
            layer = summary["layers"][layer_name]
            row.append(f"{layer['mean']:.2f} +/- {layer['sample_std']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate fully-TTFS sparsity across fine-tuned seeds"
    )
    parser.add_argument("--root", default="fine_tune_results")
    parser.add_argument(
        "--output",
        default="fine_tune_results/FULLY_TTFS_FINE_TUNING_SUMMARY.md",
    )
    args = parser.parse_args()
    root = Path(args.root)
    seeds_by_dataset = {
        "CIFAR-10": (42, 6543, 7777),
        "CIFAR-100": (42, 6543, 7777),
        "Tiny ImageNet": (42, 2344, 5435),
    }
    directories = {
        "CIFAR-10": "cifar10",
        "CIFAR-100": "cifar100",
        "Tiny ImageNet": "tinyimagenet",
    }
    summaries = {}
    for display_name, seeds in seeds_by_dataset.items():
        reports = [
            root
            / directories[display_name]
            / "fully_ttfs"
            / f"seed_{seed}"
            / "activation_sparsity.md"
            for seed in seeds
        ]
        summaries[display_name] = aggregate_dataset(reports)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        format_markdown(summaries, seeds_by_dataset),
        encoding="utf-8",
    )
    print(f"Saved fine-tuning summary: {output}")


if __name__ == "__main__":
    main()
