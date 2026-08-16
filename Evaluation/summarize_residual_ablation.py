#!/usr/bin/env python3
"""Create the complete CIFAR-100 residual-fusion ablation report."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch


SEEDS = (42, 6543, 7777)
OPERATORS = ("min", "mean", "learnable_gate")
DISPLAY_NAMES = {
    "min": "Minimum",
    "mean": "Normalized Sum (Mean)",
    "learnable_gate": "Learnable Gate",
}
LAYER_PATTERN = re.compile(
    r"^(\S+)\s+(dwconv|pw1|pw2|downsample)\s+"
    r"([\d,]+)\s+([\d,]+)\s+([\d.]+)%\s+([\d,]+)$"
)


def run_directory(root, operator, seed):
    if operator == "min":
        return root / "cifar100" / "fully_ttfs" / f"seed_{seed}"
    return (
        root
        / "cifar100"
        / "ablation_residual"
        / operator
        / f"seed_{seed}"
    )


def _required_match(pattern, text, path, label):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Missing {label} in {path}")
    return match.group(1)


def parse_activation_report(path, expected_operator):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    operator = _required_match(
        r"Detected residual operator:\s+(\w+)", text, path, "residual operator"
    ).lower()
    if operator != expected_operator:
        raise ValueError(
            f"Residual operator mismatch in {path}: "
            f"expected={expected_operator}, measured={operator}"
        )
    measured = int(_required_match(
        r"Measured TTFS points:\s+(\d+)", text, path, "measured points"
    ))
    expected = int(_required_match(
        r"Expected TTFS points:\s+(\d+)", text, path, "expected points"
    ))
    if measured != 39 or expected != 39:
        raise ValueError(
            f"Expected measured=39 and expected=39 in {path}, "
            f"got measured={measured}, expected={expected}"
        )
    sparsity = float(_required_match(
        r"Activation sparsity:\s+([\d.]+)%",
        text,
        path,
        "weighted activation sparsity",
    ))
    accuracy = float(_required_match(
        r"Classification accuracy:\s+([\d.]+)%",
        text,
        path,
        "classification accuracy",
    ))
    synops = float(_required_match(
        r"Theoretical SynOps:\s+([\d,]+)\s+per sample",
        text,
        path,
        "theoretical SynOps",
    ).replace(",", ""))
    layers = {}
    for line in text.splitlines():
        match = LAYER_PATTERN.match(line.strip())
        if match:
            name, layer_type, _, _, layer_sparsity, layer_synops = match.groups()
            if name in layers:
                raise ValueError(f"Duplicate TTFS layer {name} in {path}")
            layers[name] = {
                "type": layer_type,
                "sparsity": float(layer_sparsity),
                "synops": float(layer_synops.replace(",", "")),
            }
    if len(layers) != 39:
        raise ValueError(f"Expected 39 layer rows in {path}, found {len(layers)}")
    if abs(sum(layer["synops"] for layer in layers.values()) - synops) > 40:
        raise ValueError(f"Layerwise SynOps do not reconcile with global SynOps in {path}")
    return {
        "accuracy": accuracy,
        "sparsity": sparsity,
        "synops": synops,
        "layers": layers,
    }


def extract_gate_statistics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema") or checkpoint.get("model") or checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"No model state dictionary in {checkpoint_path}")
    rows = []
    for name, raw_gate in sorted(state.items()):
        if name.endswith(".raw_residual_gate"):
            gate = torch.sigmoid(raw_gate.detach().float())
            rows.append({
                "layer": name.removesuffix(".raw_residual_gate"),
                "mean": float(gate.mean().item()),
                "std": float(gate.std(unbiased=False).item()),
            })
    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 learnable residual gates in {checkpoint_path}, "
            f"found {len(rows)}"
        )
    return rows


def load_run(root, operator, seed):
    directory = run_directory(root, operator, seed)
    required = (
        directory / "training_summary.json",
        directory / "activation_sparsity.md",
        directory / "best_checkpoint.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The nine-run residual campaign is incomplete; missing: "
            + ", ".join(missing)
        )
    summary = json.loads(required[0].read_text(encoding="utf-8"))
    evaluation = parse_activation_report(required[1], operator)
    test_metrics = summary.get("test_metrics") or {}
    if "best_validation_accuracy" not in summary or "accuracy" not in test_metrics:
        raise ValueError(f"Incomplete accuracy metrics in {required[0]}")
    recorded_test_accuracy = float(test_metrics["accuracy"])
    if abs(recorded_test_accuracy - evaluation["accuracy"]) > 0.011:
        raise ValueError(
            f"Evaluator accuracy does not match training summary in {directory}: "
            f"evaluated={evaluation['accuracy']}, recorded={recorded_test_accuracy}"
        )
    result = {
        "seed": seed,
        "validation_accuracy": float(summary["best_validation_accuracy"]),
        "test_accuracy": evaluation.pop("accuracy"),
        **evaluation,
    }
    result["gates"] = (
        extract_gate_statistics(required[2])
        if operator == "learnable_gate" else []
    )
    return result


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def formatted(values, decimals=2):
    mean, std = mean_std(values)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_report(root):
    root = Path(root)
    results = {
        operator: [load_run(root, operator, seed) for seed in SEEDS]
        for operator in OPERATORS
    }
    lines = [
        "# CIFAR-100 Fully-TTFS Residual Fusion Ablation",
        "",
        "All values are mean ± sample standard deviation across seeds 42, 6543, and 7777. "
        "Test accuracy is evaluated from the checkpoint selected exclusively by best validation accuracy.",
        "",
        "## Main results",
        "",
        "| Residual fusion | Best validation accuracy | Test accuracy | Weighted activation sparsity | Theoretical SynOps/sample |",
        "|---|---:|---:|---:|---:|",
    ]
    for operator in OPERATORS:
        runs = results[operator]
        lines.append(
            f"| {DISPLAY_NAMES[operator]} | "
            f"{formatted([run['validation_accuracy'] for run in runs])}% | "
            f"{formatted([run['test_accuracy'] for run in runs])}% | "
            f"{formatted([run['sparsity'] for run in runs])}% | "
            f"{formatted([run['synops'] for run in runs], 0)} |"
        )

    lines.extend(["", "## Per-seed results", ""])
    for operator in OPERATORS:
        lines.extend([
            f"### {DISPLAY_NAMES[operator]}",
            "",
            "| Seed | Best validation accuracy | Test accuracy | Weighted sparsity | SynOps/sample |",
            "|---:|---:|---:|---:|---:|",
        ])
        for run in results[operator]:
            lines.append(
                f"| {run['seed']} | {run['validation_accuracy']:.2f}% | "
                f"{run['test_accuracy']:.2f}% | {run['sparsity']:.2f}% | "
                f"{run['synops']:,.0f} |"
            )
        lines.append("")

    lines.extend([
        "## Layerwise supplementary results",
        "",
        "| Layer | Type | "
        + " | ".join(f"{DISPLAY_NAMES[operator]} sparsity" for operator in OPERATORS)
        + " | "
        + " | ".join(f"{DISPLAY_NAMES[operator]} SynOps" for operator in OPERATORS)
        + " |",
        "|---|---|" + "---:|" * 6,
    ])
    layer_names = list(results["min"][0]["layers"])
    for operator in OPERATORS:
        for run in results[operator]:
            if list(run["layers"]) != layer_names:
                raise ValueError("Residual runs contain different TTFS layer sets or ordering")
    for name in layer_names:
        layer_type = results["min"][0]["layers"][name]["type"]
        sparsities = [
            formatted([run["layers"][name]["sparsity"] for run in results[operator]]) + "%"
            for operator in OPERATORS
        ]
        synops = [
            formatted([run["layers"][name]["synops"] for run in results[operator]], 0)
            for operator in OPERATORS
        ]
        lines.append("| " + " | ".join([name, layer_type, *sparsities, *synops]) + " |")

    lines.extend([
        "",
        "## Learnable-gate supplementary results",
        "",
        "Gate values are `sigmoid(raw_gate)`; 0.5 gives normalized-sum behavior.",
        "",
        "| Layer | Seed 42 mean ± channel SD | Seed 6543 mean ± channel SD | Seed 7777 mean ± channel SD |",
        "|---|---:|---:|---:|",
    ])
    gate_runs = results["learnable_gate"]
    for index in range(12):
        layer = gate_runs[0]["gates"][index]["layer"]
        values = [
            f"{run['gates'][index]['mean']:.4f} ± {run['gates'][index]['std']:.4f}"
            for run in gate_runs
        ]
        lines.append("| " + " | ".join([layer, *values]) + " |")

    lines.extend([
        "",
        "## Response to Reviewer 1, Comment 4",
        "",
        "We added a controlled residual-fusion ablation on CIFAR-100 using the same "
        "Fully-TTFS architecture, matched dense initialization, data splits, training "
        "settings, and three random seeds. Normalized sum is the arithmetic mean of the "
        "identity and residual-branch spike times. The learnable gate is initialized at "
        "0.5 and therefore begins with exactly the same behavior as normalized sum. "
        "Minimum fusion selects the earliest spike time and consequently preserves the "
        "native TTFS interpretation that an earlier event represents stronger evidence. "
        "The resulting accuracy, activation sparsity, and theoretical event-driven SynOps "
        "are reported above as mean ± sample standard deviation.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="fine_tune_results_v3")
    parser.add_argument(
        "--output",
        default=(
            "fine_tune_results_v3/cifar100/ablation_residual/"
            "RESIDUAL_FUSION_REVIEWER1_COMMENT4.md"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(Path(args.root)), encoding="utf-8")
    print(f"Saved residual-fusion report: {output.resolve()}")


if __name__ == "__main__":
    main()
