#!/usr/bin/env python3
"""Summarize best validation accuracy across selected training seeds."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


BASE_DIRECTORY = Path("results/cifar10/clean_finetune_from_94_36")
DATASET_NAME = "CIFAR-10"
SEEDS = (42, 6543, 7777)


def _accuracy(value, source):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"best_validation_accuracy in {source} must be a number"
        )
    accuracy = float(value)
    if not math.isfinite(accuracy):
        raise ValueError(
            f"best_validation_accuracy in {source} must be finite"
        )
    return accuracy


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}") from error


def extract_best_validation_accuracy(seed_directory):
    seed_directory = Path(seed_directory)
    if not seed_directory.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_directory}")

    training_summary = seed_directory / "training_summary.json"
    if training_summary.is_file():
        data = _read_json(training_summary)
        if "best_validation_accuracy" in data:
            return _accuracy(data["best_validation_accuracy"], training_summary)

    experiment_report = seed_directory / "experiment_report.json"
    if experiment_report.is_file():
        data = _read_json(experiment_report)
        results = data.get("results", {}) if isinstance(data, dict) else {}
        if isinstance(results, dict) and "best_validation_accuracy" in results:
            return _accuracy(
                results["best_validation_accuracy"], experiment_report
            )

    train_log = seed_directory / "train_log.jsonl"
    if train_log.is_file():
        accuracies = []
        try:
            lines = train_log.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"Could not read {train_log}: {error}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {train_log} at line {line_number}: {error}"
                ) from error
            if isinstance(row, dict) and "best_validation_accuracy" in row:
                accuracies.append(
                    _accuracy(row["best_validation_accuracy"], train_log)
                )
        if accuracies:
            return max(accuracies)

    for tta_report in sorted(seed_directory.rglob("tta_evaluation.json")):
        data = _read_json(tta_report)
        checkpoints = data.get("checkpoints", []) if isinstance(data, dict) else []
        accuracies = [
            _accuracy(checkpoint["best_validation_accuracy"], tta_report)
            for checkpoint in checkpoints
            if isinstance(checkpoint, dict)
            and "best_validation_accuracy" in checkpoint
        ]
        if accuracies:
            return max(accuracies)

    raise ValueError(
        "No best_validation_accuracy found in supported result files under "
        f"{seed_directory}"
    )


def load_seed_accuracies(base_directory, seeds):
    base_directory = Path(base_directory)
    return {
        seed: extract_best_validation_accuracy(base_directory / f"seed_{seed}")
        for seed in seeds
    }


def calculate_summary(accuracies):
    values = list(accuracies.values())
    if len(values) < 2:
        raise ValueError("At least two seed accuracies are required for sample std")
    return statistics.mean(values), statistics.stdev(values)


def format_summary(dataset_name, accuracies, mean_accuracy, sample_std):
    lines = [
        f"Dataset: {dataset_name}",
        *(f"Seed {seed}: {accuracy:.2f}" for seed, accuracy in accuracies.items()),
        f"Mean: {mean_accuracy:.2f}",
        f"Sample standard deviation: {sample_std:.2f}",
        f"Final: {mean_accuracy:.2f} ± {sample_std:.2f}",
    ]
    return "\n".join(lines)


def main():
    accuracies = load_seed_accuracies(BASE_DIRECTORY, SEEDS)
    mean_accuracy, sample_std = calculate_summary(accuracies)
    print(format_summary(DATASET_NAME, accuracies, mean_accuracy, sample_std))


if __name__ == "__main__":
    main()
