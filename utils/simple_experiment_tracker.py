"""Small dependency-free JSON/Markdown/CSV experiment tracker."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


REGISTRY_FIELDS = [
    "experiment_name",
    "date_time",
    "output_directory",
    "notes",
    "seed",
    "dataset_name",
    "number_of_classes",
    "input_resolution",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "preprocessing",
    "augmentation",
    "dims",
    "depths",
    "parameter_count",
    "stem_kernel",
    "stem_stride",
    "stem_padding",
    "residual_operator",
    "pw1_mode",
    "pw2_mode",
    "spike_dropout",
    "delay_enabled",
    "stage_delays",
    "t_min",
    "t_max",
    "epochs",
    "batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "label_smoothing",
    "head_dropout",
    "mixup_alpha",
    "early_stopping_patience",
    "best_epoch",
    "best_validation_accuracy",
    "final_train_accuracy",
    "final_validation_accuracy",
    "test_accuracy",
    "test_loss",
    "training_time_seconds",
    "checkpoint_path",
    "activation_sparsity",
    "dense_macs_per_sample",
    "theoretical_synops_per_sample",
    "status",
    "updated_at",
]


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _display(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    title = _display(report.get("experiment", {}).get("experiment_name"))
    lines = [f"# Experiment Report: {title}", ""]
    section_titles = {
        "experiment": "Experiment Information",
        "dataset": "Dataset",
        "architecture": "Architecture",
        "training": "Training",
        "results": "Results",
        "optional_evaluation": "Optional Evaluation Results",
    }
    for section_key, heading in section_titles.items():
        lines.extend([f"## {heading}", "", "| Field | Value |", "|---|---|"])
        section = report.get(section_key, {})
        for key, value in section.items():
            safe_value = _display(value).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {key.replace('_', ' ').title()} | {safe_value} |")
        lines.append("")
    return "\n".join(lines)


def _registry_row(report: dict[str, Any]) -> dict[str, str]:
    flat: dict[str, Any] = {}
    for section in report.values():
        if isinstance(section, dict):
            flat.update(section)
    return {field: _display(flat.get(field)) for field in REGISTRY_FIELDS}


class SimpleExperimentTracker:
    """Persist one report per output directory and upsert one global CSV row."""

    def __init__(self, output_directory: Path | str, registry_path: Path | str):
        self.output_directory = Path(output_directory).resolve()
        self.registry_path = Path(registry_path).resolve()
        self.json_path = self.output_directory / "experiment_report.json"
        self.markdown_path = self.output_directory / "experiment_report.md"

    def load_existing_report(self) -> dict[str, Any] | None:
        if not self.json_path.is_file():
            return None
        try:
            loaded = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def save(self, report: dict[str, Any]) -> None:
        report.setdefault("experiment", {})["output_directory"] = str(
            self.output_directory
        )
        report["experiment"]["updated_at"] = local_timestamp()
        _atomic_text_write(
            self.json_path,
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
        _atomic_text_write(self.markdown_path, _markdown(report))
        self._upsert_registry(_registry_row(report))

    def _upsert_registry(self, new_row: dict[str, str]) -> None:
        rows: list[dict[str, str]] = []
        existing_fields: list[str] = []
        if self.registry_path.is_file():
            with self.registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_fields = list(reader.fieldnames or [])
                rows = list(reader)

        identity = str(self.output_directory).casefold()
        updated = False
        for row in rows:
            existing_identity = str(row.get("output_directory", "")).casefold()
            if existing_identity == identity and not updated:
                row.update(new_row)
                updated = True
        if not updated:
            rows.append(new_row)

        fields = list(REGISTRY_FIELDS)
        fields.extend(field for field in existing_fields if field not in fields)
        temporary = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: row.get(field, "unknown") or "unknown"
                        for field in fields
                    }
                )
        os.replace(temporary, self.registry_path)
