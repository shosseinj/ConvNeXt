import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SEEDS = (42, 6543, 7777)
DATASETS = ("cifar10", "cifar100")
REPORT_PATTERN = re.compile(
    r"^(stages\.(\d+)\.(\d+)\.(pw[12])_ttfs)"
    r"\s+pw[12]\s+[\d,]+\s+[\d,]+\s+([\d.]+)%$",
    re.MULTILINE,
)


def report_path(dataset, seed):
    return (
        PROJECT_ROOT
        / "results"
        / dataset
        / "downsample_dense_dwconv_dense"
        / f"seed_{seed}"
        / "activation_sparsity.md"
    )


def read_layer_sparsity(dataset, seed):
    path = report_path(dataset, seed)
    text = path.read_text(encoding="utf-8")
    values = {}
    for _, stage, block, pointwise, sparsity in REPORT_PATTERN.findall(text):
        values[(int(stage), int(block), pointwise)] = float(sparsity)
    if len(values) != 24:
        raise ValueError(f"Expected 24 pointwise layers in {path}, found {len(values)}")
    return values


def aggregate():
    per_dataset = {
        dataset: {seed: read_layer_sparsity(dataset, seed) for seed in SEEDS}
        for dataset in DATASETS
    }
    blocks = sorted(
        {
            (stage, block)
            for values in per_dataset[DATASETS[0]].values()
            for stage, block, _ in values
        }
    )
    rows = []
    for dataset in DATASETS:
        for stage, block in blocks:
            row = {
                "dataset": dataset,
                "stage": stage + 1,
                "block": block + 1,
                "label": f"S{stage + 1}-B{block + 1}",
            }
            for pointwise in ("pw1", "pw2"):
                values = [
                    per_dataset[dataset][seed][(stage, block, pointwise)]
                    for seed in SEEDS
                ]
                row[f"{pointwise}_mean"] = statistics.mean(values)
                row[f"{pointwise}_std"] = statistics.stdev(values)
            rows.append(row)
    return blocks, rows


def write_csv(rows):
    path = OUTPUT_DIR / "layer_sparsity_mean_std.csv"
    fieldnames = (
        "dataset",
        "stage",
        "block",
        "label",
        "pw1_mean",
        "pw1_std",
        "pw2_mean",
        "pw2_std",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.4f}" if isinstance(row[key], float) else row[key]
                    for key in fieldnames
                }
            )
    return path


def draw_figure(blocks, rows):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    width_inches = 190 / 25.4
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width_inches, 5.45),
        sharex=True,
        constrained_layout=True,
    )
    colors = {"pw1": "#0072B2", "pw2": "#D55E00"}
    markers = {"pw1": "o", "pw2": "s"}
    x = np.arange(len(blocks))
    labels = [f"S{stage + 1}-B{block + 1}" for stage, block in blocks]
    boundaries = (1.5, 3.5, 9.5)
    stage_ranges = ((-0.5, 1.5), (1.5, 3.5), (3.5, 9.5), (9.5, 11.5))

    for panel_index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for stage_index, (start, end) in enumerate(stage_ranges):
            if stage_index % 2:
                axis.axvspan(start, end, color="#F2F2F2", zorder=0)
        for boundary in boundaries:
            axis.axvline(boundary, color="#777777", linewidth=0.65, linestyle="--", zorder=1)
        for pointwise, display_name in (("pw1", "PW1"), ("pw2", "PW2")):
            means = np.array([row[f"{pointwise}_mean"] for row in dataset_rows])
            stds = np.array([row[f"{pointwise}_std"] for row in dataset_rows])
            axis.errorbar(
                x,
                means,
                yerr=stds,
                label=display_name,
                color=colors[pointwise],
                marker=markers[pointwise],
                markersize=4.2,
                linewidth=1.25,
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
            )
        axis.set_ylim(0, 105)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_ylabel("Activation sparsity (%)")
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            0.94,
            f"({chr(ord('a') + panel_index)}) {dataset.upper().replace('CIFAR', 'CIFAR-')}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        for stage_index, (start, end) in enumerate(stage_ranges, start=1):
            axis.text(
                (start + end) / 2,
                102,
                f"Stage {stage_index}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#444444",
            )

    axes[0].legend(loc="lower right", frameon=True, framealpha=1, edgecolor="#777777")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[-1].set_xlabel("ConvNeXt block")

    base = OUTPUT_DIR / "layer_sparsity_cifar10_cifar100"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return base


def write_markdown(rows):
    path = OUTPUT_DIR / "LAYER_SPARSITY_ANALYSIS.md"
    lines = [
        "# Layer-wise activation sparsity",
        "",
        "## Figure",
        "",
        "![Layer-wise activation sparsity](layer_sparsity_cifar10_cifar100.png)",
        "",
        "Vector formats: [PDF](layer_sparsity_cifar10_cifar100.pdf) | "
        "[SVG](layer_sparsity_cifar10_cifar100.svg)",
        "",
        "## Method",
        "",
        "Layer-wise activation sparsity was aggregated over seeds 42, 6543, and 7777. "
        "Values are the arithmetic mean +/- sample standard deviation (n - 1) across "
        "three independently trained models. The evaluated dense-depthwise, "
        "dense-downsampling configuration contains 24 TTFS measurement points: PW1 "
        "and PW2 outputs in each of 12 ConvNeXt blocks.",
        "",
        "## Elsevier-ready caption",
        "",
        "**Fig. 1.** Layer-wise TTFS activation sparsity for (a) CIFAR-10 and "
        "(b) CIFAR-100. Markers denote the mean across three random seeds "
        "(42, 6543, and 7777), and error bars denote sample standard deviation. "
        "PW1 and PW2 indicate the first and second pointwise transformations within "
        "each ConvNeXt block. Dashed vertical lines separate the four network stages.",
        "",
        "## Layer statistics",
        "",
        "| Dataset | Layer | PW1 sparsity mean +/- SD (%) | PW2 sparsity mean +/- SD (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset'].upper().replace('CIFAR', 'CIFAR-')} | {row['label']} | "
            f"{row['pw1_mean']:.2f} +/- {row['pw1_std']:.2f} | "
            f"{row['pw2_mean']:.2f} +/- {row['pw2_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PW2 is consistently sparser than PW1 through most blocks, especially in "
            "the middle stages. The earliest PW1 layers remain comparatively dense, "
            "while later stages exhibit substantially higher sparsity. Error bars expose "
            "seed sensitivity and should be retained in the published figure.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "C:\\Users\\jafari.h\\Desktop\\ai_project\\.venv\\Scripts\\python.exe "
            ".\\Evaluation\\layer_sparsity_figure\\generate_layer_sparsity_figure.py",
            "```",
            "",
            "Underlying values: [CSV](layer_sparsity_mean_std.csv)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks, rows = aggregate()
    csv_path = write_csv(rows)
    figure_base = draw_figure(blocks, rows)
    markdown_path = write_markdown(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_base.with_suffix('.pdf')}")
    print(f"Wrote {figure_base.with_suffix('.svg')}")
    print(f"Wrote {figure_base.with_suffix('.png')}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
