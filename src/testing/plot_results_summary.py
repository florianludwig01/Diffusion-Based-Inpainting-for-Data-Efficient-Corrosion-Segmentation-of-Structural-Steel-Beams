"""Create cross-model plots from the multi-seed evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "output" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output" / "results_summary"
MODEL_FAMILIES = {"dinov3": "DINOv3", "yolo": "YOLO"}
METRICS = {"dataset_dice": "Dice", "dataset_iou": "IoU"}
PR_COLUMNS = ["dataset_precision", "dataset_recall"]
DATASET_ORDER = ["R150", "SFx1", "SSDx1", "R1500", "HFx1", "HSDx1", "R1650"]


def ordered_datasets(values) -> list[str]:
    """Use the experimental order while retaining any future dataset variants."""
    available = set(values)
    return [name for name in DATASET_ORDER if name in available] + sorted(
        available.difference(DATASET_ORDER)
    )


def collect_best_seed_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each seed's row with the highest dataset IoU and summarize it."""
    seed_frames = []
    for family_dir, family_label in MODEL_FAMILIES.items():
        for result_file in sorted((INPUT_ROOT / family_dir).glob("*/all_seed_results.csv")):
            dataset = result_file.parent.name
            results = pd.read_csv(result_file)
            required = {"seed", "threshold", *METRICS}
            missing = required.difference(results.columns)
            if missing:
                raise ValueError(f"Missing columns in {result_file}: {sorted(missing)}")

            best_rows = results.loc[
                results.groupby("seed")["dataset_iou"].idxmax()
            ].copy()
            best_rows.insert(0, "model_family", family_label)
            best_rows.insert(1, "dataset", dataset)
            seed_frames.append(
                best_rows[["model_family", "dataset", "seed", "threshold", *METRICS]]
            )

    if not seed_frames:
        raise FileNotFoundError(f"No all_seed_results.csv files found under {INPUT_ROOT}")

    per_seed = pd.concat(seed_frames, ignore_index=True)
    summary_rows = []
    for (family, dataset), group in per_seed.groupby(
        ["model_family", "dataset"], sort=True
    ):
        row = {
            "model_family": family,
            "dataset": dataset,
            "n_seeds": len(group),
        }
        for column, label in METRICS.items():
            values = group[column]
            best_index = values.idxmax()
            prefix = label.lower()
            row.update(
                {
                    f"{prefix}_mean": values.mean(),
                    f"{prefix}_std": values.std(ddof=1),
                    f"{prefix}_min": values.min(),
                    f"{prefix}_max": values.max(),
                    f"{prefix}_best_seed": int(per_seed.loc[best_index, "seed"]),
                    f"{prefix}_best_threshold": per_seed.loc[best_index, "threshold"],
                }
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["model_family", "dataset"], ignore_index=True
    )
    return per_seed, summary


def style_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=14, weight="bold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_family(summary: pd.DataFrame, family: str, output_file: Path) -> None:
    data = summary[summary["model_family"] == family].set_index("dataset")
    data = data.loc[ordered_datasets(data.index)].reset_index()
    x = np.arange(len(data))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(10, 1.45 * len(data)), 6))
    for offset, metric, color in [
        (-width / 2, "dice", "#2878B5"),
        (width / 2, "iou", "#E07A1F"),
    ]:
        bars = axis.bar(
            x + offset,
            data[f"{metric}_mean"],
            width,
            yerr=data[f"{metric}_std"],
            capsize=4,
            label=metric.upper() if metric == "iou" else metric.title(),
            color=color,
            edgecolor="white",
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    axis.set_xticks(x, data["dataset"])
    axis.set_xlabel("Dataset variant")
    axis.legend(frameon=False, ncol=2)
    style_axis(axis, f"{family}: Mean Best-Threshold Scores")
    fig.text(0.5, 0.01, "Best threshold selected per seed; bars show mean ± SD (n=10).",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, output_file: Path) -> None:
    datasets = ordered_datasets(summary["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.2
    fig, axis = plt.subplots(figsize=(max(12, 1.75 * len(datasets)), 6.5))
    series = [
        ("DINOv3", "dice", -1.5 * width, "#2878B5", "DINOv3 Dice"),
        ("DINOv3", "iou", -0.5 * width, "#78B7E5", "DINOv3 IoU"),
        ("YOLO", "dice", 0.5 * width, "#D95F59", "YOLO Dice"),
        ("YOLO", "iou", 1.5 * width, "#F2A65A", "YOLO IoU"),
    ]
    for family, metric, offset, color, label in series:
        indexed = summary[summary["model_family"] == family].set_index("dataset")
        values = indexed.loc[datasets]
        axis.bar(
            x + offset,
            values[f"{metric}_mean"],
            width,
            yerr=values[f"{metric}_std"],
            capsize=3,
            color=color,
            edgecolor="white",
            label=label,
        )

    axis.set_xticks(x, datasets)
    axis.set_xlabel("Dataset variant")
    axis.legend(frameon=False, ncol=2)
    style_axis(axis, "DINOv3 vs. YOLO: Mean Best-Threshold Scores")
    fig.text(0.5, 0.01, "Best threshold selected per seed; bars show mean ± SD (n=10).",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_precision_recall_statistics() -> pd.DataFrame:
    """Aggregate precision and recall at every threshold across seeds."""
    frames = []
    for family_dir, family_label in MODEL_FAMILIES.items():
        for result_file in sorted((INPUT_ROOT / family_dir).glob("*/all_seed_results.csv")):
            results = pd.read_csv(result_file)
            required = {"seed", "threshold", *PR_COLUMNS}
            missing = required.difference(results.columns)
            if missing:
                raise ValueError(f"Missing columns in {result_file}: {sorted(missing)}")

            statistics = (
                results.groupby("threshold", sort=True)[PR_COLUMNS]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            statistics.columns = [
                "threshold",
                "precision_mean",
                "precision_std",
                "precision_n",
                "recall_mean",
                "recall_std",
                "recall_n",
            ]
            statistics.insert(0, "dataset", result_file.parent.name)
            statistics.insert(0, "model_family", family_label)
            frames.append(statistics)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "model_family", "threshold"], ignore_index=True
    )


def plot_precision_recall_comparison(statistics: pd.DataFrame, output_file: Path) -> None:
    """Plot mean PR curves with seed-to-seed standard deviations."""
    datasets = ordered_datasets(statistics["dataset"].unique())
    columns = 3
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(15, 4.3 * rows), sharex=True, sharey=True,
        squeeze=False,
    )
    colors = {"DINOv3": "#2878B5", "YOLO": "#D95F59"}

    for axis, dataset in zip(axes.flat, datasets):
        for family in MODEL_FAMILIES.values():
            curve = statistics[
                (statistics["dataset"] == dataset)
                & (statistics["model_family"] == family)
            ].sort_values("recall_mean")
            recall = curve["recall_mean"].to_numpy()
            precision = curve["precision_mean"].to_numpy()
            recall_std = curve["recall_std"].fillna(0).to_numpy()
            precision_std = curve["precision_std"].fillna(0).to_numpy()
            color = colors[family]

            axis.plot(recall, precision, color=color, linewidth=2.2, label=family)
            axis.fill_between(
                recall,
                np.clip(precision - precision_std, 0, 1),
                np.clip(precision + precision_std, 0, 1),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            # Sparse horizontal and vertical error bars show both Recall and
            # Precision standard deviations without obscuring the curves.
            positions = np.arange(0, len(curve), 3)
            axis.errorbar(
                recall[positions],
                precision[positions],
                xerr=recall_std[positions],
                yerr=precision_std[positions],
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                alpha=0.55,
            )

        axis.set_title(dataset, fontsize=12, weight="bold")
        axis.grid(alpha=0.25, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("Recall")
    for axis in axes[:, 0]:
        axis.set_ylabel("Precision")
    for axis in axes.flat[:len(datasets)]:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("DINOv3 vs. YOLO: Precision–Recall Curves", fontsize=17,
                 weight="bold", y=0.99)
    fig.text(
        0.5,
        0.015,
        "Lines show seed means; shaded regions and error bars show ±1 sample standard deviation.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed, summary = collect_best_seed_results()
    per_seed.to_csv(OUTPUT_DIR / "best_results_per_seed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "best_results_summary.csv", index=False)
    plot_family(summary, "DINOv3", OUTPUT_DIR / "dino_best_dice_iou.png")
    plot_family(summary, "YOLO", OUTPUT_DIR / "yolo_best_dice_iou.png")
    plot_comparison(summary, OUTPUT_DIR / "dino_vs_yolo_best_dice_iou.png")
    pr_statistics = collect_precision_recall_statistics()
    pr_statistics.to_csv(OUTPUT_DIR / "precision_recall_summary.csv", index=False)
    plot_precision_recall_comparison(
        pr_statistics, OUTPUT_DIR / "dino_vs_yolo_precision_recall.png"
    )
    print(summary.to_string(index=False))
    print(f"\nSaved plots and CSV files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
