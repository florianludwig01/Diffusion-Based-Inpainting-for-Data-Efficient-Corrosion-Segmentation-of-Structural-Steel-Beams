import os
from pathlib import Path

import numpy as np
import pandas as pd

from testing.test import DINO_test, YOLO_test


def _seed_from_dir(path):
    try:
        return int(path.parent.name.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError(f"Invalid seed directory: {path.parent}") from exc


def find_seed_models(models_dir, model):
    extension = ".pth" if model == "dino" else ".pt"
    model_paths = list(Path(models_dir).glob(f"seed_*/best{extension}"))
    model_paths.sort(key=_seed_from_dir)
    if not model_paths:
        raise FileNotFoundError(
            f"No seed_*/best{extension} models found in {models_dir}."
        )
    return model_paths


def save_seed_aggregation(results, output_dir, parameter_column="threshold"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output_dir / "all_seed_results.csv", index=False)

    metric_columns = [
        column for column in all_results.columns
        if column not in {parameter_column, "seed"}
        and pd.api.types.is_numeric_dtype(all_results[column])
    ]
    grouped = all_results.groupby(parameter_column, sort=True)[metric_columns]
    statistics = (
        grouped.agg(["mean", "median", "std", "min", "max", "count"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "metric", "count": "n"})
    )
    statistics.to_csv(output_dir / "seed_statistics.csv", index=False)

    _save_seed_plots(
        all_results, statistics, metric_columns, parameter_column, output_dir
    )
    print(f"\nSeed results saved to: {output_dir}")


def _save_seed_plots(
    all_results, statistics, metric_columns, parameter_column, output_dir
):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_metrics = [column for column in metric_columns if column.startswith("dataset_")]
    fig, axes = plt.subplots(
        len(plot_metrics), 1, figsize=(10, 3.5 * len(plot_metrics)), squeeze=False
    )
    x_values = np.sort(all_results[parameter_column].unique())
    for axis, metric in zip(axes[:, 0], plot_metrics):
        for _, seed_rows in all_results.groupby("seed"):
            seed_rows = seed_rows.sort_values(parameter_column)
            axis.plot(
                seed_rows[parameter_column], seed_rows[metric],
                color="0.75", linewidth=0.8, alpha=0.7,
            )
        metric_stats = statistics[
            statistics["metric"] == metric
        ].sort_values(parameter_column)
        mean = metric_stats["mean"].to_numpy(dtype=float)
        std = metric_stats["std"].fillna(0).to_numpy(dtype=float)
        median = metric_stats["median"].to_numpy(dtype=float)
        axis.plot(x_values, mean, label="Mean", color="tab:blue", linewidth=2)
        axis.plot(x_values, median, label="Median", color="tab:orange", linewidth=2)
        axis.fill_between(
            x_values, mean - std, mean + std, color="tab:blue", alpha=0.2,
            label="Mean +/- standard deviation",
        )
        axis.set(title=metric, xlabel=parameter_column, ylabel=metric, ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_threshold_curves.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(plot_metrics), figsize=(4 * len(plot_metrics), 4), squeeze=False
    )
    best_per_seed = all_results.loc[
        all_results.groupby("seed")["dataset_iou"].idxmax()
    ]
    for axis, metric in zip(axes[0], plot_metrics):
        axis.boxplot(best_per_seed[metric].dropna(), tick_labels=[metric])
        axis.scatter(
            np.ones(len(best_per_seed)), best_per_seed[metric],
            color="tab:blue", alpha=0.7, zorder=3,
        )
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Metric distributions at each seed's best IoU point")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_seed_distributions.png", dpi=200)
    plt.close(fig)


def run_test(args):
    if not args.all_seeds:
        if not args.model_path:
            raise ValueError("Single-model testing requires --model_path.")
        tester = DINO_test(args) if args.model == "dino" else YOLO_test(args)
        tester.test()
        return

    if not args.models_dir:
        raise ValueError("--all_seeds requires --models_dir.")

    base_output_dir = Path(args.output_dir)
    results = []
    model_paths = find_seed_models(args.models_dir, args.model)
    for run_number, model_path in enumerate(model_paths, start=1):
        seed = _seed_from_dir(model_path)
        print(f"\nTest {run_number}/{len(model_paths)} | Seed: {seed}")
        args.seed = seed
        args.model_path = str(model_path)
        args.output_dir = str(base_output_dir / f"seed_{seed}")
        tester = DINO_test(args) if args.model == "dino" else YOLO_test(args)
        seed_results = tester.test().copy()
        seed_results["seed"] = seed
        if args.model == "yolo":
            seed_results = seed_results.rename(columns={"conf": "threshold"})
        results.append(seed_results)

    save_seed_aggregation(results, base_output_dir)
