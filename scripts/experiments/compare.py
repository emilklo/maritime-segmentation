#!/usr/bin/env python3
"""Compare RF-DETR-Seg experiments from the registry."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.registry import get_experiments, load_registry


def parse_log_file(log_path: Path) -> dict | None:
    """Parse JSON log file and extract per-epoch metrics."""
    if not log_path.exists():
        return None

    epochs = []
    train_loss = []
    val_loss = []
    map_50_95 = []
    map_50 = []
    mask_map_50_95 = []
    mask_map_50 = []

    with open(log_path) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                epoch = data.get("epoch")
                if epoch is None:
                    continue

                epochs.append(epoch)
                train_loss.append(data.get("train_loss", np.nan))
                val_loss.append(data.get("test_loss", np.nan))

                coco_eval_bbox = data.get("test_coco_eval_bbox", [])
                if len(coco_eval_bbox) >= 2:
                    map_50_95.append(coco_eval_bbox[0])
                    map_50.append(coco_eval_bbox[1])
                else:
                    map_50_95.append(np.nan)
                    map_50.append(np.nan)

                coco_eval_segm = data.get("test_coco_eval_masks", [])
                if len(coco_eval_segm) >= 2:
                    mask_map_50_95.append(coco_eval_segm[0])
                    mask_map_50.append(coco_eval_segm[1])
                else:
                    mask_map_50_95.append(np.nan)
                    mask_map_50.append(np.nan)

            except json.JSONDecodeError:
                continue

    if not epochs:
        return None

    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_loss": np.array(val_loss),
        "map_50_95": np.array(map_50_95),
        "map_50": np.array(map_50),
        "mask_map_50_95": np.array(mask_map_50_95),
        "mask_map_50": np.array(mask_map_50),
    }


def load_experiment_curves(experiments: list[dict]) -> list[tuple[dict, dict]]:
    """Load training curves for a list of experiments.

    Returns list of (experiment, curves) tuples where curves were parseable.
    """
    results = []
    for exp in experiments:
        log_path = Path(exp.get("log_path", ""))
        if not log_path.exists():
            # Also try output_dir/log.txt
            alt_path = Path(exp.get("output_dir", "")) / "log.txt"
            if alt_path.exists():
                log_path = alt_path
        curves = parse_log_file(log_path)
        if curves is not None:
            results.append((exp, curves))
    return results


def save_leaderboard_csv(experiments: list[dict], output_path: Path) -> None:
    """Save experiment results as CSV leaderboard."""
    rows = []
    for exp in experiments:
        m = exp.get("best_metrics", {}) or {}
        hp = exp.get("hyperparameters", {}) or {}
        rows.append({
            "id": exp["id"],
            "model_size": exp.get("model_size", ""),
            "status": exp.get("status", ""),
            "lr": hp.get("lr", ""),
            "lr_encoder": hp.get("lr_encoder", ""),
            "lr_scheduler": hp.get("lr_scheduler", ""),
            "epochs_config": hp.get("epochs", ""),
            "batch_size": hp.get("batch_size", ""),
            "grad_accum": hp.get("grad_accum_steps", ""),
            "weight_decay": hp.get("weight_decay", ""),
            "mask_map_50_95": m.get("mask_map_50_95", ""),
            "mask_map_50": m.get("mask_map_50", ""),
            "bbox_map_50_95": m.get("bbox_map_50_95", ""),
            "bbox_map_50": m.get("bbox_map_50", ""),
            "best_epoch": m.get("best_epoch", ""),
        })

    # Sort by mask_map_50_95 descending
    rows.sort(key=lambda r: r.get("mask_map_50_95") or 0, reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved leaderboard: {output_path}")


def plot_training_curves(
    exp_curves: list[tuple[dict, dict]], output_dir: Path
) -> None:
    """Plot training curves overlaid for multiple experiments."""
    if not exp_curves:
        print("No training curves to plot.")
        return

    # Assign colors
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves Comparison", fontsize=16, fontweight="bold")

    metric_configs = [
        ("train_loss", "Training Loss", axes[0, 0], None),
        ("map_50_95", "Detection mAP@0.5:0.95", axes[0, 1], (0, None)),
        ("mask_map_50_95", "Mask mAP@0.5:0.95", axes[1, 0], (0, None)),
        ("mask_map_50", "Mask mAP@0.5", axes[1, 1], (0, None)),
    ]

    for metric_key, title, ax, ylim in metric_configs:
        for i, (exp, curves) in enumerate(exp_curves):
            color = colors[i % len(colors)]
            label = f"{exp.get('model_size', '?')} - {exp['name']}"
            ax.plot(
                curves["epochs"], curves[metric_key],
                label=label, linewidth=2, color=color, alpha=0.8,
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_bar_comparison(experiments: list[dict], output_dir: Path) -> None:
    """Plot final metrics as grouped bar chart."""
    completed = [e for e in experiments if e.get("best_metrics")]
    if not completed:
        print("No completed experiments with metrics to compare.")
        return

    # Sort by mask mAP descending
    completed.sort(
        key=lambda e: e["best_metrics"].get("mask_map_50_95", 0), reverse=True
    )

    labels = [f"{e.get('model_size', '?')}\n{e['name']}" for e in completed]
    metrics = {
        "Mask mAP@.5:.95": [e["best_metrics"].get("mask_map_50_95", 0) for e in completed],
        "Mask mAP@.5": [e["best_metrics"].get("mask_map_50", 0) for e in completed],
        "BBox mAP@.5:.95": [e["best_metrics"].get("bbox_map_50_95", 0) for e in completed],
        "BBox mAP@.5": [e["best_metrics"].get("bbox_map_50", 0) for e in completed],
    }

    x = np.arange(len(labels))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(max(10, len(completed) * 2.5), 7))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=metric_name, color=colors[i % len(colors)],
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Experiment Metrics Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, max(max(v) for v in metrics.values()) * 1.15)

    plt.tight_layout()
    out_path = output_dir / "metrics_bar_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_model_size_comparison(experiments: list[dict], output_dir: Path) -> None:
    """Plot best result per model size."""
    sizes = ["nano", "small", "medium", "large", "xlarge", "2xlarge"]
    best_per_size = {}

    for exp in experiments:
        if not exp.get("best_metrics"):
            continue
        size = exp.get("model_size", "")
        if size not in sizes:
            continue
        current = best_per_size.get(size)
        if current is None or (
            exp["best_metrics"].get("mask_map_50_95", 0)
            > current["best_metrics"].get("mask_map_50_95", 0)
        ):
            best_per_size[size] = exp

    if not best_per_size:
        print("No completed experiments to compare across model sizes.")
        return

    present_sizes = [s for s in sizes if s in best_per_size]
    mask_maps = [
        best_per_size[s]["best_metrics"].get("mask_map_50_95", 0) for s in present_sizes
    ]
    bbox_maps = [
        best_per_size[s]["best_metrics"].get("bbox_map_50_95", 0) for s in present_sizes
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(present_sizes))
    width = 0.35

    bars1 = ax.bar(x - width / 2, mask_maps, width, label="Mask mAP@.5:.95", color="#3498db")
    bars2 = ax.bar(x + width / 2, bbox_maps, width, label="BBox mAP@.5:.95", color="#e74c3c")

    for bars, values in [(bars1, mask_maps), (bars2, bbox_maps)]:
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(present_sizes, fontsize=12)
    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("mAP@0.5:0.95", fontsize=12)
    ax.set_title("Best Performance by Model Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    out_path = output_dir / "model_size_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_hp_scatter(experiments: list[dict], output_dir: Path) -> None:
    """Plot hyperparameter impact scatter plots."""
    completed = [
        e for e in experiments
        if e.get("best_metrics") and e.get("hyperparameters")
    ]
    if len(completed) < 3:
        return  # Not enough data for meaningful scatter

    lrs = [e["hyperparameters"].get("lr", 0) for e in completed]
    mask_maps = [e["best_metrics"].get("mask_map_50_95", 0) for e in completed]
    sizes = [e.get("model_size", "?") for e in completed]

    # Color by model size
    size_set = sorted(set(sizes))
    size_to_color = {s: plt.cm.tab10.colors[i % 10] for i, s in enumerate(size_set)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for s in size_set:
        idx = [i for i, sz in enumerate(sizes) if sz == s]
        ax.scatter(
            [lrs[i] for i in idx],
            [mask_maps[i] for i in idx],
            label=s, color=size_to_color[s], s=100, edgecolors="black", linewidths=0.5,
        )

    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Mask mAP@0.5:0.95", fontsize=12)
    ax.set_title("Learning Rate vs Performance", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(title="Model Size", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = output_dir / "hp_lr_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_leaderboard(experiments: list[dict]) -> None:
    """Print leaderboard to console."""
    completed = [e for e in experiments if e.get("best_metrics")]
    if not completed:
        print("No completed experiments with metrics.")
        return

    completed.sort(
        key=lambda e: e["best_metrics"].get("mask_map_50_95", 0), reverse=True
    )

    print()
    print("=" * 90)
    print("EXPERIMENT LEADERBOARD (sorted by Mask mAP@0.5:0.95)")
    print("=" * 90)
    print(
        f"{'#':<4} {'Experiment':<30} {'Size':<8} {'Mask mAP':<10} "
        f"{'BBox mAP':<10} {'LR':<10} {'Sched':<8} {'Epoch':<6}"
    )
    print("-" * 90)

    for rank, exp in enumerate(completed, 1):
        m = exp["best_metrics"]
        hp = exp.get("hyperparameters", {})
        lr_str = f"{hp.get('lr', 0):.0e}" if isinstance(hp.get("lr"), (int, float)) else "?"
        print(
            f"{rank:<4} {exp['id'][:30]:<30} {exp.get('model_size', '?'):<8} "
            f"{m.get('mask_map_50_95', 0):<10.4f} "
            f"{m.get('bbox_map_50_95', 0):<10.4f} "
            f"{lr_str:<10} "
            f"{(hp.get('lr_scheduler') or '?'):<8} "
            f"{str(m.get('best_epoch', '-')):<6}"
        )

    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RF-DETR-Seg experiments"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Specific experiment IDs to compare",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Filter by model size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/comparison",
        help="Output directory for plots",
    )
    # Legacy support for old --old-log / --new-log interface
    parser.add_argument("--old-log", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--new-log", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--old-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--new-name", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Legacy two-model comparison mode
    if args.old_log and args.new_log:
        from scripts.compare_models import create_comparison_plots

        old_metrics = parse_log_file(Path(args.old_log))
        new_metrics = parse_log_file(Path(args.new_log))
        if old_metrics and new_metrics:
            create_comparison_plots(
                old_metrics, new_metrics, output_dir,
                args.old_name or "Model A",
                args.new_name or "Model B",
            )
        return

    # Registry-based comparison
    experiments = get_experiments(model_size=args.model_size)

    if args.ids:
        experiments = [e for e in experiments if e["id"] in args.ids]

    if not experiments:
        print("No experiments found matching filters.")
        return

    # Print leaderboard
    print_leaderboard(experiments)

    # Save CSV
    save_leaderboard_csv(experiments, output_dir / "leaderboard.csv")

    # Load curves for completed experiments
    completed = [e for e in experiments if e.get("status") == "completed"]
    exp_curves = load_experiment_curves(completed)

    # Generate plots
    if exp_curves:
        plot_training_curves(exp_curves, output_dir)
    plot_bar_comparison(experiments, output_dir)
    plot_model_size_comparison(experiments, output_dir)
    plot_hp_scatter(experiments, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
