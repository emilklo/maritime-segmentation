#!/usr/bin/env python3
"""Check status of RF-DETR-Seg experiments from the registry."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.registry import (
    get_best_experiment,
    get_experiments,
    load_registry,
    parse_log_metrics,
    save_registry,
    update_experiment_status,
)


def sync_slurm_status(registry: dict) -> bool:
    """Check SLURM for job status updates and sync to registry.

    Returns True if any updates were made.
    """
    updated = False
    job_ids = []
    job_to_exp = {}

    for exp in registry["experiments"]:
        if exp["status"] in ("submitted", "running") and exp.get("slurm_job_id"):
            job_ids.append(exp["slurm_job_id"])
            job_to_exp[exp["slurm_job_id"]] = exp

    if not job_ids:
        return False

    try:
        result = subprocess.run(
            [
                "sacct",
                "-j", ",".join(job_ids),
                "--format=JobID,State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].split(".")[0]  # Strip step suffixes
        state = parts[1]

        if job_id not in job_to_exp:
            continue

        exp = job_to_exp[job_id]

        if state in ("RUNNING",) and exp["status"] == "submitted":
            exp["status"] = "running"
            updated = True
        elif state in ("COMPLETED",) and exp["status"] != "completed":
            exp["status"] = "completed"
            # Try to parse metrics from log
            log_path = exp.get("log_path", "")
            if log_path:
                metrics = parse_log_metrics(log_path)
                if metrics:
                    exp["best_metrics"] = metrics
            updated = True
        elif state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            exp["status"] = "failed"
            updated = True

    return updated


def format_table(experiments: list[dict]) -> str:
    """Format experiments as a table string."""
    if not experiments:
        return "No experiments found."

    # Column definitions: (header, width, getter)
    columns = [
        ("ID", 32, lambda e: e["id"][:32]),
        ("Size", 8, lambda e: e.get("model_size", "?")[:8]),
        ("LR", 8, lambda e: f"{e.get('hyperparameters', {}).get('lr', '?'):.0e}" if isinstance(e.get("hyperparameters", {}).get("lr"), (int, float)) else "?"),
        ("Sched", 8, lambda e: (e.get("hyperparameters", {}).get("lr_scheduler", "?") or "?")[:8]),
        ("Status", 11, lambda e: e.get("status", "?")[:11]),
        ("Mask mAP", 9, lambda e: f"{e['best_metrics']['mask_map_50_95']:.3f}" if e.get("best_metrics") and e["best_metrics"].get("mask_map_50_95") is not None else "-"),
        ("BBox mAP", 9, lambda e: f"{e['best_metrics']['bbox_map_50_95']:.3f}" if e.get("best_metrics") and e["best_metrics"].get("bbox_map_50_95") is not None else "-"),
        ("Epoch", 5, lambda e: str(e["best_metrics"]["best_epoch"]) if e.get("best_metrics") and e["best_metrics"].get("best_epoch") is not None else "-"),
    ]

    # Build header
    header_parts = []
    sep_parts = []
    for name, width, _ in columns:
        header_parts.append(f" {name:<{width}} ")
        sep_parts.append("-" * (width + 2))

    header = "|".join(header_parts)
    separator = "+".join(sep_parts)
    top_border = "=" * len(separator)

    lines = [top_border, header, separator]

    for exp in experiments:
        row_parts = []
        for _, width, getter in columns:
            value = getter(exp)
            row_parts.append(f" {value:<{width}} ")
        lines.append("|".join(row_parts))

    lines.append(top_border)
    return "\n".join(lines)


def format_leaderboard(registry: dict) -> str:
    """Format a leaderboard showing best run per model size."""
    sizes = ["nano", "small", "medium", "large", "xlarge", "2xlarge"]
    lines = []
    lines.append("=" * 70)
    lines.append("LEADERBOARD - Best Run Per Model Size")
    lines.append("=" * 70)
    lines.append(
        f"{'Size':<10} {'Mask mAP@.5:.95':<18} {'BBox mAP@.5:.95':<18} "
        f"{'Best Epoch':<12} {'Experiment'}"
    )
    lines.append("-" * 70)

    for size in sizes:
        best = get_best_experiment(model_size=size)
        if best and best.get("best_metrics"):
            m = best["best_metrics"]
            lines.append(
                f"{size:<10} "
                f"{m.get('mask_map_50_95', 0):<18.4f} "
                f"{m.get('bbox_map_50_95', 0):<18.4f} "
                f"{str(m.get('best_epoch', '-')):<12} "
                f"{best['id']}"
            )
        else:
            lines.append(f"{size:<10} {'(no completed runs)'}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check status of RF-DETR-Seg experiments"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Filter by model size",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        choices=["submitted", "running", "completed", "failed", "pending"],
        help="Filter by status",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Show best runs per model size",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip SLURM status sync",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment Registry - Maritime Segmentation")
    print("=" * 60)
    print()

    # Load and sync
    registry = load_registry()

    if not args.no_sync:
        if sync_slurm_status(registry):
            save_registry(registry)
            print("(Synced SLURM job statuses)")
            print()

    if args.leaderboard:
        print(format_leaderboard(registry))
        return

    experiments = get_experiments(model_size=args.model_size, status=args.status)
    print(format_table(experiments))

    # Summary
    total = len(experiments)
    completed = sum(1 for e in experiments if e.get("status") == "completed")
    running = sum(1 for e in experiments if e.get("status") in ("submitted", "running"))
    failed = sum(1 for e in experiments if e.get("status") == "failed")
    print()
    print(f"Total: {total} | Completed: {completed} | Running: {running} | Failed: {failed}")


if __name__ == "__main__":
    main()
