"""Experiment registry for tracking RF-DETR training runs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path("experiments/registry.json")


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load the experiment registry from disk."""
    if not path.exists():
        return {"experiments": []}
    with open(path) as f:
        return json.load(f)


def save_registry(registry: dict, path: Path = REGISTRY_PATH) -> None:
    """Save the experiment registry to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def register_experiment(
    config: dict,
    slurm_job_id: str | None = None,
    output_dir: str | None = None,
    path: Path = REGISTRY_PATH,
) -> str:
    """Register a new experiment in the registry.

    Args:
        config: Parsed experiment config dict.
        slurm_job_id: SLURM job ID if submitted.
        output_dir: Directory where checkpoints/logs are saved.
        path: Path to registry file.

    Returns:
        The generated experiment ID.
    """
    registry = load_registry(path)

    exp_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})

    name = exp_cfg.get("name", "unnamed")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{name}-{timestamp}"

    experiment = {
        "id": experiment_id,
        "name": name,
        "description": exp_cfg.get("description", ""),
        "model_size": exp_cfg.get("model_size", "base"),
        "config_path": config.get("_config_path", ""),
        "hyperparameters": {
            "lr": training_cfg.get("lr"),
            "lr_encoder": training_cfg.get("lr_encoder"),
            "warmup_epochs": training_cfg.get("warmup_epochs"),
            "lr_scheduler": training_cfg.get("lr_scheduler"),
            "lr_drop": training_cfg.get("lr_drop"),
            "lr_min_factor": training_cfg.get("lr_min_factor"),
            "weight_decay": training_cfg.get("weight_decay"),
            "epochs": training_cfg.get("epochs"),
            "batch_size": training_cfg.get("batch_size"),
            "grad_accum_steps": training_cfg.get("grad_accum_steps"),
        },
        "slurm_job_id": slurm_job_id,
        "status": "submitted" if slurm_job_id else "pending",
        "output_dir": output_dir or "",
        "log_path": f"{output_dir}/log.txt" if output_dir else "",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "best_metrics": None,
    }

    registry["experiments"].append(experiment)
    save_registry(registry, path)
    return experiment_id


def update_experiment_status(
    experiment_id: str,
    status: str,
    metrics: dict[str, Any] | None = None,
    path: Path = REGISTRY_PATH,
) -> None:
    """Update an experiment's status and optionally its metrics.

    Args:
        experiment_id: The experiment ID to update.
        status: New status (submitted, running, completed, failed).
        metrics: Optional dict of best metrics to store.
        path: Path to registry file.
    """
    registry = load_registry(path)

    for exp in registry["experiments"]:
        if exp["id"] == experiment_id:
            exp["status"] = status
            if status == "completed":
                exp["completed_at"] = datetime.now().isoformat()
            if metrics is not None:
                exp["best_metrics"] = metrics
            break

    save_registry(registry, path)


def get_experiments(
    model_size: str | None = None,
    status: str | None = None,
    path: Path = REGISTRY_PATH,
) -> list[dict]:
    """Get experiments, optionally filtered by model size and/or status.

    Args:
        model_size: Filter by model size (nano, small, medium, etc.).
        status: Filter by status (submitted, running, completed, failed).
        path: Path to registry file.

    Returns:
        List of matching experiment dicts.
    """
    registry = load_registry(path)
    experiments = registry["experiments"]

    if model_size is not None:
        experiments = [e for e in experiments if e.get("model_size") == model_size]
    if status is not None:
        experiments = [e for e in experiments if e.get("status") == status]

    return experiments


def get_best_experiment(
    model_size: str | None = None,
    metric: str = "mask_map_50_95",
    path: Path = REGISTRY_PATH,
) -> dict | None:
    """Get the best completed experiment by a given metric.

    Args:
        model_size: Filter by model size before finding best.
        metric: Metric key to rank by (default: mask_map_50_95).
        path: Path to registry file.

    Returns:
        The experiment dict with the highest metric value, or None.
    """
    experiments = get_experiments(model_size=model_size, status="completed", path=path)

    best = None
    best_value = -1.0

    for exp in experiments:
        metrics = exp.get("best_metrics")
        if metrics and metric in metrics:
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best = exp

    return best


def parse_log_metrics(log_path: str | Path) -> dict[str, Any] | None:
    """Parse a training log.txt file and extract best metrics.

    The log file contains one JSON object per line, one per epoch.

    Args:
        log_path: Path to the log.txt file.

    Returns:
        Dict with best metrics, or None if log can't be parsed.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    best_mask_map = -1.0
    best_epoch = -1
    best_metrics: dict[str, Any] = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = data.get("epoch")
            if epoch is None:
                continue

            coco_masks = data.get("test_coco_eval_masks", [])
            coco_bbox = data.get("test_coco_eval_bbox", [])

            mask_map_50_95 = coco_masks[0] if len(coco_masks) >= 1 else 0.0

            if mask_map_50_95 > best_mask_map:
                best_mask_map = mask_map_50_95
                best_epoch = epoch
                best_metrics = {
                    "mask_map_50_95": coco_masks[0] if len(coco_masks) >= 1 else None,
                    "mask_map_50": coco_masks[1] if len(coco_masks) >= 2 else None,
                    "bbox_map_50_95": coco_bbox[0] if len(coco_bbox) >= 1 else None,
                    "bbox_map_50": coco_bbox[1] if len(coco_bbox) >= 2 else None,
                    "best_epoch": epoch,
                }

    if best_epoch < 0:
        return None
    return best_metrics
