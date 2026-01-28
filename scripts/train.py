#!/usr/bin/env python3
"""Training script for RF-DETR-Seg on LaRS dataset."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# Workaround for cuDNN issues with depthwise convolutions on some GPUs
# Completely disable cuDNN to use basic CUDA kernels
torch.backends.cudnn.enabled = False

# Patch numpy compatibility issue in rfdetr's coco_extended_metrics
def _patch_coco_extended_metrics():
    """Fix numpy argwhere scalar conversion in rfdetr."""
    try:
        import rfdetr.engine as engine
        original_func = engine.coco_extended_metrics

        def patched_coco_extended_metrics(coco_eval):
            # Get IoU thresholds
            iou_thrs = coco_eval.params.iouThrs
            # Fix: use .item() or flatten()[0] for numpy 2.x compatibility
            iou_50_idx = int(np.argwhere(np.isclose(iou_thrs, 0.50)).flatten()[0])
            iou_75_idx = int(np.argwhere(np.isclose(iou_thrs, 0.75)).flatten()[0])

            # precision has shape (T, R, K, A, M) = (iou, recall, class, area, maxDets)
            precision = coco_eval.eval["precision"]
            recall = coco_eval.eval["recall"]

            # AP metrics
            results = {
                "AP50": float(np.mean(precision[iou_50_idx, :, :, 0, 2])),
                "AP75": float(np.mean(precision[iou_75_idx, :, :, 0, 2])),
            }

            # Per-class AP50
            num_classes = precision.shape[2]
            for c in range(num_classes):
                ap50_c = float(np.mean(precision[iou_50_idx, :, c, 0, 2]))
                results[f"AP50_class_{c}"] = ap50_c

            return results

        engine.coco_extended_metrics = patched_coco_extended_metrics
        print("Applied numpy compatibility patch for coco_extended_metrics")
    except Exception as e:
        print(f"Warning: Could not apply coco_extended_metrics patch: {e}")

_patch_coco_extended_metrics()

from rfdetr import (
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
)

MODEL_CLASSES = {
    "nano": RFDETRSegNano,
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge,
    "xlarge": RFDETRSegXLarge,
    "2xlarge": RFDETRSeg2XLarge,
}


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR-Seg on LaRS maritime dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(MODEL_CLASSES.keys()),
        help="Model size (nano/small/medium/large/xlarge/2xlarge). Overrides config.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for registry updates on completion",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Determine model size: CLI > experiment config > default
    model_size = args.model_size
    if model_size is None:
        exp_cfg = config.get("experiment", {})
        model_size = exp_cfg.get("model_size")

    # Extract config values with CLI overrides
    # Support both old-style (data.dataset_dir) and experiment-style (data.dataset_dir) configs
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpoints", {})

    dataset_dir = args.dataset_dir or data_cfg.get("dataset_dir", "data/lars_rfdetr")
    output_dir = args.output_dir or ckpt_cfg.get("dir", "checkpoints")
    epochs = args.epochs or training_cfg.get("epochs", 50)
    batch_size = args.batch_size or training_cfg.get("batch_size", data_cfg.get("batch_size", 4))
    lr = args.lr or training_cfg.get("lr", 1e-4)
    grad_accum_steps = args.grad_accum or training_cfg.get("grad_accum_steps", 4)
    checkpoint_interval = ckpt_cfg.get("save_every_n_epochs", 5)

    # Resolution: only use if specified in config (model classes have their own defaults)
    resolution = data_cfg.get("input_size")

    # Fine-tuning parameters
    lr_encoder = training_cfg.get("lr_encoder", lr * 1.5)
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    lr_drop = training_cfg.get("lr_drop", 100)
    drop_path = training_cfg.get("drop_path", 0.0)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    lr_scheduler = training_cfg.get("lr_scheduler", "step")
    lr_min_factor = training_cfg.get("lr_min_factor", 0.0)

    # Early stopping
    early_stopping = training_cfg.get("early_stopping", False)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 10)
    early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 0.001)

    # Logging
    logging_cfg = config.get("logging", {})
    use_wandb = args.wandb or logging_cfg.get("wandb", False)
    use_tensorboard = args.tensorboard or logging_cfg.get("tensorboard", True)
    project = logging_cfg.get("project", "maritime-segmentation")

    print("=" * 60)
    print("RF-DETR-Seg Training on LaRS Maritime Dataset")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model size: {model_size or 'default (base)'}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    if args.experiment_id:
        print(f"Experiment ID: {args.experiment_id}")
    print()
    print("Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"  Learning rate: {lr} (encoder: {lr_encoder})")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  LR scheduler: {lr_scheduler}")
    if lr_scheduler == "step":
        print(f"  LR drop at epoch: {lr_drop}")
    else:
        print(f"  LR min factor: {lr_min_factor}")
    print(f"  Drop path: {drop_path}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping: {early_stopping} (patience: {early_stopping_patience})")
    if resolution:
        print(f"  Resolution: {resolution}")
    else:
        print("  Resolution: model default")
    print()
    print(f"Logging: W&B={use_wandb}, TensorBoard={use_tensorboard}")
    print("=" * 60)

    # Verify dataset exists
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Run `python scripts/prepare_lars.py` first."
        )

    # Initialize model
    model_cls_name = model_size or "base"
    if model_size and model_size in MODEL_CLASSES:
        model_cls = MODEL_CLASSES[model_size]
        print(f"\nInitializing {model_cls.__name__} model...")
        model = model_cls()
    else:
        # Fallback to legacy RFDETRSegPreview for backward compat
        from rfdetr import RFDETRSegPreview
        print("\nInitializing RF-DETR-Seg model (legacy preview)...")
        model = RFDETRSegPreview()

    # Workaround: some RF-DETR backbones use a different structure
    # (DinoV2 without .trunk attribute) that's incompatible with update_drop_path.
    try:
        from rfdetr.models.lwdetr import LWDETR
        original_update_drop_path = LWDETR.update_drop_path

        def safe_update_drop_path(self, drop_path):
            try:
                original_update_drop_path(self, drop_path)
            except AttributeError:
                pass

        LWDETR.update_drop_path = safe_update_drop_path
        print("Applied drop_path compatibility patch")
    except Exception as e:
        print(f"Warning: Could not apply drop_path patch: {e}")

    # Build training kwargs
    train_kwargs = dict(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        lr_encoder=lr_encoder,
        checkpoint_interval=checkpoint_interval,
        warmup_epochs=warmup_epochs,
        lr_drop=lr_drop,
        drop_path=drop_path,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        wandb=use_wandb,
        tensorboard=use_tensorboard,
    )
    if resolution:
        train_kwargs["resolution"] = resolution
    if use_wandb:
        train_kwargs["project"] = project

    # Start training
    print("Starting training...\n")
    model.train(**train_kwargs)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")

    # Update experiment registry if experiment_id was provided
    if args.experiment_id:
        try:
            from src.training.registry import parse_log_metrics, update_experiment_status

            log_path = Path(output_dir) / "log.txt"
            metrics = parse_log_metrics(log_path)

            if metrics:
                update_experiment_status(args.experiment_id, "completed", metrics)
                print(f"\nRegistry updated: {args.experiment_id} -> completed")
                print(f"  Best mask mAP@0.5:0.95: {metrics.get('mask_map_50_95', 'N/A')}")
                print(f"  Best epoch: {metrics.get('best_epoch', 'N/A')}")
            else:
                update_experiment_status(args.experiment_id, "completed")
                print(f"\nRegistry updated: {args.experiment_id} -> completed (no metrics parsed)")
        except Exception as e:
            print(f"\nWarning: Could not update registry: {e}")


if __name__ == "__main__":
    main()
