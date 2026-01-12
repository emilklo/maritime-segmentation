#!/usr/bin/env python3
"""Training script for RF-DETR-Seg on LaRS dataset."""

import argparse
from pathlib import Path

import torch
import yaml

from rfdetr import RFDETRSegPreview


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

    # Extract config values with CLI overrides
    dataset_dir = args.dataset_dir or config["data"]["dataset_dir"]
    output_dir = args.output_dir or config["checkpoints"]["dir"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["data"]["batch_size"]
    lr = args.lr or config["training"]["lr"]
    resolution = config["data"]["input_size"]
    grad_accum_steps = args.grad_accum or config["training"]["grad_accum_steps"]
    checkpoint_interval = config["checkpoints"]["save_every_n_epochs"]

    # Fine-tuning parameters
    training_cfg = config["training"]
    lr_encoder = training_cfg.get("lr_encoder", lr * 1.5)
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    lr_drop = training_cfg.get("lr_drop", 100)
    drop_path = training_cfg.get("drop_path", 0.0)
    weight_decay = training_cfg.get("weight_decay", 1e-4)

    # Early stopping
    early_stopping = training_cfg.get("early_stopping", False)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 10)
    early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 0.001)

    # Logging
    use_wandb = args.wandb or config["logging"].get("wandb", False)
    use_tensorboard = args.tensorboard or config["logging"].get("tensorboard", True)
    project = config["logging"].get("project", "maritime-segmentation")

    print("=" * 60)
    print("RF-DETR-Seg Training on LaRS Maritime Dataset")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print()
    print("Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"  Learning rate: {lr} (encoder: {lr_encoder})")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  LR drop at epoch: {lr_drop}")
    print(f"  Drop path: {drop_path}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping: {early_stopping} (patience: {early_stopping_patience})")
    print(f"  Resolution: {resolution}")
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
    print("\nInitializing RF-DETR-Seg model...")
    model = RFDETRSegPreview()

    # Start training with all fine-tuning parameters
    print("Starting training...\n")
    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        lr_encoder=lr_encoder,
        resolution=resolution,
        checkpoint_interval=checkpoint_interval,
        # Fine-tuning params
        warmup_epochs=warmup_epochs,
        lr_drop=lr_drop,
        drop_path=drop_path,
        weight_decay=weight_decay,
        # Early stopping
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        # Logging
        wandb=use_wandb,
        tensorboard=use_tensorboard,
        project=project if use_wandb else None,
    )

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
