#!/usr/bin/env python3
"""Training script for maritime instance segmentation."""

import argparse
from pathlib import Path

import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train maritime instance segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root from config",
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
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    print("=" * 60)
    print("Maritime Instance Segmentation Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print(f"Input size: {config['data']['input_size']}")
    print("=" * 60)

    # TODO: Initialize dataset
    # TODO: Initialize model
    # TODO: Initialize optimizer and scheduler
    # TODO: Training loop

    print("Training not yet implemented. Project skeleton ready.")


if __name__ == "__main__":
    main()
