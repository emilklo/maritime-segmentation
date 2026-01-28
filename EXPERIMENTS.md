# Training Management Agent

Manage RF-DETR-Seg experiments across model sizes (nano through 2xlarge) with SLURM job launching, result tracking, and comparison tools.

## Quick Start

```bash
# 1. Launch a single experiment
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml

# 2. Check status
uv run python scripts/experiments/status.py

# 3. Compare results
uv run python scripts/experiments/compare.py --output plots/comparison
```

## Available Model Sizes

| Size | Config Example |
|------|---------------|
| `nano` | `configs/experiments/nano_baseline.yaml` |
| `small` | `configs/experiments/small_baseline.yaml` |
| `medium` | `configs/experiments/medium_baseline.yaml` |
| `large` | (create your own) |
| `xlarge` | (create your own) |
| `2xlarge` | (create your own) |

## Workflow

### 1. Define an Experiment

Create a YAML config in `configs/experiments/`:

```yaml
experiment:
  name: "nano-baseline"
  description: "Nano model with default hyperparameters"
  model_size: "nano"    # nano|small|medium|large|xlarge|2xlarge

data:
  dataset_dir: "data/lars_rfdetr"

training:
  epochs: 50
  lr: 1.0e-4
  lr_encoder: 1.5e-4
  warmup_epochs: 2
  lr_scheduler: "step"    # "step" or "cosine"
  lr_drop: 40             # for step scheduler
  lr_min_factor: 0.01     # for cosine scheduler
  weight_decay: 1.0e-4
  batch_size: 4
  grad_accum_steps: 4
  early_stopping: true
  early_stopping_patience: 15

checkpoints:
  save_every_n_epochs: 5

logging:
  tensorboard: true
  wandb: false
```

### 2. Launch on Idun

```bash
# Single experiment
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml

# Multiple experiments at once
uv run python scripts/experiments/launch.py \
    --configs configs/experiments/*.yaml

# Preview without submitting
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml \
    --dry-run

# With email notifications
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml \
    --mail-user your.email@ntnu.no

# Different SLURM account
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml \
    --account your-account
```

The launcher automatically:
- Allocates SLURM resources based on model size (nano/small get 48GB/8h, large gets 64GB/16h, etc.)
- Halves batch size for SLURM (cuDNN disabled = higher memory) and doubles grad accumulation to compensate
- Creates a unique output directory under `checkpoints/`
- Registers the experiment in `experiments/registry.json`

### 3. Monitor Status

```bash
# All experiments
uv run python scripts/experiments/status.py

# Filter by model size
uv run python scripts/experiments/status.py --model-size nano

# Filter by status
uv run python scripts/experiments/status.py --status running

# Best run per model size
uv run python scripts/experiments/status.py --leaderboard
```

The status checker automatically syncs with SLURM (`sacct`) to update job statuses and parses `log.txt` for completed runs to extract metrics.

### 4. Compare Results

```bash
# Compare all completed experiments
uv run python scripts/experiments/compare.py

# Compare specific experiments by ID
uv run python scripts/experiments/compare.py \
    --ids nano-baseline-20260128-143022 nano-cosine-20260128-150011

# Compare within a model size
uv run python scripts/experiments/compare.py --model-size nano

# Custom output directory
uv run python scripts/experiments/compare.py --output plots/my_comparison
```

Generates:
- `leaderboard.csv` - All results sorted by mask mAP
- `training_curves.png` - Loss and mAP curves overlaid
- `metrics_bar_comparison.png` - Final metrics side by side
- `model_size_comparison.png` - Best result per model size
- `hp_lr_scatter.png` - Learning rate vs performance (if enough runs)

### 5. Direct Training (without launcher)

You can also run training directly with the `--model-size` flag:

```bash
uv run python scripts/train.py \
    --config configs/experiments/nano_baseline.yaml \
    --model-size nano
```

## Pre-built Experiment Configs

| Config | Model | LR | Scheduler | Epochs | Description |
|--------|-------|----|-----------|--------|-------------|
| `nano_baseline.yaml` | nano | 1e-4 | step | 50 | Default HPs |
| `nano_cosine.yaml` | nano | 1e-4 | cosine | 60 | Cosine LR schedule |
| `small_baseline.yaml` | small | 1e-4 | step | 50 | Default HPs |
| `medium_baseline.yaml` | medium | 1e-4 | step | 50 | Default HPs |
| `medium_lowlr.yaml` | medium | 5e-5 | cosine | 60 | Lower LR exploration |

## Experiment Registry

All experiments are tracked in `experiments/registry.json`. This file is auto-managed by the tools. Each entry stores:

- Experiment ID, name, description, model size
- Full hyperparameters snapshot
- SLURM job ID
- Status (submitted/running/completed/failed)
- Output directory and log path
- Best metrics (mask mAP, bbox mAP, best epoch)

## Resource Allocation by Model Size

| Size | Memory | Time Limit |
|------|--------|-----------|
| nano, small | 48 GB | 8 hours |
| medium | 64 GB | 12 hours |
| large | 64 GB | 16 hours |
| xlarge, 2xlarge | 80 GB | 24 hours |

All jobs request 1x A100 GPU, 8 CPU cores.
