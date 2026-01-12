#!/bin/bash
#==============================================================================
# SLURM Job Script: RF-DETR-Seg Training on LaRS Dataset
# Cluster: NTNU Idun HPC
#==============================================================================

#------------------------------------------------------------------------------
# Job Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=rfdetr-lars
#SBATCH --account=ie-idi          # Replace with your account
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

#------------------------------------------------------------------------------
# Resource Allocation
#------------------------------------------------------------------------------
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#------------------------------------------------------------------------------
# Notifications (optional)
#------------------------------------------------------------------------------
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@ntnu.no  # Replace with your email

#==============================================================================
# Environment Setup
#==============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Navigate to project directory
cd "$SLURM_SUBMIT_DIR"

# Create log directory if needed
mkdir -p logs/slurm

# Load required modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# Print environment info
echo ""
echo "=== Environment ==="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

#------------------------------------------------------------------------------
# UV Setup
#------------------------------------------------------------------------------

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "=== Installing Dependencies ==="
uv sync
echo ""

#==============================================================================
# Training Configuration
#==============================================================================

# Paths
DATASET_DIR="data/lars_rfdetr"
OUTPUT_DIR="checkpoints/run_${SLURM_JOB_ID}"
CONFIG="configs/default.yaml"

# Training hyperparameters (override config if needed)
EPOCHS=50
BATCH_SIZE=8
GRAD_ACCUM=2  # Effective batch size = 8 * 2 = 16

#==============================================================================
# Run Training
#==============================================================================

echo "=== Starting Training ==="
echo "Dataset: $DATASET_DIR"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo ""

uv run python scripts/train.py \
    --config "$CONFIG" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --tensorboard \
    "$@"  # Pass any additional arguments

#==============================================================================
# Completion
#==============================================================================

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "========================================"
