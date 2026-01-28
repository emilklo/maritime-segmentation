#!/bin/bash
#==============================================================================
# SLURM Job Script: Generate LaRS Panoptic Predictions
# Cluster: NTNU Idun HPC
#==============================================================================

#------------------------------------------------------------------------------
# Job Configuration
#------------------------------------------------------------------------------
#SBATCH --job-name=lars-predict
#SBATCH --account=share-ie-idi
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err

#------------------------------------------------------------------------------
# Resource Allocation (inference is less demanding than training)
#------------------------------------------------------------------------------
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00

#==============================================================================
# Environment Setup
#==============================================================================

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.8.0

echo ""
echo "=== Environment ==="
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

# Model checkpoint (best cosine LR model)
CHECKPOINT="checkpoints/cosine_run_23970692/checkpoint_best_ema.pth"

# Test images location
TEST_IMAGES="/cluster/work/emilkl/rf-detr/datasets/lars_rfdetr/test/images"

# Output directory
OUTPUT_DIR="predictions/rfdetr_seg_cosine_${SLURM_JOB_ID}"

# Inference parameters
RESOLUTION=576
CONF_THRESHOLD=0.3
MASK_THRESHOLD=0.5

#------------------------------------------------------------------------------
# Install dependencies if needed
#------------------------------------------------------------------------------

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

#------------------------------------------------------------------------------
# Run Prediction
#------------------------------------------------------------------------------

echo ""
echo "=== Running Inference ==="
echo "Checkpoint: $CHECKPOINT"
echo "Test images: $TEST_IMAGES"
echo "Output: $OUTPUT_DIR"
echo ""

uv run python scripts/generate_lars_predictions.py \
    --checkpoint "$CHECKPOINT" \
    --test-images "$TEST_IMAGES" \
    --output "$OUTPUT_DIR" \
    --resolution "$RESOLUTION" \
    --conf-threshold "$CONF_THRESHOLD" \
    --mask-threshold "$MASK_THRESHOLD"

#------------------------------------------------------------------------------
# Completion
#------------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Predictions saved to: $OUTPUT_DIR"
echo "Submission ZIP: predictions/rfdetr_seg_cosine_${SLURM_JOB_ID}.zip"
echo "========================================"
