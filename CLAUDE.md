# Maritime Segmentation for Edge Deployment

## Project Overview
Transformer-based instance segmentation for autonomous USV navigation. Target: real-time inference (25-60 FPS) on NVIDIA Jetson AGX Orin Industrial.

## Current Phase
Local development and model fine-tuning. Jetson deployment comes later.

## Target Model
RF-DETR-Seg fine-tuned on LaRS (Lakes, Rivers, Seas) maritime dataset.
- 11 classes: water, sky, boat, buoy, swimmer, paddle board, animal, obstacle, etc.
- Severe class imbalance (swimmers/obstacles are rare but safety-critical)

## Hardware Targets
- **Training**: Local GPU / Idun HPC cluster / Google Colab
- **Deployment**: Jetson AGX Orin Industrial (JetPack 6.2, L4T 36.4.3)
  - 61GB unified RAM (shared with other processes)
  - Access: `ssh mradmin@archytas`
  - Power mode: MAXN available

## Tech Stack
- **Package manager**: uv (use `uv add`, `uv run`, `uv sync`)
- PyTorch for training
- ONNX for model export
- TensorRT for Jetson optimization (build on-device)
- Docker for deployment isolation

## Project Structure
```
maritime-segmentation/
├── configs/           # Training configs, hyperparameters
│   └── experiments/   # Experiment-specific configs
├── src/
│   ├── models/        # Model definitions, custom layers
│   ├── data/          # Dataset loaders, augmentations
│   ├── training/      # Training loops, callbacks, registry
│   └── export/        # ONNX conversion scripts
├── scripts/           # CLI scripts for train/eval/export
│   ├── train.py       # Training entry point
│   ├── prepare_lars.py # Data preparation
│   ├── export_onnx.py # ONNX export
│   ├── predict.py     # Inference
│   ├── experiments/   # Experiment management
│   │   ├── launch.py  # Launch via SLURM
│   │   ├── status.py  # Check status
│   │   └── compare.py # Compare & plot results
│   └── slurm/         # SLURM templates
│       ├── train_template.slurm
│       └── predict.slurm
├── notebooks/         # Experimentation
├── deployment/        # Dockerfile, inference code (later)
├── checkpoints/       # Model weights (gitignored)
└── data/              # Dataset symlinks (gitignored)
```

## Key Constraints
- Model must fit in <8GB GPU memory for inference (other processes need headroom)
- Target latency: <40ms per frame at 640x640 input
- Must handle maritime-specific challenges: water reflections, horizon ambiguity, small distant objects

## LaRS Dataset
- Location: `~/datasets/lars` (symlinked to `data/lars`)
- Format: COCO-style instance annotations
- Annotations: `lars_v1.0.0_annotations/{split}/lars_{split}_instances_all.json`
- Images: `lars_v1.0.0_images/{split}/images/`
- Train: 2605 samples

## Commands
```bash
uv sync                 # Install dependencies
uv run python scripts/train.py --config configs/default.yaml
uv run pytest           # Run tests
```

## Idun HPC Training
Before training on Idun, prepare the dataset:
```bash
# 1. Upload LaRS dataset to Idun (or use existing location)
# 2. Prepare RF-DETR format
uv sync
uv run python scripts/prepare_lars.py --lars-root /path/to/lars/on/idun

# 3. Launch experiment with the experiment launcher
uv run python scripts/experiments/launch.py \
    --config configs/experiments/nano_baseline.yaml \
    --account your-account \
    --mail-user your.email@ntnu.no

# Or use the template directly:
#    Edit: scripts/slurm/train_template.slurm
sbatch scripts/slurm/train_template.slurm

# 5. Monitor
squeue -u $USER
tail -f logs/slurm/<job_id>-rfdetr-lars.out
```

## ONNX Export & Jetson Deployment
After training, export the model for Jetson:
```bash
# Export to ONNX
uv run python scripts/export_onnx.py \
    --checkpoint checkpoints/best_ema.pt \
    --input-size 640 \
    --validate

# Copy to Jetson
scp checkpoints/best_ema.onnx mradmin@archytas:~/models/

# On Jetson: convert to TensorRT
trtexec --onnx=best_ema.onnx \
        --saveEngine=rfdetr_seg.engine \
        --fp16
```

Model size estimates:
- RF-DETR-Seg: 34.2M params
- FP16: ~70MB weights, ~1-2GB runtime memory
- Target latency: <40ms at 640×640

## Conventions
- Use uv for all package management (never pip install directly)
- Use PyTorch Lightning or plain PyTorch (no TensorFlow)
- Config files in YAML
- Type hints in Python code
- Ruff for linting and formatting

## Current Status
- [x] Project setup
- [x] Dataset loading pipeline
- [x] RF-DETR-Seg baseline training (ready, needs CUDA GPU)
- [ ] Evaluation on LaRS val set
- [x] ONNX export script
- [ ] Jetson deployment (future)