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
├── src/
│   ├── models/        # Model definitions, custom layers
│   ├── data/          # Dataset loaders, augmentations
│   ├── training/      # Training loops, callbacks
│   └── export/        # ONNX conversion scripts
├── scripts/           # CLI scripts for train/eval/export
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
- Location: [specify path or download instructions]
- Format: COCO-style annotations with RLE masks
- Train/val/test splits predefined

## Commands
```bash
uv sync                 # Install dependencies
uv run python scripts/train.py --config configs/default.yaml
uv run pytest           # Run tests
```

## Conventions
- Use uv for all package management (never pip install directly)
- Use PyTorch Lightning or plain PyTorch (no TensorFlow)
- Config files in YAML
- Type hints in Python code
- Ruff for linting and formatting

## Current Status
- [ ] Project setup
- [ ] Dataset loading pipeline
- [ ] RF-DETR-Seg baseline training
- [ ] Evaluation on LaRS val set
- [ ] ONNX export script
- [ ] Jetson deployment (future)
```