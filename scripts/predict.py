#!/usr/bin/env python3
"""Generate LaRS panoptic segmentation predictions for leaderboard submission.

Output format for LaRS panoptic:
- R channel: class ID (0-10)
- G+B channels: instance ID (16-bit, G*256 + B)

Class mapping (LaRS):
  0: Static Obstacle (stuff)
  1: Water (stuff)
  2: Sky (stuff)
  3: Boat/ship (thing)
  4: Row boats (thing)
  5: Paddle board (thing)
  6: Buoy (thing)
  7: Swimmer (thing)
  8: Animal (thing)
  9: Float (thing)
  10: Other (thing)

Usage:
    uv run python scripts/generate_lars_predictions.py \
        --checkpoint checkpoints/cosine_run_23970692/checkpoint_best_ema.pth \
        --test-images /path/to/lars/test/images \
        --output predictions/rfdetr_seg_cosine \
        --resolution 576
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Disable cuDNN for compatibility (same as training)
torch.backends.cudnn.enabled = False


# LaRS class mapping (RF-DETR training order -> LaRS submission order)
# RF-DETR was trained with this category order from COCO annotations
RFDETR_TO_LARS = {
    0: 0,   # Static Obstacle -> 0
    1: 1,   # Water -> 1
    2: 2,   # Sky -> 2
    3: 3,   # Boat/ship -> 3
    4: 4,   # Row boats -> 4
    5: 5,   # Paddle board -> 5
    6: 6,   # Buoy -> 6
    7: 7,   # Swimmer -> 7
    8: 8,   # Animal -> 8
    9: 9,   # Float -> 9
    10: 10, # Other -> 10
}

# Stuff classes (no instance ID needed, use 0)
STUFF_CLASSES = {1, 2}  # Water, Sky

# Default class for background pixels
BACKGROUND_CLASS = 1  # Water (most common background in maritime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaRS panoptic predictions for leaderboard submission"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth file)",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        required=True,
        help="Path to LaRS test images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions/rfdetr_seg",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=576,
        help="Model input resolution (default: 576)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for mask binarization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cuda", num_classes: int = 11):
    """Load trained RF-DETR-Seg model."""
    from rfdetr import RFDETRSegPreview
    from rfdetr.config import RFDETRSegPreviewConfig

    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint first to get training args
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get args from checkpoint if available
    if "args" in checkpoint:
        args = checkpoint["args"]
        num_classes = getattr(args, "num_classes", num_classes)
        print(f"Using num_classes={num_classes} from checkpoint")

    # Create model with correct num_classes by overriding config
    # RFDETRSegPreview doesn't properly pass num_classes, so we set it directly
    model = RFDETRSegPreview.__new__(RFDETRSegPreview)
    model.model_config = RFDETRSegPreviewConfig(num_classes=num_classes)
    model.callbacks = defaultdict(list)
    model._is_optimized_for_inference = False
    model._has_warned_about_not_being_optimized_for_inference = False
    model._optimized_has_been_compiled = False
    model._optimized_batch_size = None
    model._optimized_resolution = None
    model._optimized_dtype = None

    # Skip pretrain weight download since we're loading our own checkpoint
    from rfdetr.main import RFDETRModel
    model.model = RFDETRModel(model.model_config)
    model.model.inference_model = None

    print(f"Model initialized with num_classes={model.model_config.num_classes}")

    # Prefer EMA weights if available
    if "model_ema" in checkpoint:
        print("Using EMA weights")
        state_dict = checkpoint["model_ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.model.model.load_state_dict(state_dict)
    model.model.model.to(device)
    model.model.model.eval()

    return model


def preprocess_image(image_path: Path, resolution: int, device: str):
    """Load and preprocess image for inference."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    # Resize to model resolution
    image_resized = image.resize((resolution, resolution), Image.BILINEAR)

    # Convert to tensor and normalize
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0).to(device), original_size


def instances_to_panoptic(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    original_size: tuple,
    conf_threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Convert instance segmentation output to panoptic format.

    Args:
        boxes: Detection boxes [N, 4] in xyxy format (normalized)
        scores: Confidence scores [N]
        labels: Class labels [N]
        masks: Instance masks [N, H, W] (model resolution)
        original_size: (W, H) original image size
        conf_threshold: Minimum confidence to include detection
        mask_threshold: Threshold for mask binarization

    Returns:
        Panoptic prediction as RGB image [H, W, 3]
        - R: class ID
        - G: instance ID high byte
        - B: instance ID low byte
    """
    W, H = original_size

    # Initialize panoptic output with background (water)
    panoptic = np.zeros((H, W, 3), dtype=np.uint8)
    panoptic[:, :, 0] = BACKGROUND_CLASS  # Default to water class
    # G, B = 0 for stuff classes (no instance ID)

    # Filter by confidence
    valid_mask = scores >= conf_threshold
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    masks = masks[valid_mask]

    if len(masks) == 0:
        return panoptic

    # Sort by score (lowest first, so higher confidence overwrites)
    order = np.argsort(scores)
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]
    masks = masks[order]

    # Track instance IDs per class
    instance_counters = {}

    for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
        # Map to LaRS class ID
        lars_class = RFDETR_TO_LARS.get(int(label), 10)  # Default to "Other"

        # Resize mask to original image size
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_resized.resize((W, H), Image.BILINEAR)
        mask_binary = np.array(mask_resized) > (mask_threshold * 255)

        if not mask_binary.any():
            continue

        # Set class ID in R channel
        panoptic[mask_binary, 0] = lars_class

        # Set instance ID in G+B channels
        if lars_class in STUFF_CLASSES:
            # Stuff classes have no instance ID
            panoptic[mask_binary, 1] = 0
            panoptic[mask_binary, 2] = 0
        else:
            # Thing classes get unique instance IDs
            if lars_class not in instance_counters:
                instance_counters[lars_class] = 0
            instance_counters[lars_class] += 1
            instance_id = instance_counters[lars_class]

            # Encode as G*256 + B (16-bit instance ID)
            panoptic[mask_binary, 1] = (instance_id >> 8) & 0xFF  # High byte
            panoptic[mask_binary, 2] = instance_id & 0xFF         # Low byte

    return panoptic


def run_inference(model, image_tensor, device):
    """Run model inference and get predictions."""
    with torch.no_grad():
        # RF-DETR-Seg returns detections via predict() method
        # But we need raw outputs, so use the underlying model
        outputs = model.model.model(image_tensor)

    # Post-process outputs
    # RF-DETR outputs: boxes, scores, labels, masks
    if isinstance(outputs, dict):
        boxes = outputs.get("pred_boxes", outputs.get("boxes"))
        scores = outputs.get("pred_logits", outputs.get("scores"))
        masks = outputs.get("pred_masks", outputs.get("masks"))
    elif isinstance(outputs, (list, tuple)):
        # Assume order: boxes, scores, labels, masks
        if len(outputs) >= 4:
            boxes, scores, labels, masks = outputs[:4]
        else:
            raise ValueError(f"Unexpected output format: {len(outputs)} elements")
    else:
        raise ValueError(f"Unexpected output type: {type(outputs)}")

    return boxes, scores, masks


@torch.no_grad()
def generate_predictions(
    model,
    test_dir: Path,
    output_dir: Path,
    resolution: int,
    conf_threshold: float,
    mask_threshold: float,
    device: str,
):
    """Generate predictions for all test images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all test images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in test_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    print(f"Found {len(image_paths)} test images")
    print(f"Output directory: {output_dir}")

    for image_path in tqdm(image_paths, desc="Generating predictions"):
        # Preprocess
        img_tensor, original_size = preprocess_image(image_path, resolution, device)

        # Run inference using the high-level predict API
        results = model.predict(str(image_path), threshold=conf_threshold)

        # Extract predictions from results
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.cpu().numpy() if torch.is_tensor(results.boxes) else np.array(results.boxes)
            scores = results.scores.cpu().numpy() if torch.is_tensor(results.scores) else np.array(results.scores)
            labels = results.labels.cpu().numpy() if torch.is_tensor(results.labels) else np.array(results.labels)

            # Get masks
            if hasattr(results, 'masks') and results.masks is not None:
                masks = results.masks.cpu().numpy() if torch.is_tensor(results.masks) else np.array(results.masks)
            else:
                masks = np.zeros((len(boxes), resolution, resolution))
        else:
            # No detections
            boxes = np.zeros((0, 4))
            scores = np.zeros(0)
            labels = np.zeros(0, dtype=np.int32)
            masks = np.zeros((0, resolution, resolution))

        # Convert to panoptic format
        panoptic = instances_to_panoptic(
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks=masks,
            original_size=original_size,
            conf_threshold=conf_threshold,
            mask_threshold=mask_threshold,
        )

        # Save as PNG
        output_path = output_dir / f"{image_path.stem}.png"
        Image.fromarray(panoptic).save(output_path)

    print(f"\nPredictions saved to: {output_dir}")
    print(f"Total images processed: {len(image_paths)}")


def create_submission_zip(output_dir: Path, method_name: str):
    """Create ZIP archive for submission."""
    import shutil

    zip_path = output_dir.parent / f"{method_name}.zip"
    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        output_dir.parent,
        output_dir.name
    )
    print(f"\nSubmission archive created: {zip_path}")
    print("Submit this file to https://macvi.org/")
    return zip_path


def main():
    args = parse_args()

    print("=" * 60)
    print("LaRS Panoptic Prediction Generator")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test images: {args.test_images}")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.resolution}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Mask threshold: {args.mask_threshold}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dir = Path(args.test_images)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_dir}")

    output_dir = Path(args.output)

    # Load model (11 classes for LaRS dataset)
    model = load_model(str(checkpoint_path), args.device, num_classes=11)

    # Generate predictions
    generate_predictions(
        model=model,
        test_dir=test_dir,
        output_dir=output_dir,
        resolution=args.resolution,
        conf_threshold=args.conf_threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
    )

    # Create submission ZIP
    method_name = output_dir.name
    create_submission_zip(output_dir, method_name)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Create account at https://macvi.org/")
    print("2. Go to Upload section")
    print("3. Select 'LaRS Panoptic Segmentation' track")
    print(f"4. Upload: {output_dir.parent / method_name}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
