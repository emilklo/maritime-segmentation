#!/usr/bin/env python3
"""Export trained RF-DETR-Seg model to ONNX format for Jetson deployment."""

import argparse
from pathlib import Path

import onnx
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR-Seg to ONNX for TensorRT deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: checkpoint_name.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input resolution (must be divisible by 56)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17 for TensorRT 8.6+)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model with onnxsim",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model with onnxruntime",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export with FP16 weights (smaller file)",
    )
    return parser.parse_args()


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: int = 640,
    opset_version: int = 17,
    fp16: bool = False,
) -> str:
    """Export RF-DETR-Seg checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Output .onnx path
        input_size: Input resolution (H=W)
        opset_version: ONNX opset version
        fp16: Whether to use FP16 weights

    Returns:
        Path to exported ONNX file
    """
    from rfdetr import RFDETRSegPreview

    print(f"Loading checkpoint: {checkpoint_path}")
    model = RFDETRSegPreview()

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        model.model.model.load_state_dict(checkpoint["model"])
    elif "model_ema" in checkpoint:
        # Prefer EMA weights if available
        print("Using EMA weights")
        model.model.model.load_state_dict(checkpoint["model_ema"])
    else:
        model.model.model.load_state_dict(checkpoint)

    model.model.model.eval()

    if fp16:
        model.model.model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=dtype)

    print(f"Exporting to ONNX (opset {opset_version})...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Dtype: {dtype}")

    # Export
    torch.onnx.export(
        model.model.model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["images"],
        output_names=["boxes", "scores", "labels", "masks"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "boxes": {0: "batch_size", 1: "num_detections"},
            "scores": {0: "batch_size", 1: "num_detections"},
            "labels": {0: "batch_size", 1: "num_detections"},
            "masks": {0: "batch_size", 1: "num_detections"},
        },
    )

    print(f"Exported: {output_path}")
    return output_path


def simplify_onnx(onnx_path: str) -> str:
    """Simplify ONNX model using onnxsim."""
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not installed. Run: uv add onnxsim")
        return onnx_path

    print("Simplifying ONNX model...")
    model = onnx.load(onnx_path)
    model_simplified, check = onnxsim.simplify(model)

    if check:
        simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
        onnx.save(model_simplified, simplified_path)
        print(f"Simplified: {simplified_path}")
        return simplified_path
    else:
        print("Simplification failed, using original")
        return onnx_path


def validate_onnx(onnx_path: str, input_size: int = 640) -> bool:
    """Validate ONNX model with onnxruntime."""
    import numpy as np
    import onnxruntime as ort

    print(f"Validating with ONNX Runtime...")

    # Check model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  Model structure: OK")

    # Test inference
    session = ort.InferenceSession(onnx_path)
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    outputs = session.run(None, {"images": dummy_input})
    print(f"  Inference test: OK")
    print(f"  Output shapes: {[o.shape for o in outputs]}")

    return True


def get_model_info(onnx_path: str) -> dict:
    """Get ONNX model size and info."""
    model = onnx.load(onnx_path)
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    # Count parameters
    initializers = model.graph.initializer
    total_params = sum(
        np.prod(init.dims) for init in initializers
    )

    return {
        "file_size_mb": size_mb,
        "parameters": total_params,
        "opset": model.opset_import[0].version,
    }


def main() -> None:
    args = parse_args()

    # Validate input size
    if args.input_size % 56 != 0:
        raise ValueError(f"Input size must be divisible by 56, got {args.input_size}")

    # Set output path
    checkpoint_path = Path(args.checkpoint)
    if args.output:
        output_path = args.output
    else:
        output_path = str(checkpoint_path.with_suffix(".onnx"))

    print("=" * 60)
    print("RF-DETR-Seg ONNX Export")
    print("=" * 60)

    # Export
    exported_path = export_onnx(
        checkpoint_path=str(checkpoint_path),
        output_path=output_path,
        input_size=args.input_size,
        opset_version=args.opset,
        fp16=args.fp16,
    )

    # Simplify if requested
    if args.simplify:
        exported_path = simplify_onnx(exported_path)

    # Validate if requested
    if args.validate:
        validate_onnx(exported_path, args.input_size)

    # Print model info
    import numpy as np  # needed for get_model_info
    info = get_model_info(exported_path)
    print()
    print("=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"  Output: {exported_path}")
    print(f"  File size: {info['file_size_mb']:.1f} MB")
    print(f"  Parameters: {info['parameters']/1e6:.1f}M")
    print(f"  ONNX opset: {info['opset']}")
    print()
    print("Next steps for Jetson deployment:")
    print("  1. Copy ONNX file to Jetson")
    print("  2. Convert to TensorRT:")
    print(f"     trtexec --onnx={Path(exported_path).name} \\")
    print(f"             --saveEngine=rfdetr_seg.engine \\")
    print(f"             --fp16")
    print("=" * 60)


if __name__ == "__main__":
    main()
