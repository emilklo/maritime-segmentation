#!/usr/bin/env python3
"""Prepare LaRS dataset for RF-DETR training.

RF-DETR expects:
    dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    │   └── ...
    └── test/
        └── ...

LaRS has:
    lars/
    ├── lars_v1.0.0_annotations/{split}/lars_{split}_instances_all.json
    └── lars_v1.0.0_images/{split}/images/*.jpg

This script creates symlinks to restructure the dataset.
"""

import argparse
import shutil
from pathlib import Path


def prepare_lars(
    lars_root: Path,
    output_dir: Path,
    copy_annotations: bool = False,
) -> None:
    """Create RF-DETR compatible dataset structure from LaRS.

    Args:
        lars_root: Path to LaRS dataset root.
        output_dir: Output directory for restructured dataset.
        copy_annotations: If True, copy annotations instead of symlinking.
    """
    lars_root = Path(lars_root).resolve()
    output_dir = Path(output_dir).resolve()

    # Mapping: (lars_split, rfdetr_split)
    # Note: LaRS test split has no instance annotations, so we use val for both valid and test
    split_configs = [
        ("train", "train"),
        ("val", "valid"),  # RF-DETR uses "valid" not "val"
        ("val", "test"),   # Use val as test since test has no instance annotations
    ]

    for lars_split, rfdetr_split in split_configs:
        split_dir = output_dir / rfdetr_split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Source paths - try instance annotations first
        ann_src = lars_root / f"lars_v1.0.0_annotations/{lars_split}/lars_{lars_split}_instances_all.json"
        images_src = lars_root / f"lars_v1.0.0_images/{lars_split}/images"

        if not ann_src.exists():
            print(f"Warning: No instance annotations for {lars_split} split, skipping")
            continue

        # Annotation file
        ann_dst = split_dir / "_annotations.coco.json"
        if ann_dst.exists() or ann_dst.is_symlink():
            ann_dst.unlink()

        if copy_annotations:
            shutil.copy2(ann_src, ann_dst)
            print(f"Copied annotations: {ann_src} -> {ann_dst}")
        else:
            ann_dst.symlink_to(ann_src)
            print(f"Symlinked annotations: {ann_dst} -> {ann_src}")

        # Symlink all images
        if not images_src.exists():
            print(f"Warning: Images directory not found: {images_src}")
            continue

        image_count = 0
        for img_path in images_src.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dst = split_dir / img_path.name
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(img_path)
                image_count += 1

        print(f"Symlinked {image_count} images for {rfdetr_split} split")

    print(f"\nDataset prepared at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LaRS for RF-DETR")
    parser.add_argument(
        "--lars-root",
        type=str,
        default="data/lars",
        help="Path to LaRS dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lars_rfdetr",
        help="Output directory for RF-DETR format",
    )
    parser.add_argument(
        "--copy-annotations",
        action="store_true",
        help="Copy annotation files instead of symlinking",
    )
    args = parser.parse_args()

    prepare_lars(
        lars_root=Path(args.lars_root),
        output_dir=Path(args.output_dir),
        copy_annotations=args.copy_annotations,
    )


if __name__ == "__main__":
    main()
