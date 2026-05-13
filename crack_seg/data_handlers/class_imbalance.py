import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image

from crack_seg.config import TRAIN_MASK_DIR

VALID_MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _get_mask_paths(mask_dir: Path) -> Iterable[Path]:
    mask_dir = Path(mask_dir)
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_paths = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_MASK_EXTENSIONS
    )

    if not mask_paths:
        raise ValueError(f"No valid mask files found in: {mask_dir}")

    return mask_paths


def compute_crack_pixel_proportion(
    mask_dir: Optional[Path] = None,
    threshold: int = 128,
) -> Dict[str, float]:
    """Compute crack pixel proportion for a set of binary masks.

    Args:
        mask_dir: Directory containing grayscale mask files.
        threshold: Threshold for converting grayscale mask values to binary.

    Returns:
        A dictionary with total pixel counts and crack/background proportions.
    """
    mask_dir = Path(mask_dir or TRAIN_MASK_DIR)
    mask_paths = list(_get_mask_paths(mask_dir))

    total_crack_pixels = 0
    total_pixels = 0
    total_images = 0

    for mask_path in mask_paths:
        with Image.open(mask_path).convert("L") as mask_image:
            mask_array = np.array(mask_image, dtype=np.uint8)

        binary_mask = mask_array >= threshold
        total_crack_pixels += int(binary_mask.sum())
        total_pixels += mask_array.size
        total_images += 1

    crack_ratio = float(total_crack_pixels) / total_pixels if total_pixels else 0.0
    background_ratio = 1.0 - crack_ratio

    return {
        "mask_dir": str(mask_dir),
        "image_count": float(total_images),
        "total_pixels": float(total_pixels),
        "crack_pixels": float(total_crack_pixels),
        "background_pixels": float(total_pixels - total_crack_pixels),
        "crack_ratio": crack_ratio,
        "background_ratio": background_ratio,
    }


def format_class_imbalance_report(result: Dict[str, float]) -> str:
    return (
        f"Class imbalance report for masks in: {result['mask_dir']}\n"
        f"Images processed: {int(result['image_count'])}\n"
        f"Total pixels: {int(result['total_pixels']):,}\n"
        f"Crack pixels: {int(result['crack_pixels']):,} ({result['crack_ratio']:.6f})\n"
        f"Background pixels: {int(result['background_pixels']):,} ({result['background_ratio']:.6f})\n"
    )


def report_class_imbalance(
    mask_dir: Optional[Path] = None, threshold: int = 128
) -> None:
    result = compute_crack_pixel_proportion(mask_dir=mask_dir, threshold=threshold)
    print(format_class_imbalance_report(result))


if __name__ == "__main__":
    report_class_imbalance()
