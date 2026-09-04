import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union, Optional, Sequence
import numpy as np
from PIL import Image
from tqdm import tqdm

from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import (
    get_train_val_test_datasets,
)


def mask_to_yolo_polygons(
    mask_array: np.ndarray,
    threshold: int = 128,
    min_points: int = 3,
    min_area: float = 4.0
) -> List[List[float]]:
    """
    Convert a 2D grayscale/binary mask into a list of normalized polygon coordinates for YOLO-seg.
    
    Args:
        mask_array: 2D numpy array (H, W).
        threshold: Binarization threshold (0-255).
        min_points: Minimum number of vertices for a valid polygon.
        min_area: Minimum contour area in pixels to keep.
        
    Returns:
        List of polygon coordinate lists: [[x1, y1, x2, y2, ...], ...] normalized to [0, 1].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for YOLO segmentation polygon export. "
            "Please install it with 'pip install opencv-python'."
        )

    h, w = mask_array.shape[:2]
    binary_mask = (mask_array >= threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons: List[List[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Approximate contour to reduce points while preserving shape
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2)

        if len(pts) < min_points:
            continue

        # Normalize coordinates [0, 1]
        norm_pts = []
        for x, y in pts:
            norm_pts.append(float(np.clip(x / w, 0.0, 1.0)))
            norm_pts.append(float(np.clip(y / h, 0.0, 1.0)))

        polygons.append(norm_pts)

    return polygons


def export_split_to_yolo(
    samples: Sequence[Tuple[Union[str, Path], Union[str, Path]]],
    split_name: str,
    output_dir: Path,
    class_id: int = 0,
    threshold: int = config.MASK_THRESHOLD,
    copy_images: bool = True
) -> int:
    """
    Export a single split (train, val, or test) to YOLO segmentation format.
    """
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for idx, (img_path, mask_path) in enumerate(
        tqdm(samples, desc=f"Exporting {split_name} split to YOLO")
    ):
        img_p = Path(img_path)
        mask_p = Path(mask_path)

        # Unique name prefix in case different datasets have overlapping filenames
        unique_stem = f"{img_p.parent.parent.stem}_{img_p.stem}"
        dest_img_path = images_dir / f"{unique_stem}{img_p.suffix}"
        dest_label_path = labels_dir / f"{unique_stem}.txt"

        # Copy or symlink image
        if copy_images:
            if not dest_img_path.exists():
                shutil.copy2(img_p, dest_img_path)
        else:
            if not dest_img_path.exists():
                os.link(str(img_p), str(dest_img_path))

        # Convert mask to polygons
        try:
            with Image.open(mask_p).convert("L") as mask_img:
                mask_arr = np.array(mask_img)
            polygons = mask_to_yolo_polygons(mask_arr, threshold=threshold)
        except Exception as e:
            print(f"Warning: Failed to process mask {mask_p}: {e}")
            polygons = []

        # Write label txt file
        with open(dest_label_path, "w", encoding="utf-8") as f:
            for poly in polygons:
                poly_str = " ".join(f"{coord:.6f}" for coord in poly)
                f.write(f"{class_id} {poly_str}\n")

        exported_count += 1

    return exported_count


def export_dataset_to_yolo(
    dataset_names: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    threshold: int = config.MASK_THRESHOLD
) -> Path:
    """
    Exports the combined multi-dataset into Ultralytics YOLO segmentation format and generates 'crack_data.yaml'.
    
    Returns:
        Path to the generated YAML configuration file.
    """
    dataset_names = dataset_names or config.DATASETS
    output_dir = Path(output_dir or config.YOLO_DATASET_DIR)
    data_root = Path(data_root or config.DATA_ROOT)
    train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
    val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
    test_ratio = test_ratio if test_ratio is not None else config.TEST_RATIO
    seed = seed if seed is not None else config.SPLIT_SEED

    print(f"Loading datasets {dataset_names} from {data_root}...")
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        dataset_names=dataset_names,
        data_root=data_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratified=config.STRATIFIED_SPLIT,
    )
    train_samples = train_dataset.samples
    val_samples = val_dataset.samples
    test_samples = test_dataset.samples

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        for subdirectory in ("images", "labels"):
            split_dir = output_dir / subdirectory / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    print(f"\nExporting YOLO segmentation dataset to: {output_dir.resolve()}")
    n_train = export_split_to_yolo(train_samples, "train", output_dir, threshold=threshold)
    n_val = export_split_to_yolo(val_samples, "val", output_dir, threshold=threshold)
    n_test = export_split_to_yolo(test_samples, "test", output_dir, threshold=threshold)

    # Generate YOLO dataset YAML
    yaml_path = output_dir / "crack_data.yaml"
    # Format absolute path with forward slashes for cross-platform YOLO compatibility
    abs_output_path = str(output_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# Ultralytics YOLO Segmentation Dataset Configuration
path: {abs_output_path}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: crack
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n--- YOLO Dataset Export Complete ---")
    print(f"  Train:      {n_train} samples")
    print(f"  Validation: {n_val} samples")
    print(f"  Test:       {n_test} samples")
    print(f"  YAML Config: {yaml_path.resolve()}\n")

    return yaml_path


if __name__ == "__main__":
    export_dataset_to_yolo()
