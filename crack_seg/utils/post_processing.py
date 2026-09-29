import torch
from PIL import Image
from crack_seg.data_handlers.transforms import pred_transform
import cv2
import numpy as np


def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    # Use the prediction-specific transform
    input_tensor = pred_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()
    return pred


def analyze_cracks(prob_mask, threshold=0.5, min_pixels=10):
    """
    Extract crack instances and orientations from a probability mask.

    Args:
        prob_mask (np.ndarray): 2D array of float, values in [0,1].
        threshold (float): Pixels > threshold become crack.
        min_pixels (int): Minimum number of pixels to consider a valid crack.

    Returns:
        list of dict: Each dict contains 'id', 'pixel_count', 'orientation', 'angle_deg'.
    """
    # 1. Binarize using threshold
    binary_mask = (prob_mask > threshold).astype(np.uint8)

    # 2. Morphological cleaning (remove small holes, close small gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_CLOSE, kernel
    )  # Close small gaps
    binary_mask = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel
    )  # Remove small noise

    # 3. Connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    # label 0 is background; cracks are labels 1..num_labels-1

    crack_info = []
    for label_id in range(1, num_labels):
        # Coordinates of pixels belonging to this crack
        pts = np.column_stack(np.where(labels == label_id))
        if len(pts) < min_pixels:
            continue

        # 4. Compute orientation using PCA (principal axis)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        # Covariance matrix
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # Eigenvector with largest eigenvalue = principal direction
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle_rad = np.arctan2(principal_axis[0], principal_axis[1])
        angle_deg = angle_rad * 180 / np.pi

        # Normalize angle to [0, 180) to avoid sign ambiguity
        angle_deg = angle_deg % 180

        # Classify orientation
        if angle_deg <= 20 or angle_deg >= 160:
            orientation = "horizontal"
        elif 70 <= angle_deg <= 110:
            orientation = "vertical"
        else:
            orientation = "inclined"

        crack_info.append(
            {
                "id": label_id,
                "pixel_count": len(pts),
                "orientation": orientation,
                "angle_deg": round(angle_deg, 1).item(),
            }
        )

    return crack_info


def compute_crack_skeleton_and_width(mask: np.ndarray, threshold: int = 128) -> dict:
    """
    Compute Euclidean distance transform and medial axis skeleton to measure crack width in pixels.

    Args:
        mask (np.ndarray): 2D binary or grayscale mask (0-255 or 0.0-1.0).
        threshold (int or float): Binarization threshold.

    Returns:
        dict containing:
            - 'skeleton': 2D binary uint8 array of the crack centerline
            - 'distance_map': 2D float32 array of Euclidean distances to nearest background
            - 'width_map': 2D float32 array of local crack diameters (2 * distance) along the skeleton
            - 'widths': 1D numpy array of width measurements (in pixels) for every centerline point
            - 'mean_width': float (mean width in pixels)
            - 'median_width': float (median width in pixels)
            - 'max_width': float (max width in pixels)
            - 'min_width': float (min width in pixels)
            - 'std_width': float (standard deviation of width in pixels)
            - 'total_crack_pixels': int (total foreground crack pixels)
            - 'skeleton_length_pixels': int (total length of centerline in pixels)
    """
    mask_arr = np.asarray(mask)
    if mask_arr.max() <= 1.0:
        binary_mask = (mask_arr >= (threshold / 255.0 if threshold > 1 else threshold)).astype(np.uint8)
    else:
        binary_mask = (mask_arr >= threshold).astype(np.uint8)

    total_crack_pixels = int(np.sum(binary_mask))
    if total_crack_pixels == 0:
        return {
            "skeleton": np.zeros_like(binary_mask, dtype=np.uint8),
            "distance_map": np.zeros_like(binary_mask, dtype=np.float32),
            "width_map": np.zeros_like(binary_mask, dtype=np.float32),
            "widths": np.array([], dtype=np.float32),
            "mean_width": 0.0,
            "median_width": 0.0,
            "max_width": 0.0,
            "min_width": 0.0,
            "std_width": 0.0,
            "total_crack_pixels": 0,
            "skeleton_length_pixels": 0,
        }

    # 1-pixel medial-axis skeletonization (topological centerline)
    from crack_seg.utils.metrics import _skeletonize
    skel = _skeletonize(binary_mask)

    # Euclidean distance transform (gives radius r of maximal inscribed circle at each pixel)
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Full crack width = 2 * r (diameter in pixels)
    width_map = np.zeros_like(dist_map, dtype=np.float32)
    skel_mask = skel > 0
    width_map[skel_mask] = dist_map[skel_mask] * 2.0
    widths = width_map[skel_mask]

    return {
        "skeleton": skel,
        "distance_map": dist_map,
        "width_map": width_map,
        "widths": widths,
        "mean_width": float(np.mean(widths)) if len(widths) > 0 else 0.0,
        "median_width": float(np.median(widths)) if len(widths) > 0 else 0.0,
        "max_width": float(np.max(widths)) if len(widths) > 0 else 0.0,
        "min_width": float(np.min(widths)) if len(widths) > 0 else 0.0,
        "std_width": float(np.std(widths)) if len(widths) > 0 else 0.0,
        "total_crack_pixels": total_crack_pixels,
        "skeleton_length_pixels": int(len(widths)),
    }
