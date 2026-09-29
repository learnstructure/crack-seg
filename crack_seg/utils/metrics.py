
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import cv2


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Jaccard Index) for binary segmentation."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient (also known as F1-score for segmentation)."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def _get_stats(pred, target, smooth=1e-6):
    """Helper function to get TP, FP, FN, TN."""
    pred_bin = (pred > 0.5).float()
    target_bin = target.float()

    tp = (pred_bin * target_bin).sum()
    fp = pred_bin.sum() - tp
    fn = target_bin.sum() - tp
    tn = target.numel() - tp - fp - fn
    return tp, fp, fn, tn, smooth


def pixel_accuracy(pred, target):
    """Calculates pixel-wise accuracy."""
    tp, fp, fn, tn, _ = _get_stats(pred, target)
    return (tp + tn) / (tp + tn + fp + fn + 1e-6)


def precision_score(pred, target):
    """Calculates precision."""
    tp, fp, _, _, smooth = _get_stats(pred, target)
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target):
    """Calculates recall (sensitivity)."""
    tp, _, fn, _, smooth = _get_stats(pred, target)
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target):
    """Calculates specificity."""
    _, fp, _, tn, smooth = _get_stats(pred, target)
    return (tn + smooth) / (tn + fp + smooth)


# ==============================================================================
# Centerline & Skeletonization Utilities
# ==============================================================================

def _to_2d_binary(mask, threshold: float = 0.5) -> np.ndarray:
    """Helper to convert torch tensors or arrays to a clean 2D uint8 binary array."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    arr = np.squeeze(mask).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array after squeezing, got shape {arr.shape}"
        )
    if arr.max() > 1.0:
        arr /= 255.0
    return (arr >= threshold).astype(np.uint8)


def _skeletonize(binary_mask: np.ndarray, method: str = None) -> np.ndarray:
    """
    Skeletonize a binary mask to a 1-pixel-wide centerline using scikit-image.

    Args:
        binary_mask (np.ndarray): 2-D binary mask (0 or 1 / True or False).
            Leading singleton dimensions (e.g. shape (1, H, W)) are squeezed automatically.
        method (str, optional): Thinning algorithm. If None, reads config.SKELETON_METHOD (defaults to 'lee').
            Options:
            - 'lee' (default): Lee's medial axis algorithm (preserves 8-connectivity and true topology).
            - 'zhang': Zhang-Suen thinning algorithm.

    Returns:
        np.ndarray: 2-D uint8 array with 1 for centerline and 0 for background.
    """
    if method is None:
        try:
            from crack_seg import config
            method = getattr(config, "SKELETON_METHOD", "lee")
        except (ImportError, AttributeError):
            method = "lee"

    mask_2d = np.squeeze(binary_mask)
    if mask_2d.ndim != 2:
        raise ValueError(
            f"_skeletonize expects a 2-D array after squeezing, got shape {mask_2d.shape}"
        )
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8)

    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(mask_u8.astype(bool), method=method).astype(np.uint8)
    except ImportError as e:
        raise ImportError(
            "scikit-image is required for crack skeletonization. "
            "Please install it via: pip install scikit-image"
        ) from e


def crack_length_in_pixels(mask, threshold: float = 0.5, method: str = None) -> int:
    """
    Calculate the total length of cracks in pixels by computing
    the 1-pixel-wide medial-axis skeleton.
    """
    bin_mask = _to_2d_binary(mask, threshold)
    skel = _skeletonize(bin_mask, method=method)
    return int(skel.sum())


def skeleton_iou_score(
    pred_mask,
    gt_mask,
    threshold: float = 0.5,
    tolerance_px: int = 3,
    smooth: float = 1e-6,
    method: str = None,
) -> float:
    """
    Skeleton-based IoU metric with optional tolerance dilation.

    Args:
        pred_mask: Predicted mask (Tensor or ndarray).
        gt_mask: Ground-truth mask (Tensor or ndarray).
        threshold (float): Binarization threshold. Default 0.5.
        tolerance_px (int): Dilation radius in pixels applied to each skeleton.
        smooth (float): Additive smoothing to avoid 0/0. Default 1e-6.
        method (str, optional): Skeletonization algorithm ('lee' or 'zhang').

    Returns:
        float: Skeleton IoU in [0, 1].
    """
    pred_bin = _to_2d_binary(pred_mask, threshold)
    gt_bin = _to_2d_binary(gt_mask, threshold)

    pred_skel = _skeletonize(pred_bin, method=method)
    gt_skel = _skeletonize(gt_bin, method=method)

    len_pred = pred_skel.sum()
    len_gt = gt_skel.sum()

    # Empty mask edge cases
    if len_pred == 0 and len_gt == 0:
        return 1.0
    if len_pred == 0 or len_gt == 0:
        return 0.0

    # Optional tolerance dilation
    if tolerance_px > 0:
        kernel_size = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        pred_skel = cv2.dilate(pred_skel, kernel)
        gt_skel = cv2.dilate(gt_skel, kernel)

    intersection = np.logical_and(pred_skel, gt_skel).sum()
    union = np.logical_or(pred_skel, gt_skel).sum()

    return float((intersection + smooth) / (union + smooth))


# ==============================================================================
# New Width-Independent Metrics: Centerline Buffer, clDice & Orientation
# ==============================================================================

def centerline_buffer_metrics(
    pred_mask,
    gt_mask,
    threshold: float = 0.5,
    tolerance_px: float = None,
    method: str = None,
) -> dict:
    """
    Computes crack lengths (in pixels), length of predicted cracks within the
    tolerance bound around true cracks, length of true cracks covered, and the
    resulting Centerline Precision, Recall, F1, and IoU.

    Args:
        pred_mask: Predicted mask (torch.Tensor or np.ndarray).
        gt_mask: Ground-truth mask (torch.Tensor or np.ndarray).
        threshold (float): Probability threshold for binarization.
        tolerance_px (float, optional): Euclidean distance tolerance in pixels.
            If None, reads config.CENTERLINE_TOLERANCE_PX (defaults to 20.0).
        method (str, optional): Skeletonization algorithm ('lee' or 'zhang').

    Returns:
        dict:
            - 'gt_length_px': int, true crack skeleton length in pixels
            - 'pred_length_px': int, predicted crack skeleton length in pixels
            - 'pred_len_in_bound_px': int, predicted length within tolerance_px of GT skeleton
            - 'gt_len_covered_px': int, GT length covered within tolerance_px of Pred skeleton
            - 'precision': float, fraction of predicted length within tolerance
            - 'recall': float, fraction of GT length covered
            - 'f1_score': float, harmonic mean of centerline precision & recall
            - 'iou': float, length-based Jaccard index
    """
    if tolerance_px is None:
        try:
            from crack_seg import config
            tolerance_px = getattr(config, "CENTERLINE_TOLERANCE_PX", 20.0)
        except (ImportError, AttributeError):
            tolerance_px = 20.0

    pred_bin = _to_2d_binary(pred_mask, threshold)
    gt_bin = _to_2d_binary(gt_mask, threshold)

    pred_skel = _skeletonize(pred_bin, method=method)
    gt_skel = _skeletonize(gt_bin, method=method)

    len_pred = int(pred_skel.sum())
    len_gt = int(gt_skel.sum())

    # Empty cases
    if len_pred == 0 and len_gt == 0:
        return {
            "gt_length_px": 0,
            "pred_length_px": 0,
            "pred_len_in_bound_px": 0,
            "gt_len_covered_px": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "iou": 1.0,
        }
    if len_pred == 0 or len_gt == 0:
        return {
            "gt_length_px": len_gt,
            "pred_length_px": len_pred,
            "pred_len_in_bound_px": 0,
            "gt_len_covered_px": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "iou": 0.0,
        }

    # Euclidean distance transforms
    dist_to_gt = cv2.distanceTransform(1 - gt_skel, cv2.DIST_L2, 5)
    bound_gt = (dist_to_gt <= tolerance_px).astype(np.uint8)

    dist_to_pred = cv2.distanceTransform(1 - pred_skel, cv2.DIST_L2, 5)
    bound_pred = (dist_to_pred <= tolerance_px).astype(np.uint8)

    len_pred_in_bound = int(np.logical_and(pred_skel, bound_gt).sum())
    len_gt_covered = int(np.logical_and(gt_skel, bound_pred).sum())

    prec = len_pred_in_bound / len_pred
    rec = len_gt_covered / len_gt
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    inter_len = min(len_pred_in_bound, len_gt_covered)
    union_len = len_pred + len_gt - inter_len
    iou = inter_len / union_len if union_len > 0 else 0.0

    return {
        "gt_length_px": len_gt,
        "pred_length_px": len_pred,
        "pred_len_in_bound_px": len_pred_in_bound,
        "gt_len_covered_px": len_gt_covered,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "iou": float(iou),
    }


def centerline_f1(pred_mask, gt_mask, threshold: float = 0.5, tolerance_px: float = None, method: str = None) -> float:
    """Convenience wrapper returning Centerline F1 score at given tolerance."""
    return centerline_buffer_metrics(pred_mask, gt_mask, threshold, tolerance_px, method=method)["f1_score"]


def centerline_iou(pred_mask, gt_mask, threshold: float = 0.5, tolerance_px: float = None, method: str = None) -> float:
    """Convenience wrapper returning Centerline IoU (length Jaccard) at given tolerance."""
    return centerline_buffer_metrics(pred_mask, gt_mask, threshold, tolerance_px, method=method)["iou"]


def centerline_precision(pred_mask, gt_mask, threshold: float = 0.5, tolerance_px: float = None, method: str = None) -> float:
    """Convenience wrapper returning Centerline Precision at given tolerance."""
    return centerline_buffer_metrics(pred_mask, gt_mask, threshold, tolerance_px, method=method)["precision"]


def centerline_recall(pred_mask, gt_mask, threshold: float = 0.5, tolerance_px: float = None, method: str = None) -> float:
    """Convenience wrapper returning Centerline Recall at given tolerance."""
    return centerline_buffer_metrics(pred_mask, gt_mask, threshold, tolerance_px, method=method)["recall"]


def cldice_score(pred_mask, gt_mask, threshold: float = 0.5, smooth: float = 1e-6, method: str = None) -> float:
    """
    Centerline Dice (clDice) for tubular and elongated structures (Shit et al., CVPR 2021).
    Evaluates topological connectivity and centerline presence inside mask volumes.
    """
    pred_bin = _to_2d_binary(pred_mask, threshold)
    gt_bin = _to_2d_binary(gt_mask, threshold)

    pred_skel = _skeletonize(pred_bin, method=method)
    gt_skel = _skeletonize(gt_bin, method=method)

    len_pred = pred_skel.sum()
    len_gt = gt_skel.sum()

    if len_pred == 0 and len_gt == 0:
        return 1.0
    if len_pred == 0 or len_gt == 0:
        return 0.0

    tprec = (pred_skel * gt_bin).sum() / (len_pred + smooth)
    tsens = (gt_skel * pred_bin).sum() / (len_gt + smooth)

    if tprec + tsens == 0:
        return 0.0
    return float((2.0 * tprec * tsens) / (tprec + tsens))


def crack_orientation_metrics(pred_mask, gt_mask, threshold: float = 0.5, min_pixels: int = 10) -> dict:
    """
    Calculates Mean Angular Error (MAE in degrees) and Orientation Classification Agreement
    between predicted cracks and ground truth cracks.
    """
    def _extract_cracks(binary_mask):
        num_labels, labels = cv2.connectedComponents((binary_mask > 0).astype(np.uint8), connectivity=8)
        cracks = []
        for label_id in range(1, num_labels):
            pts = np.column_stack(np.where(labels == label_id))
            if len(pts) < min_pixels:
                continue
            mean = np.mean(pts, axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            if cov.ndim < 2 or np.isnan(cov).any():
                continue
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            principal_axis = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
            angle_rad = np.arctan2(principal_axis[0], principal_axis[1])
            angle_deg = float((angle_rad * 180.0 / np.pi) % 180.0)

            if angle_deg <= 20.0 or angle_deg >= 160.0:
                cat = "horizontal"
            elif 70.0 <= angle_deg <= 110.0:
                cat = "vertical"
            else:
                cat = "inclined"

            cracks.append({
                "label": label_id,
                "center": mean,
                "angle_deg": angle_deg,
                "category": cat,
            })
        return cracks

    pred_bin = _to_2d_binary(pred_mask, threshold)
    gt_bin = _to_2d_binary(gt_mask, threshold)

    pred_cracks = _extract_cracks(pred_bin)
    gt_cracks = _extract_cracks(gt_bin)

    if len(pred_cracks) == 0 and len(gt_cracks) == 0:
        return {"mae_deg": 0.0, "category_agreement": 1.0, "n_matched": 0}
    if len(pred_cracks) == 0 or len(gt_cracks) == 0:
        return {"mae_deg": 90.0, "category_agreement": 0.0, "n_matched": 0}

    angular_errors = []
    category_matches = []

    for p_crack in pred_cracks:
        p_cen = p_crack["center"]
        best_dist = float("inf")
        best_gt = None
        for g_crack in gt_cracks:
            dist = np.linalg.norm(p_cen - g_crack["center"])
            if dist < best_dist:
                best_dist = dist
                best_gt = g_crack
        if best_gt is not None:
            diff = abs(p_crack["angle_deg"] - best_gt["angle_deg"])
            err = min(diff, 180.0 - diff)
            angular_errors.append(err)
            category_matches.append(1.0 if p_crack["category"] == best_gt["category"] else 0.0)

    mae = float(np.mean(angular_errors)) if angular_errors else 0.0
    agreement = float(np.mean(category_matches)) if category_matches else 0.0
    return {
        "mae_deg": mae,
        "category_agreement": agreement,
        "n_matched": len(angular_errors),
    }


def crack_orientation_mae(pred_mask, gt_mask, threshold: float = 0.5, min_pixels: int = 10) -> float:
    """Convenience wrapper returning Mean Angular Error (MAE in degrees)."""
    return crack_orientation_metrics(pred_mask, gt_mask, threshold, min_pixels)["mae_deg"]


# --- Loss Functions ---
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculates Dice loss.
        Expects logits (before sigmoid) and binary targets.
        """
        preds = torch.sigmoid(logits)
        dims = (1, 2, 3)  # batch-wise computation
        intersection = (preds * targets).sum(dim=dims)
        dice_score = (2. * intersection + self.smooth) / (
            preds.sum(dim=dims) + targets.sum(dim=dims) + self.smooth
        )
        return 1. - dice_score.mean()
