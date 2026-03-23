
import torch


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
        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1. - dice_score
