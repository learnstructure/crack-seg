"""
Loss functions for concrete crack segmentation.

Includes:
- DiceLoss: Region-based loss optimizing intersection-over-union.
- FocalLoss: Lin et al. (ICCV 2017) addressing severe foreground/background imbalance.
- DiceFocalLoss: Hybrid loss combining DiceLoss and FocalLoss.
- get_loss: Factory function to instantiate loss functions by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Expects raw logits (before sigmoid) and binary ground truth targets (0 or 1).
    Computes smooth intersection over union to optimize Dice coefficient directly.

    Args:
        smooth (float): Additive smoothing factor to prevent division by zero. Default: 1e-6.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model predictions before sigmoid.
            targets (torch.Tensor): Binary ground truth (0 or 1).
        
        Returns:
            torch.Tensor: Scalar Dice loss value in [0, 1].
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        targets = targets.float()

        preds = torch.sigmoid(logits)

        # Handle both 4D (B, C, H, W) and 3D (B, H, W) tensors
        dims = tuple(range(1, preds.ndim))
        intersection = (preds * targets).sum(dim=dims)
        union = preds.sum(dim=dims) + targets.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for segmentation (Lin et al., ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy well-classified background pixels (e.g. non-cracked concrete)
    and concentrates learning on hard foreground crack boundary pixels.

    Args:
        alpha (float, optional): Balancing factor for the positive class (cracks).
            Typically 0.25 or 0.75. If None, no class weighting is applied. Default: 0.25.
        gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma. Default: 2.0.
        reduction (str): Reduction applied to loss ('mean', 'sum', or 'none'). Default: 'mean'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model outputs before sigmoid.
            targets (torch.Tensor): Binary ground truth (0 or 1).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        targets = targets.float()

        # Numerically stable binary cross-entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability of true class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Alpha weighting factor: alpha for positive, (1 - alpha) for negative
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_loss = alpha_t * modulating_factor * bce_loss
        else:
            focal_loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction: '{self.reduction}'. Expected 'mean', 'sum', or 'none'.")


class DiceFocalLoss(nn.Module):
    """
    Combined loss integrating Dice Loss and Focal Loss:
        Loss = weight_dice * DiceLoss + weight_focal * FocalLoss

    Leverages Dice Loss for global topological overlap and Focal Loss for pixel-level
    class-imbalance mitigation and boundary refinement.

    Args:
        weight_dice (float): Multiplicative weight for Dice Loss. Default: 1.0.
        weight_focal (float): Multiplicative weight for Focal Loss. Default: 1.0.
        alpha (float): Alpha parameter for Focal Loss. Default: 0.25.
        gamma (float): Gamma parameter for Focal Loss. Default: 2.0.
        smooth (float): Smoothing parameter for Dice Loss. Default: 1e-6.
    """

    def __init__(
        self,
        weight_dice: float = 1.0,
        weight_focal: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.weight_dice * dice_loss + self.weight_focal * focal_loss


def get_loss(loss_name: str = "dice", **kwargs) -> nn.Module:
    """
    Factory function to retrieve a loss function by name.

    Supported options for loss_name:
    - 'dice': DiceLoss
    - 'focal': FocalLoss
    - 'dice_focal' or 'focal_dice': DiceFocalLoss
    - 'bce': nn.BCEWithLogitsLoss

    Args:
        loss_name (str): Identifier of the loss function. Default: 'dice'.
        **kwargs: Optional hyperparameter overrides passed to the loss constructor.

    Returns:
        nn.Module: Instantiated PyTorch loss module.
    """
    loss_key = str(loss_name).lower().strip()
    if loss_key == "dice":
        return DiceLoss(**kwargs)
    elif loss_key == "focal":
        return FocalLoss(**kwargs)
    elif loss_key in ("dice_focal", "focal_dice", "combo"):
        return DiceFocalLoss(**kwargs)
    elif loss_key in ("bce", "bce_with_logits"):
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Supported: 'dice', 'focal', 'dice_focal', 'bce'."
        )
