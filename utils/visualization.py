import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for display."""
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def save_prediction(image, mask, pred, filename, threshold=0.5):
    """Save image, ground truth, and prediction side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Image
    img = denormalize(image).permute(1, 2, 0).numpy()
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")
    # Ground truth
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    # Prediction
    pred_bin = (pred > threshold).float()
    axes[2].imshow(pred_bin.squeeze(), cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
