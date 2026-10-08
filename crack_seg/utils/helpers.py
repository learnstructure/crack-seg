import importlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from crack_seg.config import DEVICE

def load_model(model_name: str, checkpoint_path: str | Path, device=DEVICE):
    """Load a named model and restore weights from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if model_name.lower() in ("yolo", "yolo_seg"):
        from crack_seg.models.yolo_seg import get_model
        return get_model(str(checkpoint_path))

    model_module = importlib.import_module(f"crack_seg.models.{model_name}")
    model = model_module.get_model().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str,
    device=DEVICE,
):
    """Load a PyTorch checkpoint using an explicitly supplied model name."""
    checkpoint_path = Path(checkpoint_path)
    if not model_name:
        raise ValueError(
            "model_name is required because checkpoint filenames may include dataset names."
        )
    return load_model(model_name, checkpoint_path, device)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], save_path: str | Path | None = None):
    """Draw a training/validation loss curve and save or show the plot."""
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_red_overlay(image, mask, alpha: float = 0.6, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Overlay a binary crack mask in red (or specified color) onto the original RGB image.

    Args:
        image: PIL Image or np.ndarray (RGB).
        mask: 2D numpy array of crack predictions (boolean, 0/1, or float > 0.5).
        alpha: Blending transparency (0.0 = original only, 1.0 = solid color).
        color: RGB tuple for overlay color (default: bright red (255, 0, 0)).

    Returns:
        np.ndarray: RGB image array with crack pixels highlighted.
    """
    img_arr = np.asarray(image).copy()
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    mask_bool = mask > 0.5 if mask.dtype != bool else mask
    overlay_color = np.array(color, dtype=np.uint8)
    img_arr[mask_bool] = (
        img_arr[mask_bool] * (1.0 - alpha) + overlay_color * alpha
    ).astype(np.uint8)
    return img_arr


def predict_yolo_mask(image_path: str | Path, model, conf: float = 0.15) -> np.ndarray:
    """
    Run YOLO segmentation inference on an image and combine all detected instance masks
    into a full-resolution binary mask.

    Args:
        image_path: Path to the input image.
        model: Ultralytics YOLO model instance.
        conf: Confidence threshold for mask detection (default: 0.15).

    Returns:
        np.ndarray: 2D boolean mask of shape (H, W) where True indicates crack pixels.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    results = model.predict(source=img, conf=conf, verbose=False)[0]
    mask = np.zeros((h, w), dtype=np.float32)
    if results.masks is not None:
        for inst_mask in results.masks.data.cpu().numpy():
            inst_img = Image.fromarray((inst_mask > 0.5).astype(np.uint8) * 255).resize(
                (w, h), Image.Resampling.NEAREST
            )
            mask = np.maximum(mask, np.asarray(inst_img) / 255.0)
    return mask > 0.5

