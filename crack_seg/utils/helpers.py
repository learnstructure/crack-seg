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


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path | None = None,
    title: str = "Training and Validation Loss",
):
    """
    Draw a side-by-side training and validation loss curve:
    - Left: Linear Scale (Normal)
    - Right: Logarithmic Scale (Reveals fine gradient dynamics when loss is small, e.g. Focal Loss)

    Optionally saves the figure to save_path and dumps the numeric loss history to a .json file.
    """
    import json
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Normal (Linear) Scale
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=4)
    ax1.plot(epochs, val_losses, label="Validation Loss", marker="o", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Linear Scale (Normal)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")

    # 2. Logarithmic Scale
    pos_train = [max(float(l), 1e-8) for l in train_losses]
    pos_val = [max(float(l), 1e-8) for l in val_losses]
    ax2.plot(epochs, pos_train, label="Train Loss", marker="o", markersize=4)
    ax2.plot(epochs, pos_val, label="Validation Loss", marker="o", markersize=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss (log scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.set_title("Log Scale (Fine Dynamics)", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88, bottom=0.12, wspace=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save raw values to JSON alongside plot
        json_path = save_path.with_suffix(".json")
        try:
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "epochs": epochs,
                        "train_losses": [float(x) for x in train_losses],
                        "val_losses": [float(x) for x in val_losses],
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass
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


def predict_yolo_mask(image_path: str | Path | Image.Image, model, conf: float = 0.15) -> np.ndarray:
    """
    Run YOLO segmentation inference on an image and combine all detected instance masks
    into a full-resolution binary mask.

    Args:
        image_path: Path to the input image or PIL Image object.
        model: Ultralytics YOLO model instance.
        conf: Confidence threshold for mask detection (default: 0.15).

    Returns:
        np.ndarray: 2D boolean mask of shape (H, W) where True indicates crack pixels.
    """
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
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


def warmup_gpu(
    models: list | dict | torch.nn.Module,
    patch_batch_size: int = 16,
    patch_size: int = 448,
    device=DEVICE,
    use_amp: bool = True,
):
    """
    Warm up GPU and CUDA kernels to eliminate one-time cuDNN kernel compilation overhead
    before timing inference.

    Args:
        models: Single PyTorch model, list of models, or dict of models.
        patch_batch_size: Batch size of patches for warmup dummy tensor.
        patch_size: Size of patch dimensions (default: 448).
        device: Target execution device.
        use_amp: Whether AMP is enabled for inference.
    """
    if not torch.cuda.is_available() or "cuda" not in str(device).lower():
        return

    if isinstance(models, dict):
        model_list = [
            m[1] if isinstance(m, tuple) and len(m) == 2 else m
            for m in models.values()
        ]
    elif isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    pytorch_models = [m for m in model_list if isinstance(m, torch.nn.Module)]
    if not pytorch_models:
        return

    print("Warming up GPU and CUDA kernels...")
    _dummy = torch.zeros((patch_batch_size, 3, patch_size, patch_size), device=device)
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_amp):
            for m in pytorch_models:
                _ = m(_dummy)
    torch.cuda.synchronize()


def predict_mask(
    model,
    image: str | Path | Image.Image,
    device=DEVICE,
    model_type: str | None = None,
    use_patches: bool = True,
    patch_size: int = 448,
    stride: int = 224,
    batch_size: int = 16,
    use_amp: bool = True,
    yolo_conf: float = 0.15,
    threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Unified inference helper for PyTorch (UNet, SegFormer, etc.) and YOLO segmentation models.
    Measures wall-clock inference time.

    Args:
        model: Loaded PyTorch model or Ultralytics YOLO model.
        image: Path to image or PIL Image object.
        device: Device for PyTorch inference.
        model_type: Optional explicit type ('unet', 'segformer', 'yolo', etc.).
        use_patches: Whether to use sliding-window patch inference for PyTorch models.
        patch_size: Patch size for patched inference (default: 448).
        stride: Stride for patched inference (default: 224).
        batch_size: GPU batch size for patched inference (default: 16).
        use_amp: Whether to use Automatic Mixed Precision (default: True).
        yolo_conf: Confidence threshold for YOLO (default: 0.15).
        threshold: Probability threshold for binarizing mask (default: 0.5).

    Returns:
        tuple[np.ndarray, float]: (binary_mask of shape (H, W), elapsed_time_seconds)
    """
    import time
    from crack_seg.predict import predict, predict_with_patches

    # Determine model type if not explicitly supplied
    is_yolo = False
    if model_type is not None:
        is_yolo = str(model_type).lower() in ("yolo", "yolo_seg")
    else:
        model_cls_name = model.__class__.__name__
        if "YOLO" in model_cls_name or (
            hasattr(model, "predict") and not isinstance(model, torch.nn.Module)
        ):
            is_yolo = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if is_yolo:
        pred_mask = predict_yolo_mask(image, model, conf=yolo_conf)
    else:
        if use_patches:
            raw_pred = predict_with_patches(
                image_path=image,
                model=model,
                device=device,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                use_amp=use_amp,
            )
        else:
            raw_pred = predict(image, model, device)
        pred_mask = raw_pred > threshold

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - t0

    return pred_mask, elapsed_time


