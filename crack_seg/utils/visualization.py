import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from crack_seg.utils.helpers import create_red_overlay



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


def display_prediction(image_path, model, device, transform, mask_dir=None, threshold=0.5):
    """
    Display original image, ground truth mask, and predicted mask side by side.
    """
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Infer mask path and load ground truth mask
    img_path = Path(image_path)
    if mask_dir is None:
        mask_dir = img_path.parent.parent / "masks"
    mask_path = Path(mask_dir) / img_path.name

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
    else:
        print(f"Mask not found at {mask_path}, showing blank.")
        mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))

    # Store original image and mask for display
    original_image_for_display = np.array(image)
    original_mask_for_display = np.array(mask) / 255.0

    # Apply transform to both image and mask for the model
    transformed_image, _ = transform(image, mask)

    # Prepare tensor for model
    input_tensor = transformed_image.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().squeeze()
        pred_bin = (pred > threshold).float()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].imshow(original_image_for_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_mask_for_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (threshold={threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def compare_prediction(image_path, models, device, transform, mask_dir=None, threshold=0.5):
    """
    Display the original image, ground truth mask, and predicted masks from multiple models.

    Args:
        image_path: Path to the image file.
        models: Dictionary of model name -> PyTorch model.
        device: Device for model inference.
        transform: Function that applies preprocessing to image and mask.
        mask_dir: Optional directory containing ground truth masks.
        threshold: Threshold for binarizing sigmoid outputs.
    """
    if not isinstance(models, dict):
        raise TypeError("models must be a dict mapping model names to models")

    model_items = list(models.items())
    num_models = len(model_items)
    if num_models == 0:
        raise ValueError("models dict must contain at least one model")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Infer mask path and load ground truth mask
    img_path = Path(image_path)
    if mask_dir is None:
        mask_dir = img_path.parent.parent / "masks"
    mask_path = Path(mask_dir) / img_path.name

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
    else:
        print(f"Mask not found at {mask_path}, showing blank.")
        mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))

    original_image_for_display = np.array(image)
    original_mask_for_display = np.array(mask) / 255.0

    transformed_image, _ = transform(image, mask)
    input_tensor = transformed_image.unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for name, model in model_items:
            model.eval()
            output = model(input_tensor)
            pred = torch.sigmoid(output).cpu().squeeze()
            predictions.append((name, (pred > threshold).float()))

    ncols = 2 + num_models
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(original_image_for_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_mask_for_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    for idx, (name, pred_bin) in enumerate(predictions, start=2):
        axes[idx].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(f"{name}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def plot_overlay_comparison(
    image: str | Path | Image.Image | np.ndarray,
    overlays: dict[str, np.ndarray | Image.Image],
    original_title: str = "Original Bridge Image",
    suptitle: str | None = None,
    figsize: tuple[int, int] | None = None,
    alpha: float = 0.6,
    color: tuple[int, int, int] = (255, 0, 0),
    show: bool = True,
    save_path: str | Path | None = None,
    roi_mask: np.ndarray | None = None,
    roi_color: tuple[int, int, int] = (0, 255, 255),
    outline_roi_on_image: bool = False,
):
    """
    Display the original image and multiple model overlays side-by-side in a single row.

    Args:
        image: Path to image, PIL Image, or RGB numpy array.
        overlays: Dict mapping column title (e.g. 'UNet Overlay (T=1.85s)') to
                  either a pre-computed RGB overlay or a 2D binary mask.
                  If a 2D mask is provided, create_red_overlay is automatically applied.
        original_title: Title for the first subplot (default: 'Original Bridge Image').
        suptitle: Main figure title (e.g. image filename).
        figsize: Optional (width, height) tuple for matplotlib figure.
        alpha: Overlay alpha if mask is converted to overlay (default: 0.6).
        color: RGB color tuple for crack overlay (default: red (255, 0, 0)).
        show: Whether to call plt.show() (default: True).
        save_path: Optional path to save the figure image.
        roi_mask: Optional boolean or binary mask to outline on masks (or on image if outline_roi_on_image is True).
        roi_color: RGB tuple for ROI outline contour (default: cyan (0, 255, 255)).
        outline_roi_on_image: Whether to outline ROI contour on the original image (default: False).
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(image, dtype=np.uint8))

    display_img = img
    # Outline boundary on the base image only if explicitly requested
    if outline_roi_on_image and roi_mask is not None:
        import cv2
        roi_bin = (np.asarray(roi_mask) > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_arr = np.array(img).copy()
        line_w = max(2, max(img.size) // 500)
        cv2.drawContours(img_arr, contours, -1, roi_color, line_w)
        display_img = Image.fromarray(img_arr)

    ncols = 1 + len(overlays)
    if figsize is None:
        figsize = (6 * ncols, 6)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    # Col 0: Original image
    axes[0].imshow(display_img)
    axes[0].set_title(original_title, fontsize=14)
    axes[0].axis("off")

    # Col 1..N: Model Overlays / Masks
    for ax, (title, overlay_or_mask) in zip(axes[1:], overlays.items()):
        overlay_arr = np.asarray(overlay_or_mask)
        if overlay_arr.ndim == 2:
            if "mask" in title.lower() and "overlay" not in title.lower():
                # Display as high-contrast binary/grayscale mask
                mask_display = (overlay_arr > 0).astype(np.uint8) * 255
                if roi_mask is not None:
                    import cv2
                    roi_bin = (np.asarray(roi_mask) > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    mask_rgb = np.stack([mask_display] * 3, axis=-1)
                    line_w = max(2, max(img.size) // 500)
                    cv2.drawContours(mask_rgb, contours, -1, roi_color, line_w)
                    ax.imshow(mask_rgb)
                else:
                    ax.imshow(mask_display, cmap="gray", vmin=0, vmax=255)
            else:
                overlay_arr = create_red_overlay(img, overlay_arr, alpha=alpha, color=color)
                ax.imshow(overlay_arr)
        else:
            ax.imshow(overlay_arr)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    return fig, axes


def visualize_model_overlays(
    image: str | Path | Image.Image,
    models: dict,
    device=None,
    patch_size: int = 448,
    stride: int = 224,
    batch_size: int = 16,
    use_amp: bool = True,
    yolo_conf: float = 0.15,
    alpha: float = 0.6,
    color: tuple[int, int, int] = (255, 0, 0),
    original_title: str = "Original Bridge Image",
    suptitle: str | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    roi_mask: np.ndarray | None = None,
    roi_color: tuple[int, int, int] = (0, 255, 255),
    ground_truth_mask: np.ndarray | Image.Image | None = None,
    gt_title: str = "Ground Truth Mask",
    outline_roi_on_image: bool = False,
) -> dict:
    """
    Run inference across multiple models (UNet, SegFormer, YOLO) on an image
    using patched inference for PyTorch models and full-resolution YOLO inference,
    compute wall-clock timing, create red overlays, and plot them side by side.
    Optionally restrict predictions to an ROI mask (e.g. Column) and include Ground Truth Mask.

    Args:
        image: Image file path or PIL Image.
        models: Dict mapping display name (e.g. 'UNet', 'SegFormer (Dice+Focal)', 'YOLO-seg')
                to model object, or tuple (model_type, model_object).
        device: Device for PyTorch inference (defaults to config.DEVICE).
        patch_size: Sliding window patch size (default: 448).
        stride: Stride between patches (default: 224).
        batch_size: GPU batch size for patches (default: 16).
        use_amp: Automatic Mixed Precision (default: True).
        yolo_conf: YOLO confidence threshold (default: 0.15).
        alpha: Overlay blending opacity (default: 0.6).
        color: Overlay RGB color (default: (255, 0, 0)).
        original_title: Title for original image column.
        suptitle: Main title for figure (defaults to image filename if path is given).
        figsize: Matplotlib figure size tuple.
        save_path: Optional path to save the generated figure.
        show: Whether to display figure with plt.show().
        roi_mask: Optional boolean mask of shape (H, W) to isolate foreground ROI.
        roi_color: RGB outline color for ROI boundary.
        ground_truth_mask: Optional ground truth mask array or PIL Image.
        gt_title: Column title for ground truth mask (default: 'Ground Truth Mask').
        outline_roi_on_image: Whether to outline ROI contour on original bridge image (default: False).

    Returns:
        dict: {'predictions': dict, 'timings': dict, 'overlays': dict, 'fig': fig, 'axes': axes}
    """
    from crack_seg.utils.helpers import predict_mask

    if device is None:
        from crack_seg.config import DEVICE
        device = DEVICE

    if isinstance(image, (str, Path)):
        img_path = Path(image)
        bridge_img = Image.open(img_path).convert("RGB")
        if suptitle is None:
            suptitle = img_path.name
    else:
        bridge_img = image.convert("RGB")

    predictions = {}
    timings = {}
    overlays = {}

    roi_arr = (np.asarray(roi_mask) > 0.5) if roi_mask is not None else None

    # Add ground truth mask as first comparison panel if provided
    if ground_truth_mask is not None:
        gt_arr = np.asarray(ground_truth_mask) > 0
        if roi_arr is not None:
            gt_arr = gt_arr & roi_arr
        overlays[gt_title] = gt_arr

    for name, model_spec in models.items():
        if isinstance(model_spec, tuple) and len(model_spec) == 2:
            model_type, model = model_spec
        else:
            model_type, model = None, model_spec

        pred_mask, elapsed = predict_mask(
            model=model,
            image=image,
            device=device,
            model_type=model_type,
            use_patches=True,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            use_amp=use_amp,
            yolo_conf=yolo_conf,
        )

        # Restrict prediction to ROI mask if provided
        if roi_arr is not None:
            pred_mask = pred_mask & roi_arr

        overlay = create_red_overlay(bridge_img, pred_mask, alpha=alpha, color=color)

        predictions[name] = pred_mask
        timings[name] = elapsed
        overlays[f"{name} Overlay (T={elapsed:.2f}s)"] = overlay

    fig, axes = plot_overlay_comparison(
        image=bridge_img,
        overlays=overlays,
        original_title=original_title,
        suptitle=suptitle,
        figsize=figsize,
        alpha=alpha,
        color=color,
        show=show,
        save_path=save_path,
        roi_mask=roi_arr,
        roi_color=roi_color,
        outline_roi_on_image=outline_roi_on_image,
    )

    return {
        "predictions": predictions,
        "timings": timings,
        "overlays": overlays,
        "fig": fig,
        "axes": axes,
    }



