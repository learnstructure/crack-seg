import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path



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
