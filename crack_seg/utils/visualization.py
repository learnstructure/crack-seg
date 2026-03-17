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

    Args:
        image_path (str or Path): Path to input image.
        model (torch.nn.Module): Trained PyTorch model (in eval mode).
        device (torch.device): Device to run inference on.
        transform (callable): Preprocessing transform (should match validation transform).
        mask_dir (str or Path, optional): Directory containing masks. If None, assume
            mask is in a 'masks' folder adjacent to the image's folder, or in same folder
            with same name. Adjust logic as needed.
        threshold (float): Threshold for binarizing prediction (default 0.5).
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Infer mask path
    img_path = Path(image_path)
    if mask_dir is None:
        # Assume mask is in a 'masks' folder at same level as image's folder
        # e.g., image in .../Train/images/img.png → mask in .../Train/masks/img.png
        mask_dir = img_path.parent.parent / "masks"
    mask_path = Path(mask_dir) / img_path.name
    
    # Load ground truth mask
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")  # Grayscale
        # Resize mask to match model input size if needed
        mask = mask.resize(transform.transforms[0].size if hasattr(transform, 'transforms') else (256,256))
        mask_np = np.array(mask) / 255.0  # Normalize to [0,1]
    else:
        print(f"Mask not found at {mask_path}, showing blank.")
        mask_np = np.zeros((256,256))  # Placeholder
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()  # Remove batch & channel dims
        pred_bin = (pred > threshold).astype(np.float32)
    
    # Denormalize image for display
    # Extract mean/std from transform if possible; otherwise use ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_np = input_tensor.squeeze(0).cpu().numpy().transpose(1,2,0)  # CHW -> HWC
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (threshold={threshold})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()