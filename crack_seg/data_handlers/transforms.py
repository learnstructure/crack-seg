import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from crack_seg.config import IMG_SIZE


def normalize_image(image, mask):
    """Apply standard ImageNet normalization only to the image tensor."""
    if isinstance(image, torch.Tensor):
        image = v2.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return image, mask


# Training transform: resizes both image & mask to IMG_SIZE, applies flips, color jitter to image, and scales/normalizes
train_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_image,
    ]
)

# Validation transform: resizes both image & mask to IMG_SIZE, scales and normalizes image
val_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_image,
    ]
)

# Test transform mirrors validation preprocessing
test_transform = val_transform

# Prediction transform expects only an image and returns a normalized tensor
pred_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Transform for prediction on original image size
original_size_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)