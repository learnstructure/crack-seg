import torch
from torchvision.transforms import v2
from crack_seg.config import IMG_SIZE


def normalize_image(image, mask):
    """Apply standard ImageNet normalization to the image tensor."""
    image = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    return image, mask


# Training transform includes resizing, flips, color jitter, and normalization.
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

# Validation transform uses only resizing and normalization.
val_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_image,
    ]
)

# Test transform mirrors validation preprocessing.
test_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize_image,
    ]
)

# Prediction transform expects only an image and returns a normalized tensor.
pred_transform = v2.Compose(
    [
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
