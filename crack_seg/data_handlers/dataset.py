import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
import numpy as np

from crack_seg import config



class CrackDataset(Dataset):
    """Dataset for concrete crack segmentation images and masks.

    Supports initialization via:
    1. A list of (image_path, mask_path) tuples (recommended for multi-dataset workflows).
    2. img_dir and mask_dir paths (for backward compatibility).
    """

    def __init__(
        self,
        img_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        samples: Optional[Sequence[Tuple[Union[str, Path], Union[str, Path]]]] = None,
        transform=None,
        mask_transform=None,
        threshold: int = config.MASK_THRESHOLD,
    ):
        """Initialize the dataset.

        Args:
            img_dir: Directory containing input images (used if samples is None).
            mask_dir: Directory containing corresponding mask images (used if samples is None).
            samples: List of (image_path, mask_path) tuples.
            transform: Optional torchvision transform applied to both image and mask.
            mask_transform: Placeholder for separate mask transforms.
            threshold: Grayscale pixel threshold (0-255) for mask binarization (default 128).
        """
        self.transform = transform
        self.mask_transform = mask_transform
        self.threshold = threshold

        if samples is not None:
            if len(samples) == 0:
                raise ValueError("CrackDataset initialized with an empty 'samples' list.")
            self.samples = [(Path(img), Path(mask)) for img, mask in samples]
        elif img_dir is not None and mask_dir is not None:
            img_dir_path = Path(img_dir)
            mask_dir_path = Path(mask_dir)
            if not img_dir_path.exists():
                raise FileNotFoundError(f"Image directory not found: {img_dir_path}")
            if not mask_dir_path.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir_path}")

            images = sorted(os.listdir(img_dir_path))
            if not images:
                raise ValueError(f"No images found in image directory: {img_dir_path}")

            self.samples = []
            for img_name in images:
                img_p = img_dir_path / img_name
                mask_p = mask_dir_path / img_name
                if not mask_p.exists():
                    raise FileNotFoundError(f"Matching mask not found for image: {img_p} -> {mask_p}")
                self.samples.append((img_p, mask_p))
        else:
            raise ValueError(
                "Either non-empty 'samples' list or both 'img_dir' and 'mask_dir' must be provided."
            )

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load an image and its corresponding mask, then return transformed tensors.

        Args:
            idx: Index of the sample to load.

        Returns:
            image: Tensor representing the input RGB image of shape (3, H, W).
            mask: Tensor representing the binary segmentation mask of shape (1, H, W).
        """
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        # Binarize the grayscale mask with threshold
        mask_bin = (mask_array >= self.threshold).astype(np.float32)

        # Wrap as torchvision v2 tensors so spatial transforms apply to both
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(torch.from_numpy(mask_bin).unsqueeze(0))

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Fallback conversion when no transform is provided
            image = v2.functional.resize(image, config.IMG_SIZE)
            mask = v2.functional.resize(
                mask, config.IMG_SIZE, interpolation=v2.InterpolationMode.NEAREST
            )
            image = image.to(torch.float32) / 255.0

        # Guarantee spatial dimensions match config.IMG_SIZE
        if image.shape[-2:] != config.IMG_SIZE:
            image = v2.functional.resize(image, config.IMG_SIZE)
        if mask.shape[-2:] != config.IMG_SIZE:
            mask = v2.functional.resize(
                mask, config.IMG_SIZE, interpolation=v2.InterpolationMode.NEAREST
            )

        # Return standard float32 tensors with binary mask values {0.0, 1.0}
        image = image.as_subclass(torch.Tensor) if hasattr(image, "as_subclass") else torch.as_tensor(image)
        mask = mask.as_subclass(torch.Tensor) if hasattr(mask, "as_subclass") else torch.as_tensor(mask)
        mask = (mask > 0.5).to(torch.float32)

        return image, mask


