import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
import numpy as np

class CrackDataset(Dataset):
    """Dataset for crack segmentation images and masks.

    The dataset assumes that the image directory and mask directory contain
    matching filenames for each example.
    """

    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        """Initialize the dataset.

        Args:
            img_dir: Directory containing input images.
            mask_dir: Directory containing corresponding mask images.
            transform: Optional transform function applied to both image and mask.
            mask_transform: Placeholder for separate mask transforms (currently unused).
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Load an image and its corresponding mask, then return transformed tensors.

        Args:
            idx: Index of the sample to load.

        Returns:
            image: Tensor representing the input RGB image.
            mask: Tensor representing the binary segmentation mask.
        """
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        # Binarize the grayscale mask with a fixed threshold.
        mask_bin = (mask_array >= 128).astype(np.float32)

        mask = Mask(mask_bin)
        mask = mask.unsqueeze(0)

        if self.transform:
            # The transform is expected to accept both an image and a mask.
            image, mask = self.transform(image, mask)
        else:
            # Fallback conversion when no transform is provided.
            image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).float()
            mask = mask.unsqueeze(0)

        return image, mask
