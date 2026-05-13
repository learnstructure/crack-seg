import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
import numpy as np

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # Same filename

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale
        mask_array = np.array(mask)

        # Fixed threshold to separate crack vs background
        mask_bin = (mask_array >= 128).astype(np.float32)

        mask = Mask(mask_bin)
        mask = mask.unsqueeze(0)

        if self.transform:
            # The v2 transform takes both image and mask as input
            image, mask = self.transform(image, mask)
        else:
            # Basic fallback if no transform is provided
            image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).float()
            mask = (mask > 128).float()
            mask = mask.unsqueeze(0)

        return image, mask
