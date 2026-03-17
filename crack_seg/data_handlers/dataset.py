import os
from PIL import Image
import torch
from torch.utils.data import Dataset
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

        if self.transform:
            # The v2 transform takes both image and mask as input
            image, mask = self.transform(image, mask)
        else:
            # Basic fallback if no transform is provided
            image = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = mask.unsqueeze(0)

        return image, mask
