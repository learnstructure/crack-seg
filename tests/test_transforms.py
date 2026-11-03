import pytest
import torch
import numpy as np
from PIL import Image
from crack_seg.config import IMG_SIZE
from crack_seg.data_handlers.transforms import (
    train_transform,
    val_transform,
    pred_transform,
    original_size_transform,
)


def test_val_transform(dummy_rgb_image):
    mask = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    img_t, mask_t = val_transform(dummy_rgb_image, mask)

    assert isinstance(img_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert img_t.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
    assert mask_t.shape == (1, IMG_SIZE[0], IMG_SIZE[1])
    assert img_t.dtype == torch.float32
    assert mask_t.dtype == torch.float32


def test_train_transform(dummy_rgb_image):
    mask = Image.fromarray(np.full((100, 100), 255, dtype=np.uint8))
    img_t, mask_t = train_transform(dummy_rgb_image, mask)

    assert isinstance(img_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert img_t.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
    assert mask_t.shape == (1, IMG_SIZE[0], IMG_SIZE[1])


def test_pred_transform(dummy_rgb_image):
    tensor = pred_transform(dummy_rgb_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, IMG_SIZE[0], IMG_SIZE[1])
    assert tensor.dtype == torch.float32


def test_original_size_transform(dummy_rgb_image):
    tensor = original_size_transform(dummy_rgb_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.float32
