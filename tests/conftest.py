import pytest
import numpy as np
import torch
from PIL import Image


@pytest.fixture
def synthetic_crack_data():
    """
    Creates a synthetic 100x100 binary mask with a known horizontal line crack.
    """
    height, width = 100, 100
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask[48:52, 10:90] = 1  # 4px thick horizontal crack from x=10 to x=89

    pred_mask = np.zeros((height, width), dtype=np.uint8)
    pred_mask[49:51, 12:88] = 1  # slightly thinner crack closely overlapping

    return {
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
        "height": height,
        "width": width,
    }


@pytest.fixture
def dummy_rgb_image():
    """
    Creates a simple 100x100 RGB PIL image.
    """
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Add some variation
    arr[40:60, 40:60] = [200, 50, 50]
    return Image.fromarray(arr)


@pytest.fixture
def temp_image_file(tmp_path, dummy_rgb_image):
    """
    Saves a dummy RGB image to a temporary file path.
    """
    img_path = tmp_path / "test_bridge.jpg"
    dummy_rgb_image.save(img_path)
    return img_path
