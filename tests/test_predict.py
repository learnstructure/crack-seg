import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from crack_seg.predict import predict, predict_with_patches


class DummyPredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 conv mapping 3 channels to 1 channel
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
        nn.init.constant_(self.conv.weight, 0.5)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)


def test_predict_with_image_object(dummy_rgb_image):
    model = DummyPredictModel()
    pred = predict(dummy_rgb_image, model, device=torch.device("cpu"))

    assert isinstance(pred, np.ndarray)
    # predict uses pred_transform which resizes to config.IMG_SIZE (448, 448)
    assert pred.ndim == 2
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_predict_with_path(temp_image_file):
    model = DummyPredictModel()
    pred = predict(temp_image_file, model, device=torch.device("cpu"))

    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 2


def test_predict_with_patches_preserves_dimensions(dummy_rgb_image):
    model = DummyPredictModel()
    # dummy_rgb_image is 100x100
    pred = predict_with_patches(
        image_path=dummy_rgb_image,
        model=model,
        device=torch.device("cpu"),
        patch_size=64,
        stride=32,
        batch_size=4,
        use_amp=False,
    )

    assert isinstance(pred, np.ndarray)
    # Output must match the exact original dimensions of input image (100, 100)
    assert pred.shape == (100, 100)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_predict_with_patches_file_path(temp_image_file):
    model = DummyPredictModel()
    pred = predict_with_patches(
        image_path=temp_image_file,
        model=model,
        device=torch.device("cpu"),
        patch_size=64,
        stride=32,
        batch_size=2,
        use_amp=False,
    )

    assert isinstance(pred, np.ndarray)
    assert pred.shape == (100, 100)
