import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from crack_seg.utils.helpers import (
    create_red_overlay,
    warmup_gpu,
    predict_mask,
)
from crack_seg.utils.visualization import (
    denormalize,
    plot_overlay_comparison,
    visualize_model_overlays,
)


class DummySegmentationModel(nn.Module):
    """Simple model that outputs deterministic logits for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)


def test_create_red_overlay_numpy(dummy_rgb_image):
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:30, 20:30] = True

    overlay = create_red_overlay(dummy_rgb_image, mask, alpha=0.5, color=(255, 0, 0))
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8

    # Masked pixels should have altered red channel
    assert overlay[25, 25, 0] > 128


def test_create_red_overlay_with_path(temp_image_file):
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:20, 10:20] = 1.0

    img = Image.open(temp_image_file)
    overlay = create_red_overlay(img, mask, alpha=0.7)
    assert overlay.shape == (100, 100, 3)


def test_denormalize():
    tensor = torch.zeros((3, 32, 32))
    denorm = denormalize(tensor)
    assert denorm.shape == (3, 32, 32)
    # Check that values changed by adding mean
    assert denorm.sum() > 0


def test_plot_overlay_comparison(dummy_rgb_image, tmp_path):
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True

    overlays = {
        "Test UNet": mask,
        "Test SegFormer": create_red_overlay(dummy_rgb_image, mask),
    }

    save_file = tmp_path / "comparison.png"
    fig, axes = plot_overlay_comparison(
        image=dummy_rgb_image,
        overlays=overlays,
        original_title="Original Test",
        show=False,
        save_path=save_file,
    )

    assert len(axes) == 3
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_warmup_gpu():
    model = DummySegmentationModel()
    # Should run without error on either cpu or cuda
    warmup_gpu([model], patch_batch_size=2, patch_size=64, device=torch.device("cpu"))


def test_predict_mask_with_dummy_model(dummy_rgb_image):
    model = DummySegmentationModel()
    pred_mask, elapsed = predict_mask(
        model=model,
        image=dummy_rgb_image,
        device=torch.device("cpu"),
        use_patches=True,
        patch_size=64,
        stride=32,
        batch_size=2,
        use_amp=False,
    )

    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == (100, 100)
    assert elapsed >= 0.0


def test_visualize_model_overlays_pipeline(dummy_rgb_image, tmp_path):
    model = DummySegmentationModel()
    models = {
        "DummyModel1": model,
        "DummyModel2": ("unet", model),
    }

    out_file = tmp_path / "vis_output.png"
    result = visualize_model_overlays(
        image=dummy_rgb_image,
        models=models,
        device=torch.device("cpu"),
        patch_size=64,
        stride=32,
        batch_size=2,
        use_amp=False,
        save_path=out_file,
        show=False,
    )

    assert "predictions" in result
    assert "timings" in result
    assert "overlays" in result
    assert "DummyModel1" in result["predictions"]
    assert "DummyModel2" in result["predictions"]
    assert out_file.exists()
