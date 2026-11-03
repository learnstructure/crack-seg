import pytest
import torch
from crack_seg.models import unet, deeplabv3plus


def test_unet_instantiation_and_forward():
    model = unet.get_model()
    model.eval()

    # Small dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 1, 224, 224)


def test_deeplabv3plus_instantiation_and_forward():
    model = deeplabv3plus.get_model()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 1, 224, 224)
