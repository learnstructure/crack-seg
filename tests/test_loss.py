import pytest
import torch
from crack_seg.utils.loss import (
    DiceLoss,
    FocalLoss,
    DiceFocalLoss,
    get_loss,
)


def test_dice_loss_perfect_prediction():
    loss_fn = DiceLoss()
    # High positive logits for class 1
    logits = torch.full((2, 1, 32, 32), 10.0)
    targets = torch.ones((2, 1, 32, 32))

    loss = loss_fn(logits, targets)
    assert loss.item() < 0.05
    assert loss.item() >= 0.0


def test_dice_loss_complete_mismatch():
    loss_fn = DiceLoss()
    # Negative logits for class 1 (predicting 0) while target is 1
    logits = torch.full((2, 1, 32, 32), -10.0)
    targets = torch.ones((2, 1, 32, 32))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.95


def test_dice_loss_backward():
    loss_fn = DiceLoss()
    logits = torch.randn((2, 1, 16, 16), requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_focal_loss_values_and_backward():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn((2, 1, 16, 16), requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_dice_focal_loss():
    loss_fn = DiceFocalLoss(weight_dice=0.5, weight_focal=0.5)
    logits = torch.randn((2, 1, 16, 16), requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_get_loss_factory():
    assert isinstance(get_loss("dice"), DiceLoss)
    assert isinstance(get_loss("focal"), FocalLoss)
    assert isinstance(get_loss("dice_focal"), DiceFocalLoss)
    assert isinstance(get_loss("bce"), torch.nn.BCEWithLogitsLoss)

    with pytest.raises(ValueError):
        get_loss("unknown_loss_type")
