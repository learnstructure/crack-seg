import pytest
import numpy as np
import torch
from crack_seg.utils.metrics import (
    iou_score,
    dice_coefficient,
    pixel_accuracy,
    precision_score,
    recall_score,
    specificity_score,
    _skeletonize,
    cldice_score,
    centerline_buffer_metrics,
    crack_length_in_pixels,
    crack_orientation_metrics,
    crack_orientation_mae,
)


def test_area_metrics_perfect_match():
    # Exactly matching tensors
    target = torch.zeros((1, 1, 32, 32))
    target[:, :, 10:20, 10:20] = 1.0
    pred = target.clone()

    assert torch.isclose(iou_score(pred, target), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(dice_coefficient(pred, target), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(pixel_accuracy(pred, target), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(precision_score(pred, target), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(recall_score(pred, target), torch.tensor(1.0), atol=1e-4)


def test_area_metrics_complete_mismatch():
    target = torch.zeros((1, 1, 32, 32))
    target[:, :, :16, :] = 1.0
    pred = torch.zeros((1, 1, 32, 32))
    pred[:, :, 16:, :] = 1.0

    assert torch.isclose(iou_score(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(dice_coefficient(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(pixel_accuracy(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(precision_score(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(recall_score(pred, target), torch.tensor(0.0), atol=1e-4)


def test_area_metrics_all_zero_prediction():
    target = torch.zeros((1, 1, 32, 32))
    target[:, :, 10:20, 10:20] = 1.0
    pred = torch.zeros((1, 1, 32, 32))

    assert torch.isclose(iou_score(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(dice_coefficient(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(precision_score(pred, target), torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(recall_score(pred, target), torch.tensor(0.0), atol=1e-4)


def test_skeletonize_algorithms(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]

    skel_lee = _skeletonize(gt_mask, method="lee")
    skel_zhang = _skeletonize(gt_mask, method="zhang")

    assert skel_lee.shape == gt_mask.shape
    assert skel_zhang.shape == gt_mask.shape
    assert skel_lee.sum() > 0
    assert skel_zhang.sum() > 0
    # Skeleton should be much thinner than original 4px mask
    assert skel_lee.sum() < gt_mask.sum()


def test_skeletonize_morphological_ops(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]

    skel_open = _skeletonize(gt_mask, morph_op="open", morph_kernel_size=3)
    skel_close = _skeletonize(gt_mask, morph_op="close", morph_kernel_size=3, morph_iterations=1)

    assert skel_open.shape == gt_mask.shape
    assert skel_close.shape == gt_mask.shape


def test_cldice_score(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]
    pred_mask = synthetic_crack_data["pred_mask"]

    # Identical masks
    score_perfect = cldice_score(gt_mask, gt_mask)
    assert score_perfect >= 0.95

    # Overlapping synthetic cracks
    score_overlap = cldice_score(pred_mask, gt_mask)
    assert 0.0 <= score_overlap <= 1.0
    assert score_overlap > 0.5


def test_centerline_buffer_metrics(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]
    pred_mask = synthetic_crack_data["pred_mask"]

    metrics = centerline_buffer_metrics(pred_mask, gt_mask, tolerance_px=5.0)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "iou" in metrics
    assert "gt_length_px" in metrics
    assert "pred_length_px" in metrics

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0


def test_crack_length_in_pixels(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]
    length = crack_length_in_pixels(gt_mask)
    assert length > 0
    # The synthetic crack spans from x=10 to x=89 (~80 px)
    assert 60 <= length <= 90


def test_crack_orientation_metrics(synthetic_crack_data):
    gt_mask = synthetic_crack_data["gt_mask"]
    pred_mask = synthetic_crack_data["pred_mask"]

    ori = crack_orientation_metrics(pred_mask, gt_mask)
    assert "mae_deg" in ori
    assert "pred_angle_deg" in ori
    assert "gt_angle_deg" in ori

    # Since crack is horizontal, angular error between parallel cracks should be very low
    mae = crack_orientation_mae(pred_mask, gt_mask)
    assert mae < 15.0
