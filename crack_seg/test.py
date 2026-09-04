
import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import (
    get_train_val_test_datasets,
    load_multiple_datasets,
    create_splits,
)
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import val_transform
from crack_seg.utils.metrics import (
    iou_score,
    dice_coefficient,
    pixel_accuracy,
    precision_score,
    recall_score,
    specificity_score,
)


def evaluate(model, data_loader, device):
    """Runs evaluation on the provided data loader and returns a dict of metrics."""
    model.eval()
    metric_lists = {
        "iou": [],
        "dice": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
    }

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for pred, mask in zip(preds, masks):
                metric_lists["iou"].append(iou_score(pred, mask).item())
                metric_lists["dice"].append(dice_coefficient(pred, mask).item())
                metric_lists["accuracy"].append(pixel_accuracy(pred, mask).item())
                metric_lists["precision"].append(precision_score(pred, mask).item())
                metric_lists["recall"].append(recall_score(pred, mask).item())
                metric_lists["specificity"].append(specificity_score(pred, mask).item())

    mean_metrics = {key: float(np.mean(values)) for key, values in metric_lists.items()}
    return mean_metrics


def evaluate_yolo_model(checkpoint_path: str):
    """Evaluate YOLO segmentation model on the test split."""
    from crack_seg.models.yolo_seg import evaluate_yolo

    print(f"\nEvaluating YOLO segmentation model: {checkpoint_path}")
    results = evaluate_yolo(checkpoint_path=checkpoint_path, split="test")
    print("\n--- YOLO Test Set Evaluation ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test a segmentation model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint (.pth for PyTorch, .pt for YOLO).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="PyTorch model key for .pth checkpoints, e.g. unet or deeplabv3plus.",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Also evaluate and display metrics separately for each dataset in the test split.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path and str(checkpoint_path).endswith(".pt"):
        evaluate_yolo_model(checkpoint_path)
        return

    if checkpoint_path is None and config.MODEL_NAME == "yolo_seg":
        dataset_suffix = "_".join(config.DATASETS)
        checkpoint_path = (
            config.CHECKPOINT_DIR
            / "yolo_runs"
            / f"yolo_seg_{dataset_suffix}.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Configured YOLO checkpoint not found: {checkpoint_path}. "
                "Train the configured YOLO model or pass --checkpoint explicitly."
            )
        evaluate_yolo_model(str(checkpoint_path))
        return

    # Determine PyTorch model name
    if checkpoint_path:
        if not args.model:
            raise ValueError(
                "--model is required for .pth checkpoints; do not infer it from a dataset-aware filename."
            )
        model_name = args.model
        print(f"Loading model '{model_name}' from checkpoint: {checkpoint_path}")

        try:
            model_module = importlib.import_module(f"crack_seg.models.{model_name}")
            model = model_module.get_model().to(config.DEVICE)
        except ImportError:
            print(f"Error: Model '{model_name}' not found in crack_seg/models.")
            return

        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    else:
        model_name = config.MODEL_NAME
        dataset_suffix = "_".join(config.DATASETS)
        checkpoint_path = config.CHECKPOINT_DIR / f"{model_name}_{dataset_suffix}.pth"
        print(f"Loading model '{model_name}' from default checkpoint: {checkpoint_path}")

        model_module = importlib.import_module(f"crack_seg.models.{model_name}")
        model = model_module.get_model().to(config.DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    # Build multi-dataset test split
    _, _, test_dataset = get_train_val_test_datasets(
        test_transform=val_transform,
        dataset_names=config.DATASETS,
        data_root=config.DATA_ROOT,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SPLIT_SEED,
        stratified=config.STRATIFIED_SPLIT,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Combined Evaluation
    print(f"\n--- Evaluating Overall Combined Test Set ({len(test_dataset)} samples) ---")
    test_metrics = evaluate(model, test_loader, config.DEVICE)

    print(f"\n--- Combined Test Set Evaluation for {model_name.upper()} ---")
    print(
        f"Test Metrics -> IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, Specificity: {test_metrics['specificity']:.4f}\n"
    )

    # Optional Per-Dataset breakdown
    if args.per_dataset and len(config.DATASETS) > 1:
        print("--- Per-Dataset Breakdown ---")
        dataset_samples_dict = load_multiple_datasets(config.DATASETS, data_root=config.DATA_ROOT)
        for ds_name, ds_samples in dataset_samples_dict.items():
            # Extract only test portion for this dataset
            _, _, ds_test_samples = create_splits(
                {ds_name: ds_samples},
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                seed=config.SPLIT_SEED,
                stratified=False,
            )
            if not ds_test_samples:
                continue

            ds_test_ds = CrackDataset(samples=ds_test_samples, transform=val_transform)
            ds_loader = DataLoader(
                ds_test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
            )
            ds_metrics = evaluate(model, ds_loader, config.DEVICE)
            print(
                f"[{ds_name}] (N={len(ds_test_ds)}) -> IoU: {ds_metrics['iou']:.4f}, "
                f"Dice: {ds_metrics['dice']:.4f}, Accuracy: {ds_metrics['accuracy']:.4f}, "
                f"Precision: {ds_metrics['precision']:.4f}, Recall: {ds_metrics['recall']:.4f}\n"
            )


if __name__ == "__main__":
    main()
