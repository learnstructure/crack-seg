import importlib
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import get_train_val_test_datasets
from crack_seg.data_handlers.transforms import train_transform, val_transform
from crack_seg.utils.helpers import plot_loss_curve
from crack_seg.utils.metrics import (
    DiceLoss,
    iou_score,
    dice_coefficient,
    pixel_accuracy,
    precision_score,
    recall_score,
    specificity_score,
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def train_pytorch():
    """Train and validate standard PyTorch / SMP segmentation models."""
    dataset_suffix = "_".join(config.DATASETS)
    checkpoint_stem = f"{config.MODEL_NAME}_{dataset_suffix}"

    print(f"\n==========================================")
    print(f"  Training Model: {config.MODEL_NAME.upper()}")
    print(f"  Active Datasets: {config.DATASETS}")
    print(f"  Device: {config.DEVICE}")
    print(f"==========================================\n")

    # Build multi-dataset training and validation sets
    train_dataset, val_dataset, _ = get_train_val_test_datasets(
        train_transform=train_transform,
        val_transform=val_transform,
        dataset_names=config.DATASETS,
        data_root=config.DATA_ROOT,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SPLIT_SEED,
        stratified=config.STRATIFIED_SPLIT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Load model dynamically from crack_seg.models
    model_module = importlib.import_module(f"crack_seg.models.{config.MODEL_NAME}")
    model = model_module.get_model().to(config.DEVICE)

    # Loss function
    criterion = DiceLoss() if config.LOSS == "dice" else nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Reduce learning rate when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5
    )

    lowest_val_loss = float("inf")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} - Training")
        ):
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            # Print first-batch memory stats for GPU monitoring
            if batch_idx == 0 and config.DEVICE.type == "cuda":
                print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        metric_lists = {
            "iou": [],
            "dice": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "specificity": [],
        }

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate metrics for each item in the batch
                preds = torch.sigmoid(outputs)
                for pred, mask in zip(preds, masks):
                    metric_lists["iou"].append(iou_score(pred, mask).item())
                    metric_lists["dice"].append(dice_coefficient(pred, mask).item())
                    metric_lists["accuracy"].append(pixel_accuracy(pred, mask).item())
                    metric_lists["precision"].append(precision_score(pred, mask).item())
                    metric_lists["recall"].append(recall_score(pred, mask).item())
                    metric_lists["specificity"].append(
                        specificity_score(pred, mask).item()
                    )

        val_loss /= len(val_loader)
        mean_metrics = {key: float(np.mean(values)) for key, values in metric_lists.items()}

        print(
            f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
        )
        print(
            f"Val Metrics -> IoU: {mean_metrics['iou']:.4f}, Dice: {mean_metrics['dice']:.4f}, "
            f"Accuracy: {mean_metrics['accuracy']:.4f}, Precision: {mean_metrics['precision']:.4f}, "
            f"Recall: {mean_metrics['recall']:.4f}, Specificity: {mean_metrics['specificity']:.4f}\n"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save model checkpoint when validation loss improves
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            save_path = config.CHECKPOINT_DIR / f"{checkpoint_stem}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with val loss {val_loss:.4f}")

    plot_save_path = config.CHECKPOINT_DIR / f"{checkpoint_stem}_loss_curve.png"
    plot_loss_curve(train_losses, val_losses, save_path=plot_save_path)
    print(f"Loss curve saved to {plot_save_path}")


def train_yolo_pipeline():
    """Train YOLO segmentation models using the Ultralytics framework."""
    from crack_seg.data_handlers.yolo_exporter import export_dataset_to_yolo
    from crack_seg.models.yolo_seg import train_yolo

    dataset_suffix = "_".join(config.DATASETS)

    # Re-export on every run so dataset and split configuration cannot become stale.
    export_dataset_to_yolo()

    train_yolo(
        data_yaml=config.YOLO_DATA_YAML,
        weights=config.YOLO_MODEL_WEIGHTS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        imgsz=config.IMG_SIZE[0],
        experiment_name=f"yolo_seg_{dataset_suffix}",
    )


def main():
    if config.MODEL_NAME.lower().startswith("yolo"):
        train_yolo_pipeline()
    else:
        train_pytorch()


if __name__ == "__main__":
    # Windows multiprocessing support
    torch.multiprocessing.freeze_support()
    main()

