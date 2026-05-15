import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from crack_seg.config import *
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import train_transform, val_transform
from crack_seg.utils.metrics import (
    DiceLoss,
    iou_score,
    dice_coefficient,
    pixel_accuracy,
    precision_score,
    recall_score,
    specificity_score,
)
import importlib
from crack_seg.utils.helpers import plot_loss_curve

torch.cuda.empty_cache()


def main():
    """Train and validate the segmentation model over multiple epochs."""

    # Build training and validation datasets
    train_dataset = CrackDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=train_transform,
    )
    val_dataset = CrackDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Load model
    model_module = importlib.import_module(f"crack_seg.models.{MODEL_NAME}")
    model = model_module.get_model().to(DEVICE)

    # Loss function
    criterion = DiceLoss() if LOSS == "dice" else nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Reduce learning rate when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5
    )

    best_val_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
        ):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            # Print first-batch memory stats for debug/monitoring.
            if batch_idx == 0:
                # print(torch.cuda.memory_summary())
                print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            optimizer.step()
            train_loss += loss.item()
            # Uncomment for shape/debug checks:
            # print(images.shape, masks.shape, masks.min(), masks.max())

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
                images, masks = images.to(DEVICE), masks.to(DEVICE)
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
        # Compute averages for validation metrics.
        mean_metrics = {key: np.mean(values) for key, values in metric_lists.items()}

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

        # Save the model checkpoint when validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with val loss {val_loss:.4f}")

    plot_save_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, save_path=plot_save_path)
    print(f"Loss curve saved to {plot_save_path}")


if __name__ == "__main__":
    # Required for Windows multiprocessing support.
    torch.multiprocessing.freeze_support()
    print(f"Using device: {DEVICE}")
    main()
