import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from crack_seg.config import *
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import train_transform, val_transform
from crack_seg.utils.metrics import DiceLoss, iou_score, dice_coefficient
import importlib


def main():
    # Prepare datasets
    train_dataset = CrackDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=train_transform,
    )
    val_dataset = CrackDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,  # Using test as validation for simplicity
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

    # Dynamically load model
    model_module = importlib.import_module(f"crack_seg.models.{MODEL_NAME}")
    model = model_module.get_model().to(DEVICE)

    # Loss function
    if LOSS == "dice":
        criterion = DiceLoss()
    elif LOSS == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()  # default

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5
    )

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"
        ):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        ious, dices = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Metrics
                preds = torch.sigmoid(outputs)
                for pred, mask in zip(preds, masks):
                    ious.append(iou_score(pred, mask).item())
                    dices.append(dice_coefficient(pred, mask).item())

        val_loss /= len(val_loader.dataset)
        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, IoU = {mean_iou:.4f}, Dice = {mean_dice:.4f}"
        )

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"),
            )
            print(f"Best model saved with val loss {val_loss:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
