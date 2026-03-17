import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from crack_seg.config import *
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import val_transform
from crack_seg.utils.metrics import iou_score, dice_coefficient
import importlib


def main():
    # Test dataset
    test_dataset = CrackDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        transform=val_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Load model
    model_module = importlib.import_module(f"crack_seg.models.{MODEL_NAME}")
    model = model_module.get_model().to(DEVICE)
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pth"))
    model.load_state_dict(checkpoint)
    model.eval()

    ious, dices = [], []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            for pred, mask in zip(preds, masks):
                ious.append(iou_score(pred, mask).item())
                dices.append(dice_coefficient(pred, mask).item())

    print(
        f"Test Results: Mean IoU = {sum(ious)/len(ious):.4f}, Mean Dice = {sum(dices)/len(dices):.4f}"
    )


if __name__ == "__main__":
    main()
