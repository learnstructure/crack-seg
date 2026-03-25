
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import importlib

from crack_seg.config import *
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import val_transform
from crack_seg.utils.metrics import (
    iou_score, dice_coefficient, pixel_accuracy, 
    precision_score, recall_score, specificity_score
)

def evaluate(model, data_loader, device):
    """Runs evaluation on the provided data loader and returns a dict of metrics."""
    model.eval()
    metric_lists = { "iou": [], "dice": [], "accuracy": [], "precision": [], "recall": [], "specificity": [] }
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating on Test Set"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Calculate metrics over the whole batch for efficiency
            metric_lists["iou"].append(iou_score(preds, masks).item())
            metric_lists["dice"].append(dice_coefficient(preds, masks).item())
            metric_lists["accuracy"].append(pixel_accuracy(preds, masks).item())
            metric_lists["precision"].append(precision_score(preds, masks).item())
            metric_lists["recall"].append(recall_score(preds, masks).item())
            metric_lists["specificity"].append(specificity_score(preds, masks).item())

    # Calculate mean of all metrics
    mean_metrics = {key: np.mean(values) for key, values in metric_lists.items()}
    return mean_metrics

def main():
    # Load model
    model_module = importlib.import_module(f"crack_seg.models.{MODEL_NAME}")
    model = model_module.get_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth", map_location=DEVICE))

    # Prepare dataset
    test_dataset = CrackDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        transform=val_transform,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Evaluate
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\n--- Test Set Evaluation ---")
    print(
        f"Test Metrics -> IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, Specificity: {test_metrics['specificity']:.4f}\n"
    )

if __name__ == "__main__":
    main()
