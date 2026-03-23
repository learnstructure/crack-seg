
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import importlib
import numpy as np
from crack_seg.config import *
from crack_seg.data_handlers.dataset import CrackDataset
from crack_seg.data_handlers.transforms import test_transform
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
            
            for pred, mask in zip(preds, masks):
                metric_lists["iou"].append(iou_score(pred, mask).item())
                metric_lists["dice"].append(dice_coefficient(pred, mask).item())
                metric_lists["accuracy"].append(pixel_accuracy(pred, mask).item())
                metric_lists["precision"].append(precision_score(pred, mask).item())
                metric_lists["recall"].append(recall_score(pred, mask).item())
                metric_lists["specificity"].append(specificity_score(pred, mask).item())

    # Calculate mean of all metrics
    mean_metrics = {key: np.mean(values) for key, values in metric_lists.items()}
    return mean_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model on the test set.")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint file (e.g., checkpoints/unet_best.pth)"
    )
    args = parser.parse_args()

    # Prepare test dataset
    test_dataset = CrackDataset(
        TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Load model architecture dynamically
    checkpoint_name = os.path.basename(args.checkpoint)
    model_name_from_file = checkpoint_name.split('_')[0]
    print(f"Loading model: {model_name_from_file}")

    try:
        model_module = importlib.import_module(f"crack_seg.models.{model_name_from_file}")
        model = model_module.get_model().to(DEVICE)
    except ImportError:
        print(f"Error: Model '{model_name_from_file}' not found.")
        return

    # Load the trained weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))

    # Evaluate the model
    mean_metrics = evaluate(model, test_loader, DEVICE)
    
    print("\n--- Test Set Evaluation ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mean IoU: {mean_metrics['iou']:.4f}")
    print(f"Mean Dice Coefficient: {mean_metrics['dice']:.4f}")
    print(f"Pixel Accuracy: {mean_metrics['accuracy']:.4f}")
    print(f"Precision: {mean_metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {mean_metrics['recall']:.4f}")
    print(f"Specificity: {mean_metrics['specificity']:.4f}")
    print("---------------------------")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

#python -m crack_seg.test --checkpoint checkpoints/deeplabv3_best.pth