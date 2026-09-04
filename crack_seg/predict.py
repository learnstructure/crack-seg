import argparse
import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from crack_seg import config
from crack_seg.data_handlers.transforms import (
    pred_transform,
    original_size_transform,
)


def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = pred_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()
    return pred


def predict_with_patches(image_path, model, device, patch_size=448, stride=224):
    """
    Predicts segmentation masks on large images by dividing them into overlapping patches.
    """
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    input_tensor = original_size_transform(image).unsqueeze(0).to(device)

    preds_sum = torch.zeros((1, 1, image_height, image_width), device=device)
    overlap_count = torch.zeros((1, 1, image_height, image_width), device=device)

    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            y_end = min(y + patch_size, image_height)
            x_end = min(x + patch_size, image_width)

            patch = input_tensor[:, :, y:y_end, x:x_end]

            pad_h = patch_size - patch.shape[2]
            pad_w = patch_size - patch.shape[3]
            if pad_h > 0 or pad_w > 0:
                patch = torch.nn.functional.pad(
                    patch, (0, pad_w, 0, pad_h), mode="constant", value=0
                )

            with torch.no_grad():
                patch_pred = model(patch)
                patch_pred = torch.sigmoid(patch_pred)

            preds_sum[:, :, y:y_end, x:x_end] += patch_pred[:, :, : y_end - y, : x_end - x]
            overlap_count[:, :, y:y_end, x:x_end] += 1

    final_pred = preds_sum / overlap_count
    final_pred = final_pred.cpu().numpy().squeeze()
    return final_pred


def main():
    parser = argparse.ArgumentParser(description="Run crack segmentation inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth for PyTorch, .pt for YOLO).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="PyTorch model key for .pth checkpoints, e.g. unet or deeplabv3plus.",
    )
    parser.add_argument(
        "--use-patches",
        action="store_true",
        help="Use patch-based sliding window inference for large images.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=448,
        help="Patch size for patched inference.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=224,
        help="Stride for patched inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path for saved prediction mask.",
    )
    args = parser.parse_args()

    output_filename = (
        args.output or f"{Path(args.image).stem}_prediction.png"
    )

    # Check if checkpoint is YOLO model
    if args.checkpoint.endswith(".pt") or "yolo" in os.path.basename(args.checkpoint).lower():
        from crack_seg.models.yolo_seg import predict_yolo
        print(f"Running YOLO segmentation inference using: {args.checkpoint}")
        predict_yolo(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            save=True,
            output_path=output_filename,
        )
        return

    # PyTorch Model Inference
    if not args.model:
        raise ValueError(
            "--model is required for .pth checkpoints; do not infer it from a dataset-aware filename."
        )
    model_name = args.model
    print(f"Loading PyTorch model: {model_name} from {args.checkpoint}")

    try:
        model_module = importlib.import_module(
            f"crack_seg.models.{model_name}"
        )
        model = model_module.get_model().to(config.DEVICE)
    except ImportError:
        print(f"Error: Model '{model_name}' not found in crack_seg/models.")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=config.DEVICE))
    model.eval()

    if args.use_patches:
        pred = predict_with_patches(
            args.image, model, config.DEVICE, args.patch_size, args.stride
        )
    else:
        pred = predict(args.image, model, config.DEVICE)

    result = Image.fromarray((pred * 255).astype(np.uint8))
    result.save(output_filename)
    print(f"Prediction saved as: {output_filename}")


if __name__ == "__main__":
    main()

