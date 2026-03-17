
import torch
from PIL import Image
import argparse
from crack_seg.config import *
from crack_seg.data_handlers.transforms import pred_transform # Use the dedicated prediction transform
import importlib
import numpy as np
import os

def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    # Use the prediction-specific transform
    input_tensor = pred_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    # ---- Dynamically determine model name from checkpoint path ----
    checkpoint_name = os.path.basename(args.checkpoint)
    model_name_from_file = checkpoint_name.split('_')[0]

    print(f"Loading model: {model_name_from_file}")

    # Load model dynamically
    try:
        model_module = importlib.import_module(f"crack_seg.models.{model_name_from_file}")
        model = model_module.get_model().to(DEVICE)
    except ImportError:
        print(f"Error: Model '{model_name_from_file}' not found in crack_seg/models.")
        return

    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    pred = predict(args.image, model, DEVICE)
    # Save or display prediction
    result = Image.fromarray((pred * 255).astype(np.uint8))
    # Save with a more descriptive name
    output_filename = f"{os.path.basename(args.image).split('.')[0]}_prediction.png"
    result.save(output_filename)
    print(f"Prediction saved as {output_filename}")

if __name__ == "__main__":
    main()

#python crack_seg/predict.py --image ./CConCrack/Test/images/CFD_001.jpg --checkpoint checkpoints/unet_best.pth