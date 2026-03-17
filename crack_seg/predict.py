import torch
from PIL import Image
import argparse
from crack_seg.config import *
from crack_seg.data_handlers.transforms import val_transform
import importlib
import numpy as np


def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)
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

    # Load model
    model_module = importlib.import_module(f"crack_seg.models.{MODEL_NAME}")
    model = model_module.get_model().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    pred = predict(args.image, model, DEVICE)
    # Save or display prediction
    result = Image.fromarray((pred * 255).astype(np.uint8))
    result.save("prediction.png")
    print("Prediction saved as prediction.png")


if __name__ == "__main__":
    main()

#run via command
#python predict.py --image /path/to/image.jpg --checkpoint checkpoints/unet_best.pth
