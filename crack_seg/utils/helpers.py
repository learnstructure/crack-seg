import importlib
from pathlib import Path
import torch
from crack_seg.config import DEVICE

def load_model(model_name: str, checkpoint_path: str | Path, device=DEVICE):
    model_module = importlib.import_module(f"crack_seg.models.{model_name}")
    model = model_module.get_model().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def load_model_from_checkpoint(checkpoint_path: str | Path, device=DEVICE):
    checkpoint_path = Path(checkpoint_path)
    model_name = checkpoint_path.stem.split("_")[0]
    return load_model(model_name, checkpoint_path, device)