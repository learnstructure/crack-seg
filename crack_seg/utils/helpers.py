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


def plot_loss_curve(train_losses: list[float], val_losses: list[float], save_path: str | Path | None = None):
    """Plot training and validation loss curves and optionally save the figure."""
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
