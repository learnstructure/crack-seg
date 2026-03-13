import torch
from pathlib import Path

# Paths
DATA_ROOT = Path("CConCrack")  # Change to your dataset path
TRAIN_IMG_DIR = DATA_ROOT / "Train" / "images"
TRAIN_MASK_DIR = DATA_ROOT / "Train" / "masks"
TEST_IMG_DIR = DATA_ROOT / "Test" / "images"
TEST_MASK_DIR = DATA_ROOT / "Test" / "masks"

# Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# Model (change this to switch architectures)
MODEL_NAME = "unet"  # Options: "unet", "deeplabv3", etc.
ENCODER_NAME = "resnet34"  # For models that use encoders
PRETRAINED = True

# Data
IMG_SIZE = (256, 256)  # Resize images to this size
NUM_CLASSES = 1  # Binary segmentation

# Loss and metrics
LOSS = "dice"  # "dice", "bce", "combined"
METRICS = ["iou", "dice"]

# Checkpoints
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
