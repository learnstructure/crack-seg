import torch
from pathlib import Path

# Base Paths
# BASE_DIR: directory of the crack-seg repository
BASE_DIR = Path(__file__).resolve().parent.parent
# WORKSPACE_DIR: workspace root containing 'crack-seg' and 'data'
WORKSPACE_DIR = BASE_DIR.parent
# DATA_ROOT: default path to the multi-dataset folder in workspace
DATA_ROOT = WORKSPACE_DIR / "data"

# Active Datasets to combine for training, validation, and testing
# Supported: "CConCrack", "NCCD-PF_Dataset", "DeepCrack", "CRACK500"
# DATASETS = ["CConCrack", "NCCD-PF_Dataset"]
DATASETS = ["NCCD-PF_Dataset"]

# NCCD-PF specific: set True to use only images with cracks (recommended), or False for all images
NCCD_CRACKED_ONLY = True

# Dataset Split Configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42
STRATIFIED_SPLIT = True  # Ensure each dataset is split proportionally across train/val/test

# Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# Model Configuration
# Options: "unet", "deeplabv3", "deeplabv3plus", "segformer", "segnet", "unetplusplus", "fpn", "linknet", "pspnet", "yolo_seg"
MODEL_NAME = "unet"  # Choose from the supported models

ENCODER_NAME = "resnet34"  # For SMP models
PRETRAINED = True

# Ultralytics YOLO Segmentation Configuration
# YOLO26 is the latest Ultralytics family. Options: "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt".
YOLO_MODEL_WEIGHTS = "yolo26n-seg.pt"
YOLO_DATASET_DIR = BASE_DIR / "yolo_data"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "crack_data.yaml"

# Data Preprocessing
IMG_SIZE = (448, 448)  # Resize images to this size (H, W)
NUM_CLASSES = 1  # Binary segmentation
MASK_THRESHOLD = 128  # Grayscale threshold to binarize masks

# Loss and Metrics
LOSS = "dice"  # "dice", "bce"
METRICS = ["iou", "dice", "accuracy", "precision", "recall", "specificity"]

# Checkpoints
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

