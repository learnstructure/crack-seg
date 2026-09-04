# Concrete Crack Surface Segmentation (`crack_seg`)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)]()

A modular, extensible deep learning framework for semantic segmentation and detection of concrete cracks in civil infrastructure. Built with PyTorch, Segmentation Models PyTorch (SMP), and Ultralytics YOLO, supporting multi-dataset training, stratified splitting, comprehensive metric evaluation, and patch-based inference.

---

## 📌 Key Features

- **Multi-Dataset Aggregation & Stratified Splitting**:
  - Combine multiple benchmark crack datasets (e.g., **CConCrack**, **NCCD-PF_Dataset**, **DeepCrack**, **CRACK500**) seamlessly.
  - Stratified, deterministic splitting ensuring balanced representation across **Train** (80%), **Validation** (10%), and **Test** (10%) splits without copying or duplicating gigabytes of files on disk.
  - Automatic pairing of differing image/mask naming conventions (e.g. `image_X.jpg` <-> `mask_X.png`) and empty-mask filtering for pre-failure datasets (NCCD-PF).

- **Diverse Model Architectures**:
  - **Standard PyTorch / SMP Semantic Segmentation**: UNet, DeepLabV3, DeepLabV3+, SegFormer (Mix Transformer), SegNet, UNet++, FPN, LinkNet, PSPNet with pre-trained backbones (ResNet, EfficientNet, MiT).
  - **YOLO Segmentation (Ultralytics)**: Full support for YOLOv8-seg, YOLOv11-seg, etc., with automatic mask-to-polygon dataset export.

- **Comprehensive Evaluation & Diagnostics**:
  - Computes IoU (Jaccard Index), Dice Coefficient (F1-score), Pixel Accuracy, Precision, Recall, and Specificity.
  - Supports combined multi-dataset testing as well as `--per-dataset` metric breakdowns to analyze generalization per domain.
  - Class imbalance reporting tool to measure foreground crack pixel ratios.

- **Inference on High-Resolution Images**:
  - Direct full-image prediction.
  - Overlapping sliding-window patch prediction (`--use-patches`) for ultra-high-resolution inspection images.

---

## 📂 Project Structure

```
crack-segmentation/
├── data/                                 # Datasets root directory
│   ├── CConCrack/                        # Train/Test images & masks
│   ├── NCCD-PF_Dataset/                  # Pre-failure narrow crack dataset
│   ├── DeepCrack/                        # DeepCrack train_img/train_lab, etc.
│   └── CRACK500/                         # CRACK500 dataset
└── crack-seg/                            # Main repository package
    ├── checkpoints/                      # Saved .pth and .pt model checkpoints
    ├── yolo_data/                        # Auto-generated YOLO polygon dataset & crack_data.yaml
    ├── main.ipynb                        # Interactive exploration & demo notebook
    ├── pyproject.toml                    # Package dependencies & configuration
    └── crack_seg/                        # Source package
        ├── config.py                     # Central configuration (datasets, model, hyperparameters)
        ├── train.py                      # Unified training pipeline (PyTorch & YOLO)
        ├── test.py                       # Evaluation script with per-dataset metric breakdowns
        ├── predict.py                    # Inference script (standard and patch-based)
        ├── data_handlers/
        │   ├── dataset.py                # PyTorch CrackDataset class
        │   ├── dataset_loaders.py        # Multi-dataset adapters, registry & stratified splitter
        │   ├── transforms.py             # Data augmentations (torchvision v2)
        │   ├── yolo_exporter.py          # Binary mask -> YOLO polygon txt converter
        │   └── class_imbalance.py        # Dataset class imbalance calculation & diagnostics
        ├── models/
        │   ├── unet.py                   # UNet (SMP)
        │   ├── deeplabv3.py              # DeepLabV3 (SMP)
        │   ├── deeplabv3plus.py          # DeepLabV3+ (SMP)
        │   ├── segformer.py              # SegFormer (MiT backbone)
        │   ├── segnet.py                 # SegNet implementation
        │   ├── unetplusplus.py           # UNet++ (SMP)
        │   ├── fpn.py, linknet.py...     # FPN, LinkNet, PSPNet
        │   └── yolo_seg.py               # Ultralytics YOLO segmentation wrapper
        └── utils/
            ├── helpers.py                # Loss plotting & general utilities
            ├── metrics.py                # Loss functions & evaluation metrics
            ├── post_processing.py        # Crack length/width estimation
            └── visualization.py          # Mask overlays & visual diagnostics
```

---

## 🚀 Installation

### 1. Prerequisites
- Python >= 3.8
- CUDA-capable GPU recommended for training

### 2. Conda Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd crack-segmentation/crack-seg

# Activate your Conda environment (example: structeng or crack-seg)
conda activate structeng

# Install crack_seg in editable mode with YOLO support
pip install -e ".[yolo]"
```

Or install core dependencies directly:
```bash
pip install -e .
pip install ultralytics opencv-python  # Required for YOLO segmentation
```

---

## ⚙️ Configuration (`crack_seg/config.py`)

All global parameters are managed in `crack_seg/config.py`:

```python
# Select active datasets to combine
DATASETS = ["CConCrack", "NCCD-PF_Dataset"]

# Split configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42
STRATIFIED_SPLIT = True

# Model architecture selection
# Options: "unet", "deeplabv3", "deeplabv3plus", "segformer", "segnet", "unetplusplus", "fpn", "linknet", "pspnet", "yolo_seg"
MODEL_NAME = "unet"
ENCODER_NAME = "resnet34"

# YOLO settings
YOLO_MODEL_WEIGHTS = "yolov8n-seg.pt"  # or "yolo11n-seg.pt", "yolov8s-seg.pt"

# Training parameters
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
IMG_SIZE = (448, 448)
```

---

## 📖 Usage Guide

### 1. Inspect Datasets & Class Imbalance
Verify dataset pairings, view split numbers, and inspect foreground crack ratios:

```bash
# Inspect dataset discovery and sample counts per split:
python -m crack_seg.data_handlers.dataset_loaders

# Compute class imbalance report for the active training split:
python -m crack_seg.data_handlers.class_imbalance
```

---

### 2. Training

#### A. Training PyTorch / SMP Segmentation Models
Set `MODEL_NAME = "unet"` (or any SMP model) in `config.py`, then run:
```bash
python -m crack_seg.train
```
- Trains with the configured loss (`dice` or `bce`), validates each epoch, and saves the selected checkpoint to `checkpoints/{MODEL_NAME}_{DATASET}.pth`.
- Generates a training and validation loss curve at `checkpoints/{MODEL_NAME}_{DATASET}_loss_curve.png`.

#### B. Training YOLO Segmentation Models
Set `MODEL_NAME = "yolo_seg"` in `config.py`, then run:
```bash
python -m crack_seg.train
```
- Automatically exports the multi-dataset into normalized polygon annotations in `yolo_data/` and runs the Ultralytics YOLO segmentation training pipeline. Checkpoints are saved under `checkpoints/yolo_runs/`.

---

### 3. Evaluation & Testing

#### Evaluating on Combined Test Set
```bash
# Evaluate PyTorch model:
python -m crack_seg.test --model unet --checkpoint checkpoints/unet_NCCD-PF_Dataset.pth

# Evaluate YOLO model:
python -m crack_seg.test --checkpoint checkpoints/yolo_runs/yolo_seg_NCCD-PF_Dataset.pt
```

#### Evaluating with Per-Dataset Breakdown
View model performance separated across individual datasets to measure cross-domain generalization:
```bash
python -m crack_seg.test --model unet --checkpoint checkpoints/unet_NCCD-PF_Dataset.pth --per-dataset
```

Output example:
```
--- Combined Test Set Evaluation for UNET ---
Test Metrics -> IoU: 0.7642, Dice: 0.8663, Accuracy: 0.9851, Precision: 0.8812, Recall: 0.8520

--- Per-Dataset Breakdown ---
[CConCrack]       (N=44)  -> IoU: 0.7910, Dice: 0.8833, Accuracy: 0.9880, Precision: 0.8950, Recall: 0.8720
[NCCD-PF_Dataset] (N=69)  -> IoU: 0.7470, Dice: 0.8550, Accuracy: 0.9832, Precision: 0.8720, Recall: 0.8390
```

---

### 4. Single-Image Prediction & Inference

```bash
# Inference with a PyTorch checkpoint:
python -m crack_seg.predict --image path/to/image.jpg --model unet --checkpoint checkpoints/unet_NCCD-PF_Dataset.pth

# Patch-based sliding window inference for large images:
python -m crack_seg.predict --image path/to/large_image.jpg --model unet --checkpoint checkpoints/unet_NCCD-PF_Dataset.pth --use-patches --patch-size 448 --stride 224

# Inference with a YOLO segmentation checkpoint:
python -m crack_seg.predict --image path/to/image.jpg --checkpoint checkpoints/yolo_runs/yolo_seg_NCCD-PF_Dataset.pt
```

Predictions are saved as binary masks (e.g. `{image_name}_prediction.png`).

---

## 📊 Supported Models

| Architecture | Model Key in Config | Framework | Default Backbone |
| :--- | :--- | :--- | :--- |
| **UNet** | `unet` | SMP | ResNet-34 / ResNet-101 |
| **DeepLabV3** | `deeplabv3` | SMP | ResNet-34 |
| **DeepLabV3+** | `deeplabv3plus`| SMP | ResNet-34 |
| **SegFormer** | `segformer` | SMP | MiT-B2 |
| **UNet++** | `unetplusplus` | SMP | ResNet-34 |
| **FPN** | `fpn` | SMP | ResNet-34 |
| **LinkNet** | `linknet` | SMP | ResNet-34 |
| **PSPNet** | `pspnet` | SMP | ResNet-34 |
| **SegNet** | `segnet` | Custom PyTorch | VGG-style Encoder |
| **YOLOv8-seg**| `yolo_seg` | Ultralytics | `yolov8n-seg.pt` |
| **YOLOv11-seg**| `yolo_seg` | Ultralytics | `yolo11n-seg.pt` |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




