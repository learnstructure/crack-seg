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

- **Comprehensive Evaluation & Width-Independent Metrics**:
  - **Standard Area Metrics**: IoU (Jaccard Index), Dice Coefficient (F1-score), Pixel Accuracy, Precision, Recall, and Specificity.
  - **Width-Independent Centerline & Topology Metrics**:
    - **Distance-Tolerant Centerline Metrics (Tolerance $\tau$)**: Evaluates crack length in pixels within Euclidean distance bounds ($\tau \in \{5, 10, 20\}\text{ px}$), measuring Centerline Precision, Recall, F1, and Length-based IoU without penalty from arbitrary annotation width.
    - **clDice (Centerline Dice - CVPR 2021)**: Measures topological connectivity and centerline presence inside mask volumes.
    - **Crack Orientation & Angular Error (MAE$_\theta$)**: PCA principal-axis angle computation, Mean Angular Error between matched cracks, and structural classification (Horizontal, Vertical, Inclined).
  - Supports combined multi-model benchmark comparisons (`eval.py`) as well as `--per-dataset` metric breakdowns (`test.py`) to analyze generalization across domains.
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
    ├── eval.py                           # Multi-model benchmarking with width-independent & centerline metrics
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

# Skeletonization and Centerline Evaluation Configuration
# Options: "lee" (default, Lee's medial axis algorithm), "zhang" (Zhang-Suen thinning)
SKELETON_METHOD = "lee"
CENTERLINE_TOLERANCE_PX = 20.0  # Euclidean distance tolerance buffer in pixels
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

#### A. Multi-Model Benchmark with Width-Independent Metrics (`eval.py`)
In concrete crack segmentation, ground-truth annotations often vary wildly in drawn width (e.g. 2 px to 25 px thick), which severely penalizes standard area-based IoU even when the crack trajectory is captured with sub-millimeter precision. 

The `eval.py` benchmark evaluates all active PyTorch SMP and YOLO checkpoints on the unseen test split and groups results into three complementary categories:
1. **Area-based Metrics**: Standard IoU, Dice, Accuracy, Precision, Recall, Specificity.
2. **Topology & Centerline Metrics (Width-Independent)**:
   - **`clDice`**: Centerline Dice (CVPR 2021) measuring topological connectivity.
   - **`Centerline_Prec@20`**: % of predicted crack length within 20px Euclidean bound of true crack.
   - **`Centerline_Rec@20`**: % of true crack length detected within 20px Euclidean bound.
   - **`Centerline_F1@20`**: Harmonic mean of centerline precision & recall.
   - **`Centerline_IoU@20`**: Length-based Jaccard Index.
   - **`Orientation_MAE(deg)`**: Mean Angular Error in degrees between matched cracks ($0^\circ$ = perfect alignment).
3. **Skeleton Dilated IoU (Legacy)**: IoU computed on dilated centerlines across radii $\tau \in \{1, 3, 6, 20\}\text{ px}$.

```bash
# Run default evaluation across all discovered checkpoints:
python eval.py

# Switch skeletonization algorithm ('lee' or 'zhang'):
python eval.py --skeleton-method zhang

# Customize Euclidean tolerance radius (e.g. ±10 px total corridor width of ~20 px):
python eval.py --tolerance-px 10.0

# View all CLI options:
python eval.py --help
```

Output example:
```text
Evaluation Results — 2726 Unseen Test Samples
========================================================================================================================
Metric                  UNet (NCCD-PF)    UNet (CCon+NCCD)   UNet (4 Datasets)     YOLO-seg (NCCD) YOLO-seg (4 Datasets)
------------------------------------------------------------------------------------------------------------------------
--- 1. Area-based ---
------------------------------------------------------------------------------------------------------------------------
IoU                             0.1513              0.5282              0.5587              0.1165              0.4728
Dice                            0.2138              0.6616              0.6906              0.1544              0.5856
Accuracy                        0.9243              0.9757              0.9793              0.9600              0.9767
Precision                       0.3212              0.6772              0.6833              0.8443              0.7459
Recall                          0.2574              0.7277              0.7688              0.1667              0.6033
Specificity                     0.9573              0.9854              0.9875              0.9983              0.9891
------------------------------------------------------------------------------------------------------------------------
--- 2. Topology & Centerline (Width-Independent) ---
------------------------------------------------------------------------------------------------------------------------
clDice                          0.2410              0.6745              0.7120              0.1680              0.6105
Centerline_Prec@20              0.4520              0.8920              0.9150              0.8850              0.9020
Centerline_Rec@20               0.3810              0.8710              0.9040              0.2430              0.8210
Centerline_F1@20                0.4136              0.8814              0.9095              0.3812              0.8596
Centerline_IoU@20               0.3250              0.8120              0.8410              0.2840              0.7780
Orientation_MAE(deg)           24.5000              7.8200              6.3500             18.4000              9.1200
------------------------------------------------------------------------------------------------------------------------
--- 3. Skeleton Dilated IoU (Legacy) ---
------------------------------------------------------------------------------------------------------------------------
skel_IoU(tol=1)                 0.1062              0.2452              0.2559              0.0862              0.2134
skel_IoU(tol=3)                 0.1755              0.4226              0.4471              0.1360              0.3716
skel_IoU(tol=6)                 0.2190              0.5472              0.5792              0.1647              0.4874
skel_IoU(tol=20)                0.3120              0.7240              0.7580              0.2210              0.6650
========================================================================================================================
```

---

#### B. Single-Model Testing with Per-Dataset Breakdown (`test.py`)
Evaluate a single checkpoint with generalization breakdowns separated across individual datasets:
```bash
# Evaluate PyTorch model:
python -m crack_seg.test --model unet --checkpoint checkpoints/unet_NCCD-PF_Dataset.pth --per-dataset

# Evaluate YOLO model:
python -m crack_seg.test --checkpoint checkpoints/yolo_runs/yolo_seg_NCCD-PF_Dataset.pt --per-dataset
```

---

#### C. Programmatic Python API for Metrics
You can use the new width-independent functions directly in your own code or notebooks:

```python
from crack_seg.utils.metrics import (
    centerline_buffer_metrics,
    crack_length_in_pixels,
    cldice_score,
    crack_orientation_metrics,
)

# 1. Measure crack lengths and tolerance buffer parameters
metrics = centerline_buffer_metrics(pred_mask, gt_mask, tolerance_px=20.0)
print(f"True Crack Length:       {metrics['gt_length_px']} px")
print(f"Pred Crack Length:       {metrics['pred_length_px']} px")
print(f"Pred in 20px Bound:      {metrics['pred_len_in_bound_px']} px ({metrics['precision']*100:.1f}%)")
print(f"True Covered in 20px:    {metrics['gt_len_covered_px']} px ({metrics['recall']*100:.1f}%)")
print(f"Centerline F1@20:        {metrics['f1_score']:.4f}")
print(f"Centerline IoU@20:       {metrics['iou']:.4f}")

# 2. Topology connectivity (clDice)
print(f"clDice:                  {cldice_score(pred_mask, gt_mask):.4f}")

# 3. Crack orientation & Mean Angular Error (degrees)
ori = crack_orientation_metrics(pred_mask, gt_mask)
print(f"Orientation MAE:         {ori['mae_deg']:.1f}°")
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




