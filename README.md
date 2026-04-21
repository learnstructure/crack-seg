# Concrete Crack Surface Segmentation

## Description

This project implements deep learning models for automatic crack detection and segmentation in concrete surface images. It leverages semantic segmentation techniques to identify and delineate cracks, supporting infrastructure inspection, maintenance, and quality control in civil engineering. The project is built with PyTorch and supports multiple segmentation architectures for comparison and experimentation.

## Features

- **Multiple Segmentation Models**: Support for popular architectures including UNet, DeepLabV3, DeepLabV3+, SegFormer, SegNet, UNet++, FPN, LinkNet, and PSPNet.
- **Training and Evaluation**: Scripts for training models with validation, testing on held-out data, and comprehensive metrics (IoU, Dice, Pixel Accuracy, Precision, Recall, Specificity).
- **Single-Image Prediction**: Inference on individual images with output masks.
- **Visualization Tools**: Utilities for displaying predictions, overlays, and analysis.
- **Post-Processing**: Crack analysis including length and width calculations.
- **Dataset Support**: Built-in handling for the CConCrack dataset with proper transforms and data loading.
- **Configurable**: Easy modification of models, hyperparameters, and paths via configuration file.
- **GPU Optimized**: Efficient training with CUDA support, memory monitoring, and pinned memory.

## Installation

### Prerequisites

- Python >= 3.8
- GPU with CUDA support recommended for training (optional for inference)

### Using pip

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd crack-seg
pip install -e .
```

### Using Conda

Alternatively, use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate crack-seg
```

For a full environment with additional tools:

```bash
conda env create -f environment_full.yml
```

## Usage

### Training

To train a segmentation model:

1. Modify `crack_seg/config.py` to select the model (e.g., set `MODEL_NAME = "unet"`) and adjust parameters as needed.
2. Ensure the dataset is placed in the `CConCrack/` directory.
3. Run the training script:

```bash
python -m crack_seg.train
```

This will train for 50 epochs by default, validate after each epoch, and save the best model to `checkpoints/{MODEL_NAME}_best.pth`.

### Testing

To evaluate a trained model on the test set:

```bash
python -m crack_seg.test --checkpoint checkpoints/unet_best.pth
```

This outputs mean metrics (IoU, Dice, etc.) for the test set. If no checkpoint is specified, it uses the default from config.

### Prediction

To perform inference on a single image:

```bash
python -m crack_seg.predict --image ./CConCrack/Test/images/CFD_001.jpg --checkpoint checkpoints/unet_best.pth
```

This saves the predicted binary mask as `{image_name}_prediction.png` in the current directory.

## Models

The project supports dynamic loading of segmentation models. Select a model by setting `MODEL_NAME` in `crack_seg/config.py`. Available models include:

- **UNet**: Classic encoder-decoder architecture from segmentation-models-pytorch (SMP).
- **DeepLabV3**: Uses atrous convolution for multi-scale context (SMP).
- **DeepLabV3+**: Improved version with an additional decoder (SMP).
- **SegFormer**: Transformer-based model with Mix Transformer encoder (SMP).
- **SegNet**: Custom PyTorch implementation with encoder-decoder and bilinear upsampling.
- **UNet++**: Nested U-Net with dense skip connections (SMP).
- **FPN**: Feature Pyramid Network (SMP).
- **LinkNet**: Lightweight encoder-decoder (SMP).
- **PSPNet**: Pyramid Scene Parsing Network (SMP).

All models use pre-trained encoders (e.g., ResNet34, EfficientNet-B3) and output logits for binary segmentation.

## Dataset

The project uses the CConCrack dataset for concrete crack segmentation:

- **Structure**: Organized into `Train/`, `Validation/`, and `Test/` directories, each containing `images/` and `masks/` subdirectories.
- **Format**: Images are RGB, masks are grayscale (binary after thresholding at 128).
- **Transforms**: Training includes resizing (448x448), random flips, color jitter, and normalization. Validation/Test use resize and normalization only.
- **Loading**: PyTorch DataLoader with batch size 8, 4 workers, and pinned memory for efficiency.

Place the dataset in the `CConCrack/` directory at the project root.

## Configuration

Key settings are defined in `crack_seg/config.py`:

- **Paths**: Dataset root (`CConCrack`), train/val/test image/mask directories.
- **Model**: Name (e.g., "unet"), encoder (e.g., "resnet34"), pre-trained flag.
- **Training**: Device ("cuda" if available), batch size (8), epochs (50), learning rate (1e-4), workers (4).
- **Data**: Image size (448x448), number of classes (1 for binary).
- **Loss/Metrics**: Loss function ("dice"), metrics list (["iou", "dice", "accuracy", "precision", "recall", "specificity"]).
- **Checkpoints**: Save directory (`checkpoints/`).

Modify this file to customize experiments.

## Dependencies

Core dependencies (from `pyproject.toml`):

- `torch`: Deep learning framework.
- `segmentation-models-pytorch`: Pre-built segmentation models.
- `numpy`: Numerical computations.
- `Pillow`: Image processing.
- `matplotlib`: Visualization.
- `tqdm`: Progress bars.

Development dependencies include `pytest`, `black`, and `flake8`.

## Demo

For interactive exploration, open `main.ipynb` in Jupyter Notebook. It includes:

- GPU availability checks.
- Model loading and prediction examples.
- Visualization of results.

## Checkpoints

Pre-trained models are available in `checkpoints/`:

- `unet_best.pth`
- `segnet_best.pth`
- `deeplabv3_best.pth`
- `deeplabv3plus_best.pth`
- `segformer_best.pth`
- `unetplusplus_best.pth`
- And others.

Use these for testing or as starting points for fine-tuning.



