from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import shutil
from PIL import Image

from crack_seg import config


def get_model(weights: Optional[str] = None):
    """
    Load an Ultralytics YOLO segmentation model.
    
    Args:
        weights: Pretrained weights file or model name (e.g. 'yolov8n-seg.pt', 'yolo11n-seg.pt').
        
    Returns:
        ultralytics.YOLO model instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "The 'ultralytics' package is required to use YOLO segmentation models. "
            "Please install it with 'pip install ultralytics'."
        )

    weights = weights or config.YOLO_MODEL_WEIGHTS
    model = YOLO(weights)
    return model


def train_yolo(
    data_yaml: Optional[Union[str, Path]] = None,
    weights: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    project_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Train a YOLO segmentation model on the crack dataset.
    
    Args:
        data_yaml: Path to dataset YAML configuration file.
        weights: Pretrained model weights.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        imgsz: Square input image size.
        device: Target compute device ('0', 'cpu', etc.).
        project_dir: Directory where runs and checkpoints are saved.
        experiment_name: Name of the experiment folder. Defaults to the model and
            configured dataset names joined with underscores.
        
    Returns:
        Training results object.
    """
    model = get_model(weights)

    data_yaml = str(data_yaml or config.YOLO_DATA_YAML)
    dataset_suffix = "_".join(config.DATASETS)
    experiment_name = experiment_name or f"yolo_seg_{dataset_suffix}"
    epochs = epochs if epochs is not None else config.EPOCHS
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    imgsz = imgsz if imgsz is not None else config.IMG_SIZE[0]
    device = device if device is not None else ("0" if config.DEVICE.type == "cuda" else "cpu")
    project_dir = str(project_dir or (config.CHECKPOINT_DIR / "yolo_runs"))

    print(f"\n--- Starting YOLO Segmentation Training ---")
    print(f"  Model:       {weights or config.YOLO_MODEL_WEIGHTS}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch Size:  {batch_size}")
    print(f"  Image Size:  {imgsz}")
    print(f"  Device:      {device}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project_dir,
        name=experiment_name,
        **kwargs
    )

    run_weights_dir = Path(results.save_dir) / "weights"
    selected_checkpoint = run_weights_dir / "best.pt"
    if not selected_checkpoint.exists():
        raise FileNotFoundError(
            f"Ultralytics did not produce the expected checkpoint: {selected_checkpoint}"
        )

    named_checkpoint = Path(project_dir) / f"{experiment_name}.pt"
    named_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, named_checkpoint)
    print(f"YOLO checkpoint saved to {named_checkpoint}")

    return results


def evaluate_yolo(
    checkpoint_path: Union[str, Path],
    data_yaml: Optional[Union[str, Path]] = None,
    split: str = "test",
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate a trained YOLO segmentation checkpoint on a specific split (val or test).
    
    Returns:
        Dictionary with segmentation and detection evaluation metrics.
    """
    model = get_model(str(checkpoint_path))
    data_yaml = str(data_yaml or config.YOLO_DATA_YAML)
    imgsz = imgsz if imgsz is not None else config.IMG_SIZE[0]
    device = device if device is not None else ("0" if config.DEVICE.type == "cuda" else "cpu")

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
        **kwargs
    )

    # Extract mask / segmentation metrics
    results_dict = {
        "mask_map50": float(metrics.seg.map50) if hasattr(metrics, "seg") else 0.0,
        "mask_map50_95": float(metrics.seg.map) if hasattr(metrics, "seg") else 0.0,
        "box_map50": float(metrics.box.map50) if hasattr(metrics, "box") else 0.0,
        "box_map50_95": float(metrics.box.map) if hasattr(metrics, "box") else 0.0,
    }
    return results_dict


def predict_yolo(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    save: bool = True,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Perform crack segmentation on an input image using a trained YOLO-seg model.
    
    Returns:
        Binary mask as a 2D numpy array with values 0 or 255.
    """
    weights = str(checkpoint_path or config.YOLO_MODEL_WEIGHTS)
    model = get_model(weights)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    results = model.predict(source=img, conf=conf_threshold, save=False)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if results and len(results) > 0 and results[0].masks is not None:
        # masks.data contains tensors of shape (N, H, W)
        masks_data = results[0].masks.data.cpu().numpy()
        import cv2
        for mask in masks_data:
            resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, (resized_mask > 0.5).astype(np.uint8) * 255)

    if save:
        out_file = output_path or f"{Path(image_path).stem}_yolo_prediction.png"
        Image.fromarray(combined_mask).save(out_file)
        print(f"YOLO prediction mask saved to: {out_file}")

    return combined_mask
