# ==============================================================================
# Crack Segmentation — Model Evaluation Script
#
# Evaluates trained UNet variants and YOLO on the held-out test split and
# prints a comparison table of standard + skeleton-based IoU metrics.
# ==============================================================================

# --- Standard library ---
import functools
from pathlib import Path

# --- Third-party ---
import numpy as np
import torch
from PIL import Image

# --- crack_seg project ---
from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import get_train_val_test_datasets
from crack_seg.data_handlers.transforms import train_transform, val_transform
from crack_seg.models.yolo_seg import get_model as get_yolo_model
from crack_seg.utils.helpers import load_model
from crack_seg.utils.metrics import (
    iou_score,
    dice_coefficient,
    pixel_accuracy,
    precision_score,
    recall_score,
    specificity_score,
    skeleton_iou_score,
    cldice_score,
    centerline_f1,
    centerline_iou,
    centerline_precision,
    centerline_recall,
    crack_orientation_mae,
)

import argparse

parser = argparse.ArgumentParser(description="Evaluate crack segmentation models.")
parser.add_argument(
    "--skeleton-method",
    type=str,
    choices=["lee", "zhang"],
    default=getattr(config, "SKELETON_METHOD", "lee"),
    help="Centerline skeletonization algorithm: 'lee' (default) or 'zhang' (Zhang-Suen).",
)
parser.add_argument(
    "--tolerance-px",
    type=float,
    default=getattr(config, "CENTERLINE_TOLERANCE_PX", 20.0),
    help="Euclidean tolerance buffer in pixels for centerline metrics (default 20.0).",
)
parser.add_argument(
    "--prune-spurs",
    action="store_true",
    default=getattr(config, "PRUNE_SKELETON_SPURS", False),
    help="Option B: Prune dead-end skeleton spurs connected to junctions.",
)
parser.add_argument(
    "--max-spur-len",
    type=int,
    default=getattr(config, "MAX_SPUR_LEN", 10),
    help="Maximum length of dead-end spurs to prune in pixels (default 10).",
)
parser.add_argument(
    "--morph-open",
    action="store_true",
    default=getattr(config, "MORPH_OPEN_MASK", False),
    help="Option A: Apply morphological opening (erode then dilate) on mask before skeletonization.",
)
parser.add_argument(
    "--open-kernel-size",
    type=int,
    default=getattr(config, "MORPH_OPEN_KERNEL_SIZE", 3),
    help="Kernel size for morphological opening (default 3).",
)
parser.add_argument(
    "--morph-op",
    type=str,
    default=getattr(config, "MORPH_OP", None),
    help="Morphological operation on mask before thinning: 'open', 'close', 'close_open', 'open_close', 'dilate', 'erode'.",
)
parser.add_argument(
    "--morph-kernel-size",
    type=int,
    default=getattr(config, "MORPH_KERNEL_SIZE", 3),
    help="Structuring element kernel size in pixels (default 3).",
)
parser.add_argument(
    "--morph-iterations",
    type=str,
    default=getattr(config, "MORPH_ITERATIONS", 1),
    help="Number of morphological iterations (e.g. 1, 2, or '5,1' for dilate 5, erode 1).",
)
parser.add_argument(
    "--morph-dilate-iters",
    "--dilate-iterations",
    type=int,
    default=getattr(config, "MORPH_DILATE_ITERATIONS", None),
    help="Number of dilation iterations for morphological closing (default: same as morph_iterations).",
)
parser.add_argument(
    "--morph-erode-iters",
    "--erode-iterations",
    type=int,
    default=getattr(config, "MORPH_ERODE_ITERATIONS", None),
    help="Number of erosion iterations for morphological closing (default: same as morph_iterations).",
)
cli_args, _ = parser.parse_known_args()
config.SKELETON_METHOD = cli_args.skeleton_method
config.CENTERLINE_TOLERANCE_PX = cli_args.tolerance_px
config.PRUNE_SKELETON_SPURS = cli_args.prune_spurs
config.MAX_SPUR_LEN = cli_args.max_spur_len
config.MORPH_OPEN_MASK = cli_args.morph_open
config.MORPH_OPEN_KERNEL_SIZE = cli_args.open_kernel_size
config.MORPH_OP = cli_args.morph_op
config.MORPH_KERNEL_SIZE = cli_args.morph_kernel_size
config.MORPH_ITERATIONS = cli_args.morph_iterations
config.MORPH_DILATE_ITERATIONS = cli_args.morph_dilate_iters
config.MORPH_ERODE_ITERATIONS = cli_args.morph_erode_iters

# ==============================================================================
# 0. Environment Info
# ==============================================================================
print("PyTorch version  :", torch.__version__)
print("CUDA available   :", torch.cuda.is_available())
print("Device (config)  :", config.DEVICE)
print(f"Skeleton Method  : {config.SKELETON_METHOD}")
print(f"Centerline Tol   : ±{config.CENTERLINE_TOLERANCE_PX} px")
print(f"Prune Spurs      : {config.PRUNE_SKELETON_SPURS} (max len: {config.MAX_SPUR_LEN} px)")
if config.MORPH_DILATE_ITERATIONS is not None or config.MORPH_ERODE_ITERATIONS is not None:
    d_i = config.MORPH_DILATE_ITERATIONS if config.MORPH_DILATE_ITERATIONS is not None else config.MORPH_ITERATIONS
    e_i = config.MORPH_ERODE_ITERATIONS if config.MORPH_ERODE_ITERATIONS is not None else config.MORPH_ITERATIONS
    print(f"Morph Op         : {config.MORPH_OP} (ksize: {config.MORPH_KERNEL_SIZE} px, dilate={d_i}, erode={e_i})")
else:
    print(f"Morph Op         : {config.MORPH_OP} (ksize: {config.MORPH_KERNEL_SIZE} px, iters: {config.MORPH_ITERATIONS})")
print()

# ==============================================================================
# 1. Model Checkpoint Paths
# Add or remove entries here to control which models are evaluated.
# ==============================================================================
MODELS_TO_EVALUATE = {
    'UNet (NCCD-PF)': {
        'type': 'unet',
        'path': config.CHECKPOINT_DIR / 'unet_NCCD-PF_Dataset.pth',
    },
    'UNet (CCon+NCCD)': {
        'type': 'unet',
        'path': config.CHECKPOINT_DIR / 'unet_CConCrack_NCCD-PF_Dataset.pth',
    },
    'UNet (4 Datasets)': {
        'type': 'unet',
        'path': config.CHECKPOINT_DIR / 'unet_CConCrack_NCCD-PF_Dataset_DeepCrack_CRACK500.pth',
    },
    'YOLO-seg (NCCD)': {
        'type': 'yolo',
        'path': config.CHECKPOINT_DIR / 'yolo_runs' / 'yolo_seg_NCCD-PF_Dataset.pt',
    },
    'YOLO-seg (4 Datasets)': {
        'type': 'yolo',
        'path': config.CHECKPOINT_DIR / 'yolo_runs' / 'yolo_seg_CConCrack_NCCD-PF_Dataset_DeepCrack_CRACK500.pt',
    },
}

# Discover which checkpoints actually exist on disk
active_models = {}
for model_name, info in MODELS_TO_EVALUATE.items():
    ckpt_path = Path(info['path'])
    if ckpt_path.exists():
        active_models[model_name] = info
        print(f"  ✓  Found  : {model_name}  ({ckpt_path.name})")
    else:
        # Check if a loss-suffixed version exists (e.g. _dice.pth, _focal.pth, _dice_focal.pth)
        stem = ckpt_path.stem
        found_loss_ckpt = None
        for suffix in [f"_{config.LOSS}", "_focal", "_dice_focal", "_dice"]:
            candidate = ckpt_path.parent / f"{stem}{suffix}{ckpt_path.suffix}"
            if candidate.exists():
                found_loss_ckpt = candidate
                break
        if found_loss_ckpt is not None:
            active_models[f"{model_name} ({found_loss_ckpt.stem.split('_')[-1]})"] = {
                'type': info['type'],
                'path': found_loss_ckpt,
            }
            print(f"  ✓  Found  : {model_name}  ({found_loss_ckpt.name})")
        else:
            print(f"  ✗  Missing: {model_name}  ({ckpt_path.name}) — skipping")

# Auto-discover any additional trained PyTorch models (e.g. segformer or new loss variants)
for pt_file in sorted(config.CHECKPOINT_DIR.glob("*.pth")):
    if any(m['path'].resolve() == pt_file.resolve() for m in active_models.values()):
        continue
    stem_lower = pt_file.stem.lower()
    inferred_type = None
    for arch in ["segformer", "deeplabv3plus", "deeplabv3", "unetplusplus", "unet", "segnet", "fpn", "linknet", "pspnet"]:
        if stem_lower.startswith(arch):
            inferred_type = arch
            break
    if inferred_type:
        active_models[pt_file.stem] = {'type': inferred_type, 'path': pt_file}
        print(f"  ✓  Discovered: {pt_file.stem}  ({pt_file.name})")

print()
if not active_models:
    raise FileNotFoundError("No valid model checkpoints found. Check CHECKPOINT_DIR in config.")

# ==============================================================================
# 2. Build Test Split
# ==============================================================================
train_ds, _, test_ds = get_train_val_test_datasets(
    train_transform=train_transform,
    val_transform=val_transform,
    dataset_names=config.DATASETS,
    data_root=config.DATA_ROOT,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    seed=config.SPLIT_SEED,
    stratified=config.STRATIFIED_SPLIT,
)

# Sanity-check: no test images leaked into training
train_paths = {img.resolve() for img, _ in train_ds.samples}
test_paths  = {img.resolve() for img, _ in test_ds.samples}
overlap = train_paths & test_paths
if overlap:
    raise RuntimeError(
        f"Evaluation aborted: {len(overlap)} test image(s) also appear in training data."
    )
print(f"Test set: {len(test_ds)} unseen images (no training overlap confirmed)\n")

# ==============================================================================
# 3. Load Active Models into Memory
# ==============================================================================
loaded_models = {}
for model_name, info in active_models.items():
    if info['type'] == 'yolo':
        loaded_models[model_name] = ('yolo', get_yolo_model(str(info['path'])))
    else:
        loaded_models[model_name] = (
            'unet', load_model('unet', info['path'], device=config.DEVICE)
        )

# ==============================================================================
# 4. Define Metrics
#
# Metrics are split into three complementary categories:
#   1. Standard Area-based Metrics: IoU, Dice, Accuracy, Precision, Recall, Specificity
#   2. Topology & Centerline Metrics (Width-Independent):
#        - clDice: Centerline Dice (CVPR 2021)
#        - Centerline_Prec@20: % of predicted crack length within 20px bound of true crack
#        - Centerline_Rec@20: % of true crack length detected within 20px bound
#        - Centerline_F1@20: Harmonic mean of centerline precision & recall
#        - Centerline_IoU@20: Length-based Jaccard Index
#        - Orientation_MAE: Mean Angular Error in degrees between matched cracks
#   3. Skeleton Dilated IoU (Legacy):
#        - tol=1 (strict ±1 px), tol=3 (standard ±3 px), tol=6 (lenient ±6 px), tol=20 (±20 px)
# ==============================================================================
FLOAT_METRIC_NAMES = {
    'clDice',
    'Centerline_Prec@20',
    'Centerline_Rec@20',
    'Centerline_F1@20',
    'Centerline_IoU@20',
    'Orientation_MAE(deg)',
    'skel_IoU(tol=1)',
    'skel_IoU(tol=3)',
    'skel_IoU(tol=6)',
    'skel_IoU(tol=20)',
}

metric_functions = {
    # --- Standard area-based metrics (return torch.Tensor) ---
    'IoU'                  : iou_score,
    'Dice'                 : dice_coefficient,
    'Accuracy'             : pixel_accuracy,
    'Precision'            : precision_score,
    'Recall'               : recall_score,
    'Specificity'          : specificity_score,

    # --- Topology & Centerline Metrics (Width-Independent) ---
    'clDice'               : cldice_score,
    'Centerline_Prec@20'   : functools.partial(centerline_precision, tolerance_px=20.0),
    'Centerline_Rec@20'    : functools.partial(centerline_recall, tolerance_px=20.0),
    'Centerline_F1@20'     : functools.partial(centerline_f1, tolerance_px=20.0),
    'Centerline_IoU@20'    : functools.partial(centerline_iou, tolerance_px=20.0),
    'Orientation_MAE(deg)' : crack_orientation_mae,

    # --- Skeleton Dilated IoU (Legacy) ---
    'skel_IoU(tol=1)'      : functools.partial(skeleton_iou_score, tolerance_px=1),
    'skel_IoU(tol=3)'      : functools.partial(skeleton_iou_score, tolerance_px=3),
    'skel_IoU(tol=6)'      : functools.partial(skeleton_iou_score, tolerance_px=6),
    'skel_IoU(tol=20)'     : functools.partial(skeleton_iou_score, tolerance_px=20),
}

# ==============================================================================
# 5. Run Evaluation
# ==============================================================================
all_model_metrics = {
    model_name: {metric_name: [] for metric_name in metric_functions}
    for model_name in loaded_models
}

print("Running evaluation...")
with torch.no_grad():
    for sample_index in range(len(test_ds)):
        image_tensor, target = test_ds[sample_index]
        image_path, _ = test_ds.samples[sample_index]

        for model_name, (m_type, model_obj) in loaded_models.items():

            # --- Get prediction tensor ---
            if m_type == 'unet':
                output = model_obj(image_tensor.unsqueeze(0).to(config.DEVICE))
                prediction = torch.sigmoid(output).squeeze(0).cpu()   # (1, H, W)

            elif m_type == 'yolo':
                yolo_image = Image.open(image_path).convert('RGB')
                yolo_result = model_obj.predict(
                    source=yolo_image,
                    conf=0.25,
                    imgsz=config.IMG_SIZE[0],
                    verbose=False,
                )[0]
                # Merge all instance masks into a single probability map
                yolo_mask = np.zeros(
                    (yolo_image.height, yolo_image.width), dtype=np.float32
                )
                if yolo_result.masks is not None:
                    for inst_mask in yolo_result.masks.data.cpu().numpy():
                        inst_img = Image.fromarray(
                            (inst_mask > 0.5).astype(np.uint8) * 255
                        ).resize(
                            (yolo_image.width, yolo_image.height),
                            Image.Resampling.NEAREST,
                        )
                        yolo_mask = np.maximum(
                            yolo_mask, np.asarray(inst_img) / 255.0
                        )
                # Resize to match the target shape used by the dataset
                th, tw = target.shape[-2:]
                yolo_pred_img = Image.fromarray(
                    (yolo_mask * 255).astype(np.uint8)
                ).resize((tw, th), Image.Resampling.NEAREST)
                prediction = (
                    torch.from_numpy(np.asarray(yolo_pred_img) / 255.0)
                    .unsqueeze(0)
                    .float()
                )  # (1, H, W)

            # --- Compute each metric ---
            for m_name, m_fn in metric_functions.items():
                if m_name in FLOAT_METRIC_NAMES:
                    score = m_fn(prediction, target)
                else:
                    score = m_fn(prediction, target).item()
                all_model_metrics[model_name][m_name].append(score)

        # Progress indicator every 200 samples
        if (sample_index + 1) % 200 == 0:
            print(f"  ... {sample_index + 1}/{len(test_ds)} samples done")

print(f"  ... {len(test_ds)}/{len(test_ds)} samples done\n")

# ==============================================================================
# 6. Print Comparison Table
# ==============================================================================
mean_metrics = {
    model_name: {
        metric_name: float(np.mean(values))
        for metric_name, values in metrics.items()
    }
    for model_name, metrics in all_model_metrics.items()
}

COL_W  = 20   # width per model column
NAME_W = 24   # width for metric name column

header = f"{'Metric':<{NAME_W}}" + "".join(
    f"{name:>{COL_W}}" for name in loaded_models
)
sep  = "=" * len(header)
dash = "-" * len(header)

print(f"Evaluation Results — {len(test_ds)} Unseen Test Samples")
print(sep)
print(header)

area_metrics = ['IoU', 'Dice', 'Accuracy', 'Precision', 'Recall', 'Specificity']
topo_metrics = [
    'clDice',
    'Centerline_Prec@20',
    'Centerline_Rec@20',
    'Centerline_F1@20',
    'Centerline_IoU@20',
    'Orientation_MAE(deg)',
]
skel_metrics = ['skel_IoU(tol=1)', 'skel_IoU(tol=3)', 'skel_IoU(tol=6)', 'skel_IoU(tol=20)']

print(dash)
print(f"{'--- 1. Area-based ---':<{NAME_W}}")
print(dash)
for m_name in area_metrics:
    row = f"{m_name:<{NAME_W}}"
    for model_name in loaded_models:
        row += f"{mean_metrics[model_name][m_name]:>{COL_W}.4f}"
    print(row)

print(dash)
print(f"{'--- 2. Topology & Centerline (Width-Independent) ---':<{NAME_W}}")
print(dash)
for m_name in topo_metrics:
    row = f"{m_name:<{NAME_W}}"
    for model_name in loaded_models:
        row += f"{mean_metrics[model_name][m_name]:>{COL_W}.4f}"
    print(row)

print(dash)
print(f"{'--- 3. Skeleton Dilated IoU (Legacy) ---':<{NAME_W}}")
print(dash)
for m_name in skel_metrics:
    row = f"{m_name:<{NAME_W}}"
    for model_name in loaded_models:
        row += f"{mean_metrics[model_name][m_name]:>{COL_W}.4f}"
    print(row)

print(sep)
print("Metric Notes:")
print("  • Centerline_*@20 : Evaluates crack length within 20px Euclidean distance of true crack.")
print("  • clDice          : Centerline Dice (CVPR 2021) measuring topological connectivity.")
print("  • Orientation_MAE : Mean Angular Error in degrees (lower is better, 0° = perfect alignment).")
print("  • skel_IoU(tol=N) : Dilated skeleton area IoU (tol=20 is ±20px dilation).")
