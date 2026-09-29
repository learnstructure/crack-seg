#!/usr/bin/env python
"""
Visualize skeletonization on 20 random masks from any dataset.
Run: python visualize_skeleton.py
"""
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import from crack_seg
from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import load_dataset_partitions
from crack_seg.utils.metrics import _skeletonize, skeleton_iou_score


def load_random_masks(num_masks=20, dataset_names=None):
    """Load random masks from specified datasets."""
    if dataset_names is None:
        dataset_names = config.DATASETS  # Uses all 4 datasets by default
    
    all_pairs = []
    for ds_name in dataset_names:
        try:
            partitions = load_dataset_partitions(ds_name, data_root=config.DATA_ROOT)
            for split_name, pairs in partitions.items():
                all_pairs.extend(pairs)
            print(f"  {ds_name}: {len(all_pairs)} total pairs loaded")
        except Exception as e:
            print(f"  {ds_name}: Failed to load - {e}")
    
    if not all_pairs:
        raise RuntimeError("No image-mask pairs found!")
    
    # Sample random pairs
    selected = random.sample(all_pairs, min(num_masks, len(all_pairs)))
    return selected


def load_mask_as_binary(mask_path, threshold=128):
    """Load mask image and convert to binary (0/1)."""
    mask_img = Image.open(mask_path).convert('L')  # grayscale
    mask_arr = np.array(mask_img)
    # Binarize
    if mask_arr.max() > 1:
        mask_arr = (mask_arr > threshold).astype(np.uint8)
    else:
        mask_arr = (mask_arr > 0.5).astype(np.uint8)
    return mask_arr


def plot_skeleton_comparison(mask_paths, save_path=None):
    """Plot all masks and skeletons side by side in one large figure (2 columns: mask | skeleton)."""
    n = len(mask_paths)
    
    # 2 columns (mask, skeleton), n rows
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    
    for idx, (img_path, mask_path) in enumerate(mask_paths):
        # Load mask
        mask = load_mask_as_binary(mask_path)
        
        # Compute skeleton
        skel = _skeletonize(mask)
        
        # Left: Original mask
        ax1 = axes[idx, 0]
        ax1.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f"Mask: {mask_path.name}", fontsize=12, pad=10)
        ax1.axis('off')
        
        # Right: Skeleton overlay
        ax2 = axes[idx, 1]
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask > 0] = [200, 200, 200]  # light gray for mask
        overlay[skel > 0] = [255, 0, 0]      # red for skeleton
        ax2.imshow(overlay)
        ax2.set_title(f"Skeleton (red) | skel={skel.sum()}px mask={mask.sum()}px ratio={skel.sum()/max(mask.sum(),1):.4f}", 
                      fontsize=12, pad=10)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def main():
    print("Loading random masks from datasets...")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Datasets: {config.DATASETS}")
    
    # Load 20 random masks
    mask_pairs = load_random_masks(num_masks=20)
    print(f"\nSelected {len(mask_pairs)} random masks")
    
    # Plot all in one figure: 2 columns (mask | skeleton) × 20 rows
    save_path = Path("skeleton_comparison_all.png")
    plot_skeleton_comparison(mask_pairs, save_path=save_path)
    
    # Also print some stats
    print("\n--- Skeleton Statistics ---")
    for img_path, mask_path in mask_pairs[:5]:
        mask = load_mask_as_binary(mask_path)
        skel = _skeletonize(mask)
        print(f"{mask_path.name}: mask={mask.sum()}px, skel={skel.sum()}px, ratio={skel.sum()/max(mask.sum(),1):.4f}")


if __name__ == "__main__":
    main()