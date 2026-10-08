#!/usr/bin/env python
"""
Visualize skeletonization comparisons on random masks from datasets:
Col 1: Binary Mask
Col 2: Raw Skeleton
Col 3: Option A — Morphological Opening (Erosion -> Dilation)
Col 4: Option B — Morphological Closing (Dilation -> Erosion, multi-iteration)

Run: python experimenting/visualize_skeleton.py
"""
import os
import sys

# Ensure conda env Library/bin DLLs (OpenBLAS, OpenMP, etc.) are discoverable on Windows
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_bin)
            except Exception:
                pass
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ.get("PATH", "")

import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import from crack_seg
from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import load_dataset_partitions
from crack_seg.utils.metrics import _skeletonize


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


def plot_skeleton_comparison(
    mask_paths,
    open_kernel_size=3,
    close_kernel_size=3,
    close_iterations=1,
    save_path=None,
    show=True,
    max_spur_len=10,
):
    """
    Plot all masks, raw skeletons, Option A (opened mask skeletons), and Option B (closed mask skeletons)
    side by side in one large figure (4 columns).
    """
    n = len(mask_paths)
    
    fig, axes = plt.subplots(n, 4, figsize=(26, 4.5 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    
    for idx, (img_path, mask_path) in enumerate(mask_paths):
        mask = load_mask_as_binary(mask_path)
        
        # 1. Raw skeleton
        skel_raw = _skeletonize(mask, prune_spurs=False, morph_op=None)
        # 2. Option A: Morphological opening on mask before skeletonization (Erosion then Dilation)
        skel_open = _skeletonize(mask, prune_spurs=False, morph_op="open", morph_kernel_size=open_kernel_size, morph_iterations=1)
        # 3. Option B: Morphological closing on mask before skeletonization (Dilation then Erosion)
        skel_close = _skeletonize(
            mask,
            prune_spurs=False,
            morph_op="close",
            morph_kernel_size=close_kernel_size,
            morph_iterations=close_iterations,
        )
        # Bridged pixels (present in closed skeleton but not raw)
        bridged_b = np.logical_and(skel_close, np.logical_not(skel_raw)).astype(np.uint8)
        
        raw_sum = int(skel_raw.sum())
        open_sum = int(skel_open.sum())
        close_sum = int(skel_close.sum())
        bridged_b_sum = int(bridged_b.sum())
        
        # Col 1: Original mask
        ax1 = axes[idx, 0]
        ax1.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f"Mask: {mask_path.name}\n(area: {mask.sum():,} px)", fontsize=11, pad=8)
        ax1.axis('off')
        
        # Col 2: Raw Skeleton (red)
        ax2 = axes[idx, 1]
        overlay_raw = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay_raw[mask > 0] = [200, 200, 200]  # light gray for mask
        overlay_raw[skel_raw > 0] = [255, 0, 0]  # red for raw skeleton
        ax2.imshow(overlay_raw)
        ax2.set_title(f"Raw Skeleton (red)\nLength: {raw_sum:,} px", fontsize=11, pad=8)
        ax2.axis('off')
        
        # Col 3: Option A — Morphological Opening (cyan)
        ax3 = axes[idx, 2]
        overlay_open = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay_open[mask > 0] = [200, 200, 200]
        overlay_open[skel_open > 0] = [0, 180, 255]  # cyan for opened skeleton
        ax3.imshow(overlay_open)
        pct_open_diff = ((open_sum - raw_sum) / max(raw_sum, 1)) * 100.0
        ax3.set_title(
            f"Option A: Morph. Opened ({open_kernel_size}x{open_kernel_size})\nLength: {open_sum:,} px ({pct_open_diff:+.1f}%)",
            fontsize=11, pad=8
        )
        ax3.axis('off')

        # Col 4: Option B — Morphological Closing (green + orange)
        ax4 = axes[idx, 3]
        overlay_close = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay_close[mask > 0] = [200, 200, 200]           # light gray for mask
        overlay_close[skel_close > 0] = [0, 200, 80]        # green for closed skeleton
        overlay_close[bridged_b > 0] = [255, 140, 0]        # orange for newly bridged connections
        ax4.imshow(overlay_close)
        pct_close_diff = ((close_sum - raw_sum) / max(raw_sum, 1)) * 100.0
        iter_str = f", iters={close_iterations}" if close_iterations > 1 else ""
        ax4.set_title(
            f"Option B: Morph. Closed ({close_kernel_size}x{close_kernel_size}{iter_str})\n"
            f"Length: {close_sum:,} px | Bridged: {bridged_b_sum:,} px ({pct_close_diff:+.1f}%)", 
            fontsize=11, pad=8
        )
        ax4.axis('off')
    
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.15, hspace=0.25)
    
    if save_path is None:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "skeleton_comparison_all.png"
    elif save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    if show:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() not in ("agg", "cairo", "pdf", "ps", "svg"):
                plt.show()
            else:
                plt.close()
        except Exception:
            plt.close()
    else:
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize crack skeletons comparing Raw vs Opening vs Closing.")
    parser.add_argument("--num-masks", type=int, default=20, help="Number of masks to visualize (default 20)")
    parser.add_argument("--open-kernel-size", type=int, default=3, help="Option A: morphological open kernel size in px (default 3)")
    parser.add_argument("--close-kernel-size", type=int, default=3, help="Option B: morphological close kernel size in px (default 3)")
    parser.add_argument("--close-iterations", type=int, default=1, help="Option B: number of closing iterations (dilate N times then erode N times, default 1)")
    parser.add_argument("--max-spur-len", type=int, default=10, help="Legacy spur pruning length (default 10)")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save figure (default: experimenting/output/skeleton_comparison_all.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive window (only save to disk)")
    args = parser.parse_args()

    print("Loading random masks from datasets...")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Datasets: {config.DATASETS}")
    
    # Load random masks
    mask_pairs = load_random_masks(num_masks=args.num_masks)
    print(f"\nSelected {len(mask_pairs)} random masks")
    
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "skeleton_comparison_all.png"

    plot_skeleton_comparison(
        mask_pairs,
        open_kernel_size=args.open_kernel_size,
        close_kernel_size=args.close_kernel_size,
        close_iterations=args.close_iterations,
        max_spur_len=args.max_spur_len,
        save_path=save_path,
        show=not args.no_show
    )
    
    # Print stats table
    print(f"\n--- Skeleton Comparison: Raw vs Option A (Open {args.open_kernel_size}px) vs Option B (Close {args.close_kernel_size}px, iters={args.close_iterations}) ---")
    print(f"{'Mask Name':<35} | {'Raw':>8} | {'Opt A (Open)':>14} | {'Opt B (Close)':>14} | {'Close Diff':>16}")
    print("-" * 98)
    for img_path, mask_path in mask_pairs:
        mask = load_mask_as_binary(mask_path)
        skel_raw = _skeletonize(mask, prune_spurs=False, morph_op=None)
        skel_open = _skeletonize(mask, prune_spurs=False, morph_op="open", morph_kernel_size=args.open_kernel_size, morph_iterations=1)
        skel_close = _skeletonize(mask, prune_spurs=False, morph_op="close", morph_kernel_size=args.close_kernel_size, morph_iterations=args.close_iterations)
        diff_close = int(skel_close.sum()) - int(skel_raw.sum())
        pct_diff = (diff_close / max(int(skel_raw.sum()), 1)) * 100.0
        print(f"{mask_path.name:<35} | {skel_raw.sum():>6d} px | {skel_open.sum():>12d} px | {skel_close.sum():>12d} px | {diff_close:>+7d} px ({pct_diff:+4.1f}%)")


if __name__ == "__main__":
    main()


# ==============================================================================
# Example Command-Line Usages:
# ==============================================================================
# 1. Run with defaults (saves to experimenting/output/skeleton_comparison_all.png):
#    python experimenting/visualize_skeleton.py
#
# 2. Visualize a different number of masks (e.g., 10 masks):
#    python experimenting/visualize_skeleton.py --num-masks 10
#
# 3. Multiple closing iterations (e.g., 2 or 3 iterations of dilation followed by erosion):
#    python experimenting/visualize_skeleton.py --close-iterations 2 --close-kernel-size 3
#
# 4. Save output image to a custom path or filename:
#    python experimenting/visualize_skeleton.py --save-path experimenting/skeleton_comparison.png --no-show
#
# 5. Run headless without interactive GUI popup window (only save to disk):
#    python experimenting/visualize_skeleton.py --no-show
# ==============================================================================
