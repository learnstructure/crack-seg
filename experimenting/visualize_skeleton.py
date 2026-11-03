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
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import from crack_seg
from crack_seg import config
from crack_seg.data_handlers.dataset_loaders import load_dataset_partitions
from crack_seg.utils.metrics import _skeletonize


def save_figure_unique(fig, save_path):
    """Save a figure to a new path without overwriting an existing file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_path = save_path.with_name(
        f"{save_path.stem}_{timestamp}_{uuid4().hex[:8]}{save_path.suffix or '.png'}"
    )

    try:
        with unique_path.open("xb") as output_file:
            fig.savefig(
                output_file,
                format=unique_path.suffix.lstrip("."),
                dpi=150,
                bbox_inches="tight",
            )
    except Exception:
        unique_path.unlink(missing_ok=True)
        raise

    print(f"Saved to: {unique_path}")
    return unique_path


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
    open_kernel_size=10,
    close_kernel_size=10,
    close_iterations=1,
    close_dilate_iters=None,
    close_erode_iters=None,
    save_path=None,
    show=True,
    max_spur_len=10,
):
    """
    Plot all masks, raw skeletons, Option A (opened mask skeletons), and Option B (closed mask skeletons)
    side by side in one large figure (4 columns).
    """
    n = len(mask_paths)

    d_iters = close_dilate_iters
    e_iters = close_erode_iters
    if isinstance(close_iterations, str) and ("," in close_iterations or ":" in close_iterations):
        sep = "," if "," in close_iterations else ":"
        parts = close_iterations.split(sep)
        if len(parts) == 2:
            if d_iters is None:
                d_iters = int(parts[0].strip())
            if e_iters is None:
                e_iters = int(parts[1].strip())
    elif isinstance(close_iterations, (list, tuple)) and len(close_iterations) == 2:
        if d_iters is None:
            d_iters = int(close_iterations[0])
        if e_iters is None:
            e_iters = int(close_iterations[1])

    if d_iters is None:
        try:
            d_iters = int(close_iterations)
        except (ValueError, TypeError):
            d_iters = 1
    if e_iters is None:
        try:
            e_iters = int(close_iterations)
        except (ValueError, TypeError):
            e_iters = 1
    
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
            dilate_iterations=d_iters,
            erode_iterations=e_iters,
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
        if d_iters != e_iters:
            iter_str = f", dilate={d_iters}, erode={e_iters}"
        elif d_iters > 1:
            iter_str = f", iters={d_iters}"
        else:
            iter_str = ""
        ax4.set_title(
            f"Option B: Morph. Closed ({close_kernel_size}x{close_kernel_size}{iter_str})\n"
            f"Length: {close_sum:,} px | Bridged: {bridged_b_sum:,} px ({pct_close_diff:+.1f}%)", 
            fontsize=11, pad=8
        )
        ax4.axis('off')
    
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, wspace=0.15, hspace=0.25)
    
    if save_path is None:
        output_dir = Path(__file__).resolve().parent / "skeleton_visualizations"
        save_path = output_dir / "skeleton_comparison_all.png"

    save_figure_unique(fig, save_path)
    
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
    parser.add_argument(
        "--close-iterations",
        type=str,
        default="1",
        help="Option B: number of closing iterations (e.g. 1, 2, or '5,1' for dilate 5, erode 1, default 1)",
    )
    parser.add_argument(
        "--close-dilate-iters",
        "--dilate-iterations",
        type=int,
        default=None,
        help="Option B: number of dilation iterations before erosion (default: same as --close-iterations)",
    )
    parser.add_argument(
        "--close-erode-iters",
        "--erode-iterations",
        type=int,
        default=None,
        help="Option B: number of erosion iterations after dilation (default: same as --close-iterations)",
    )
    parser.add_argument("--max-spur-len", type=int, default=10, help="Legacy spur pruning length (default 10)")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Base path for the figure (a unique filename is always used; default folder: experimenting/skeleton_visualizations)",
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
        output_dir = Path(__file__).resolve().parent / "skeleton_visualizations"
        save_path = output_dir / "skeleton_comparison_all.png"

    # Resolve dilation and erosion iterations
    d_iters = args.close_dilate_iters
    e_iters = args.close_erode_iters
    if "," in str(args.close_iterations) or ":" in str(args.close_iterations):
        sep = "," if "," in str(args.close_iterations) else ":"
        parts = str(args.close_iterations).split(sep)
        if len(parts) == 2:
            if d_iters is None:
                d_iters = int(parts[0].strip())
            if e_iters is None:
                e_iters = int(parts[1].strip())
    if d_iters is None:
        try:
            d_iters = int(args.close_iterations)
        except (ValueError, TypeError):
            d_iters = 1
    if e_iters is None:
        try:
            e_iters = int(args.close_iterations)
        except (ValueError, TypeError):
            e_iters = 1

    plot_skeleton_comparison(
        mask_pairs,
        open_kernel_size=args.open_kernel_size,
        close_kernel_size=args.close_kernel_size,
        close_iterations=args.close_iterations,
        close_dilate_iters=d_iters,
        close_erode_iters=e_iters,
        max_spur_len=args.max_spur_len,
        save_path=save_path,
        show=not args.no_show
    )
    
    # Print stats table
    if d_iters != e_iters:
        opt_b_label = f"Opt B (D{d_iters}:E{e_iters})"
        iter_desc = f"dilate={d_iters}, erode={e_iters}"
    elif d_iters > 1:
        opt_b_label = f"Opt B (iters={d_iters})"
        iter_desc = f"iters={d_iters}"
    else:
        opt_b_label = "Opt B (Close)"
        iter_desc = "iters=1"

    print(f"\n--- Skeleton Comparison: Raw vs Option A (Open {args.open_kernel_size}px) vs Option B (Close {args.close_kernel_size}px, {iter_desc}) ---")
    print(f"{'Mask Name':<35} | {'Raw':>8} | {'Opt A (Open)':>14} | {opt_b_label:>16} | {'Close Diff':>16}")
    print("-" * 100)
    for img_path, mask_path in mask_pairs:
        mask = load_mask_as_binary(mask_path)
        skel_raw = _skeletonize(mask, prune_spurs=False, morph_op=None)
        skel_open = _skeletonize(mask, prune_spurs=False, morph_op="open", morph_kernel_size=args.open_kernel_size, morph_iterations=1)
        skel_close = _skeletonize(
            mask,
            prune_spurs=False,
            morph_op="close",
            morph_kernel_size=args.close_kernel_size,
            morph_iterations=args.close_iterations,
            dilate_iterations=d_iters,
            erode_iterations=e_iters,
        )
        diff_close = int(skel_close.sum()) - int(skel_raw.sum())
        pct_diff = (diff_close / max(int(skel_raw.sum()), 1)) * 100.0
        print(f"{mask_path.name:<35} | {skel_raw.sum():>6d} px | {skel_open.sum():>12d} px | {skel_close.sum():>14d} px | {diff_close:>+7d} px ({pct_diff:+4.1f}%)")


if __name__ == "__main__":
    main()


# ==============================================================================
# Example Command-Line Usages:
# ==============================================================================
# 1. Run with defaults (saves a uniquely named image in experimenting/skeleton_visualizations):
#    python experimenting/visualize_skeleton.py
#
# 2. Visualize a different number of masks (e.g., 10 masks):
#    python experimenting/visualize_skeleton.py --num-masks 10
#
# 3. Multiple symmetric closing iterations (e.g., 2 iterations: dilate 2, erode 2):
#    python experimenting/visualize_skeleton.py --close-iterations 2 --close-kernel-size 3
#
# 4. Asymmetric Dilation-Erosion (dilate 5 times, erode 1 time at the end):
#    python experimenting/visualize_skeleton.py --dilate-iterations 5 --erode-iterations 1
#    or:
#    python experimenting/visualize_skeleton.py --close-iterations 5,1
#    python experimenting/visualize_skeleton.py --close-kernel-size 10 --open-kernel-size 10 --dilate-iterations 5 --erode-iterations 1

# 5. Save output image using a custom base path (a unique filename is still used):
#    python experimenting/visualize_skeleton.py --save-path experimenting/skeleton_comparison.png --no-show
#
# 6. Run headless without interactive GUI popup window (only save to disk):
#    python experimenting/visualize_skeleton.py --no-show
# ==============================================================================
