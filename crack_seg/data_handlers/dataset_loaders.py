import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Sequence
from PIL import Image
import numpy as np

from crack_seg import config
from crack_seg.data_handlers.dataset import CrackDataset

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in VALID_IMG_EXTS


def load_cconcrack(dataset_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Load image and mask pairs from the CConCrack dataset.
    Scans both Train and Test folders.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"CConCrack directory not found: {dataset_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for split_dir_name in ["Train", "Test", "Validation"]:
        split_dir = dataset_dir / split_dir_name
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        mask_dir = split_dir / "masks"

        if not (img_dir.exists() and mask_dir.exists()):
            continue

        # Map mask filename stem to mask path
        mask_map = {
            m.stem: m for m in mask_dir.iterdir() if m.is_file() and _is_image_file(m)
        }

        for img_path in sorted(img_dir.iterdir()):
            if img_path.is_file() and _is_image_file(img_path):
                if img_path.stem in mask_map:
                    pairs.append((img_path, mask_map[img_path.stem]))

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found for CConCrack in: {dataset_dir}")

    return pairs


def _load_image_mask_dirs(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
) -> List[Tuple[Path, Path]]:
    """Pair image and mask files in two directories by filename stem."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.exists() or not mask_dir.exists():
        return []

    mask_map = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and _is_image_file(path)
    }
    pairs = [
        (image_path, mask_map[image_path.stem])
        for image_path in sorted(image_dir.iterdir())
        if image_path.is_file()
        and _is_image_file(image_path)
        and image_path.stem in mask_map
    ]
    if not pairs:
        raise FileNotFoundError(
            f"No paired image/mask files found.\n"
            f"  image_dir : {image_dir} (exists={image_dir.exists()})\n"
            f"  mask_dir  : {mask_dir} (exists={mask_dir.exists()})"
        )
    return pairs


def load_dataset_partitions(
    dataset_name: str,
    data_root: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Load official partitions without merging them before evaluation.

    Datasets without an official partition are returned under ``all`` and are
    split later with the configured seed and ratios.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    dataset_path = data_root / dataset_name
    key = dataset_name.lower().strip()

    if key == "cconcrack":
        if not dataset_path.exists():
            raise FileNotFoundError(f"CConCrack directory not found: {dataset_path}")
        partitions = {
            "train": _load_image_mask_dirs(dataset_path / "Train" / "images", dataset_path / "Train" / "masks"),
            "val": _load_image_mask_dirs(dataset_path / "Validation" / "images", dataset_path / "Validation" / "masks"),
            "test": _load_image_mask_dirs(dataset_path / "Test" / "images", dataset_path / "Test" / "masks"),
        }
    elif key == "deepcrack":
        if not dataset_path.exists():
            raise FileNotFoundError(f"DeepCrack directory not found: {dataset_path}")
        partitions = {
            "train": _load_image_mask_dirs(dataset_path / "train_img", dataset_path / "train_lab"),
            "test": _load_image_mask_dirs(dataset_path / "test_img", dataset_path / "test_lab"),
        }
        partitions["val"] = []
    elif key == "crack500":
        if not dataset_path.exists():
            raise FileNotFoundError(f"CRACK500 directory not found: {dataset_path}")
        partitions = {"train": [], "val": [], "test": []}
        for partition, folder_names in {
            "train": ("traindata", "traincrop"),
            "val": ("valdata", "valcrop"),
            "test": ("testdata", "testcrop"),
        }.items():
            for folder_name in folder_names:
                # Resolve nested same-name subdirectory (e.g. traindata/traindata/)
                folder = _resolve_crack500_folder(dataset_path, folder_name)
                if folder is None:
                    continue
                # In CRACK500, input photos are .jpg/.jpeg/.JPG; masks are .png
                image_files = [
                    f for f in sorted(folder.iterdir())
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg"] and not f.stem.endswith("_mask")
                ]
                for image_path in image_files:
                    mask_path = next(
                        (
                            candidate
                            for candidate in (
                                folder / f"{image_path.stem}_mask.png",
                                folder / f"{image_path.stem}.png",
                            )
                            if candidate.exists()
                        ),
                        None,
                    )
                    if mask_path is not None:
                        partitions[partition].append((image_path, mask_path))
                    else:
                        raise FileNotFoundError(
                            f"Missing matching mask for CRACK500 image: {image_path}"
                        )
    else:
        partitions = {"all": load_dataset_by_name(dataset_name, data_root=data_root, **kwargs)}

    available = {name: samples for name, samples in partitions.items() if samples}
    if not available:
        raise ValueError(f"No valid image-mask pairs found for dataset '{dataset_name}' at {dataset_path}")
    return available


def load_nccd_pf(
    dataset_dir: Union[str, Path],
    cracked_only: bool = True
) -> List[Tuple[Path, Path]]:
    """
    Load image and mask pairs from NCCD-PF_Dataset.
    Handles 'Dataset_for_semantic_segmentation' folder and 'image_X' <-> 'mask_X' naming.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"NCCD-PF directory not found: {dataset_dir}")

    # Check semantic segmentation subfolder if exists
    sem_seg_dir = dataset_dir / "Dataset_for_semantic_segmentation"
    target_dir = sem_seg_dir if sem_seg_dir.exists() else dataset_dir

    if cracked_only and (target_dir / "Images_cracked").exists() and (target_dir / "Masks_cracked").exists():
        img_dir = target_dir / "Images_cracked"
        mask_dir = target_dir / "Masks_cracked"
    else:
        img_dir = target_dir / "Images"
        mask_dir = target_dir / "Masks"

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError(
            f"Images or Masks directory not found in NCCD-PF dataset at: {target_dir}"
        )

    # Build mask lookup by numeric ID
    id_pattern = re.compile(r"\d+")
    mask_map = {}
    for mask_path in mask_dir.iterdir():
        if mask_path.is_file() and _is_image_file(mask_path):
            match = id_pattern.search(mask_path.stem)
            if match:
                mask_map[match.group(0)] = mask_path

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.is_file() and _is_image_file(img_path):
            match = id_pattern.search(img_path.stem)
            if match and match.group(0) in mask_map:
                mask_path = mask_map[match.group(0)]
                
                # If loading from un-filtered directory and cracked_only is True, verify mask is not blank
                if cracked_only and img_dir.name == "Images":
                    try:
                        with Image.open(mask_path) as m_img:
                            m_arr = np.array(m_img)
                            if not np.any(m_arr):
                                continue
                    except Exception:
                        continue

                pairs.append((img_path, mask_path))

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found for NCCD-PF in: {dataset_dir}")

    return pairs


def load_deepcrack(dataset_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Load image and mask pairs from DeepCrack dataset.
    Scans train_img/train_lab and test_img/test_lab.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"DeepCrack directory not found: {dataset_dir}")

    pairs: List[Tuple[Path, Path]] = []
    subsets = [("train_img", "train_lab"), ("test_img", "test_lab")]

    for img_folder, mask_folder in subsets:
        img_dir = dataset_dir / img_folder
        mask_dir = dataset_dir / mask_folder

        if not (img_dir.exists() and mask_dir.exists()):
            continue

        mask_map = {
            m.stem: m for m in mask_dir.iterdir() if m.is_file() and _is_image_file(m)
        }

        for img_path in sorted(img_dir.iterdir()):
            if img_path.is_file() and _is_image_file(img_path):
                if img_path.stem in mask_map:
                    pairs.append((img_path, mask_map[img_path.stem]))

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found for DeepCrack in: {dataset_dir}")

    return pairs


def _resolve_crack500_folder(dataset_dir: Path, f_name: str) -> Optional[Path]:
    """Resolve a CRACK500 split folder, handling the nested same-name subdirectory.

    CRACK500 is distributed with each split stored in a self-named subfolder:
        CRACK500/traindata/traindata/  (contains the actual files)
    This helper returns the innermost directory that actually contains image
    files, checking both the top-level folder and the nested variant.
    """
    candidates = [
        dataset_dir / f_name / f_name,  # nested: CRACK500/traindata/traindata/
        dataset_dir / f_name,            # flat:   CRACK500/traindata/
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(
            f.is_file() and _is_image_file(f) for f in candidate.iterdir()
        ):
            return candidate
    return None


def load_crack500(dataset_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Load image and mask pairs from CRACK500 dataset.
    Scans traindata/traincrop, valdata/valcrop, testdata/testcrop.

    The CRACK500 distribution stores files in self-named subdirectories
    (e.g. ``CRACK500/traindata/traindata/``).  Images and their binary masks
    share the same filename stem; masks have a ``_mask`` suffix
    (e.g. ``20160222_081011.jpg`` + ``20160222_081011_mask.png``).
    Cropped sub-datasets (``traincrop``, etc.) use same-stem ``.png`` masks.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"CRACK500 directory not found: {dataset_dir}")

    pairs: List[Tuple[Path, Path]] = []
    folder_candidates = [
        "traindata", "traincrop", "valdata", "valcrop", "testdata", "testcrop"
    ]

    for f_name in folder_candidates:
        folder = _resolve_crack500_folder(dataset_dir, f_name)
        if folder is None:
            continue

        # In CRACK500, input photos are .jpg/.jpeg/.JPG; masks are .png
        imgs = [
            f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg"] and not f.stem.endswith("_mask")
        ]
        for img_path in imgs:
            possible_masks = [
                folder / f"{img_path.stem}_mask.png",
                folder / f"{img_path.stem}.png",
            ]
            mask_path = next((m for m in possible_masks if m.exists()), None)
            if mask_path is not None:
                pairs.append((img_path, mask_path))
            else:
                raise FileNotFoundError(
                    f"Missing matching mask for CRACK500 image: {img_path}"
                )

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found for CRACK500 in: {dataset_dir}")

    return pairs


def load_generic_dataset(
    img_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None
) -> List[Tuple[Path, Path]]:
    """
    Generic loader that pairs image files with mask files by matching file stems.
    If mask_dir is None, assumes dataset has 'images' and 'masks' subdirectories.
    """
    img_path = Path(img_dir)
    if mask_dir is None:
        mask_path = img_path / "masks"
        img_path = img_path / "images"
    else:
        mask_path = Path(mask_dir)

    if not (img_path.exists() and mask_path.exists()):
        raise FileNotFoundError(f"Image or mask directory not found: {img_path}, {mask_path}")

    mask_map = {
        m.stem: m for m in mask_path.iterdir() if m.is_file() and _is_image_file(m)
    }

    pairs: List[Tuple[Path, Path]] = []
    for img_f in sorted(img_path.iterdir()):
        if img_f.is_file() and _is_image_file(img_f):
            if img_f.stem in mask_map:
                pairs.append((img_f, mask_map[img_f.stem]))

    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found between:\n"
            f"  images: {img_path}\n"
            f"  masks : {mask_path}"
        )

    return pairs


DATASET_REGISTRY = {
    "cconcrack": load_cconcrack,
    "nccd-pf_dataset": load_nccd_pf,
    "nccd-pf": load_nccd_pf,
    "nccd": load_nccd_pf,
    "deepcrack": load_deepcrack,
    "crack500": load_crack500,
}


def load_dataset_by_name(
    dataset_name: str,
    data_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> List[Tuple[Path, Path]]:
    """
    Load sample pairs for a specified dataset name from data_root.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    key = dataset_name.lower().strip()

    loader_fn = DATASET_REGISTRY.get(key)
    dataset_path = data_root / dataset_name

    if loader_fn:
        if key.startswith("nccd"):
            cracked_only = kwargs.get("cracked_only", config.NCCD_CRACKED_ONLY)
            return loader_fn(dataset_path, cracked_only=cracked_only)
        return loader_fn(dataset_path)
    
    # Fallback to generic folder loader if not registered
    return load_generic_dataset(dataset_path)


def load_multiple_datasets(
    dataset_names: Sequence[str],
    data_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Load samples from a list of dataset names.
    Returns a dictionary mapping dataset_name -> list of (image_path, mask_path).
    """
    data_root = Path(data_root or config.DATA_ROOT)
    dataset_samples: Dict[str, List[Tuple[Path, Path]]] = {}

    for name in dataset_names:
        try:
            samples = load_dataset_by_name(name, data_root=data_root, **kwargs)
            dataset_samples[name] = samples
            print(f"Loaded dataset '{name}': {len(samples)} image-mask pairs.")
        except Exception:
            raise

    return dataset_samples


def create_splits(
    dataset_samples_dict: Dict[str, List[Tuple[Path, Path]]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratified: bool = True
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split samples into disjoint Train, Validation, and Test sets.
    
    Args:
        dataset_samples_dict: Dict of dataset_name -> list of (img_path, mask_path).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.
        stratified: If True, splits each dataset proportionally so all splits are balanced.
        
    Returns:
        (train_samples, val_samples, test_samples)
    """
    # Normalize ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, Path]] = []
    val_samples: List[Tuple[Path, Path]] = []
    test_samples: List[Tuple[Path, Path]] = []

    # Reject duplicate image/mask pairs before splitting so the same sample
    # cannot appear in more than one partition through different dataset entries.
    seen_images = set()
    for samples in dataset_samples_dict.values():
        for image_path, mask_path in samples:
            image_key = Path(image_path).resolve()
            if image_key in seen_images:
                raise ValueError(
                    f"Duplicate image found across dataset inputs: {image_key}"
                )
            seen_images.add(image_key)

    if stratified:
        for ds_name, samples in dataset_samples_dict.items():
            shuffled = list(samples)
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))

            ds_train = shuffled[:n_train]
            ds_val = shuffled[n_train:n_train + n_val]
            ds_test = shuffled[n_train + n_val:]

            train_samples.extend(ds_train)
            val_samples.extend(ds_val)
            test_samples.extend(ds_test)
    else:
        # Pool all samples together before splitting
        all_samples: List[Tuple[Path, Path]] = []
        for samples in dataset_samples_dict.values():
            all_samples.extend(samples)

        rng.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]

    split_sets = [
        {(Path(image).resolve(), Path(mask).resolve()) for image, mask in samples}
        for samples in (train_samples, val_samples, test_samples)
    ]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise RuntimeError("Train, validation, and test splits overlap.")

    return train_samples, val_samples, test_samples


def get_train_val_test_datasets(
    train_transform=None,
    val_transform=None,
    test_transform=None,
    dataset_names: Optional[Sequence[str]] = None,
    data_root: Optional[Union[str, Path]] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    stratified: Optional[bool] = None
) -> Tuple[CrackDataset, CrackDataset, CrackDataset]:
    """
    High-level factory function to build CrackDataset objects for train, val, and test.
    Reads defaults from crack_seg.config if arguments are not provided.
    """
    dataset_names = dataset_names or config.DATASETS
    data_root = data_root or config.DATA_ROOT
    train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
    val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
    test_ratio = test_ratio if test_ratio is not None else config.TEST_RATIO
    seed = seed if seed is not None else config.SPLIT_SEED
    stratified = stratified if stratified is not None else config.STRATIFIED_SPLIT

    train_samples: List[Tuple[Path, Path]] = []
    val_samples: List[Tuple[Path, Path]] = []
    test_samples: List[Tuple[Path, Path]] = []
    unsplit_samples: Dict[str, List[Tuple[Path, Path]]] = {}

    for dataset_name in dataset_names:
        partitions = load_dataset_partitions(dataset_name, data_root=data_root)
        if "all" in partitions:
            unsplit_samples[dataset_name] = partitions["all"]
            continue

        if partitions.get("train") and not partitions.get("val"):
            # No official validation split: carve val out of train proportionally.
            # We use val/(train+val) as the effective val fraction so that the
            # remaining train portion still represents ~train_ratio of all data.
            tv_total = train_ratio + val_ratio
            if tv_total <= 0:
                raise ValueError(
                    f"train_ratio + val_ratio must be > 0, got {tv_total}"
                )
            effective_val_ratio = val_ratio / tv_total
            effective_train_ratio = train_ratio / tv_total
            train_part, val_part, _ = create_splits(
                {dataset_name: partitions["train"]},
                train_ratio=effective_train_ratio,
                val_ratio=effective_val_ratio,
                test_ratio=0.0,
                seed=seed,
                stratified=False,
            )
            if not val_part:
                raise RuntimeError(
                    f"Validation split for '{dataset_name}' is empty after sub-splitting "
                    f"the official train set ({len(partitions['train'])} samples). "
                    f"Increase val_ratio or add more training data."
                )
            partitions["train"] = train_part
            partitions["val"] = val_part

        train_samples.extend(partitions.get("train", []))
        val_samples.extend(partitions.get("val", []))
        test_samples.extend(partitions.get("test", []))

    if unsplit_samples:
        split_train, split_val, split_test = create_splits(
            unsplit_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=stratified,
        )
        train_samples.extend(split_train)
        val_samples.extend(split_val)
        test_samples.extend(split_test)

    split_sets = [
        {(Path(image).resolve(), Path(mask).resolve()) for image, mask in samples}
        for samples in (train_samples, val_samples, test_samples)
    ]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise RuntimeError("Train, validation, and test samples overlap.")

    train_ds = CrackDataset(samples=train_samples, transform=train_transform)
    val_ds = CrackDataset(samples=val_samples, transform=val_transform)
    test_ds = CrackDataset(samples=test_samples, transform=test_transform or val_transform)

    print(
        f"\nDataset Splits Created (Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples):\n"
        f"  - Train:      {len(train_ds)} samples\n"
        f"  - Validation: {len(val_ds)} samples\n"
        f"  - Test:       {len(test_ds)} samples\n"
    )

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print("Inspecting configured datasets in:", config.DATA_ROOT)
    train_ds, val_ds, test_ds = get_train_val_test_datasets()
    print("Dataset loader initialized successfully!")
