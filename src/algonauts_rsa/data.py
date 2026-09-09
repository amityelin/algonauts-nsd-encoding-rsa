"""Dataset loading and ROI extraction helpers."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def load_training_fmri(subject_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    base = subject_dir / "training_split" / "training_fmri"
    left_path = base / "lh_training_fmri.npy"
    right_path = base / "rh_training_fmri.npy"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Missing training fMRI:\n  {left_path}\n  {right_path}")
    left = np.load(left_path)
    right = np.load(right_path)
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"LH/RH row mismatch: {left.shape} vs {right.shape}")
    return left, right


def get_roi_label_map(subject_dir: Path, roi_class: str, hemisphere: str):
    mask_path = subject_dir / "roi_masks" / f"{hemisphere}.{roi_class}_challenge_space.npy"
    map_path = subject_dir / "roi_masks" / f"mapping_{roi_class}.npy"
    if not mask_path.exists() or not map_path.exists():
        raise FileNotFoundError(f"Missing ROI files:\n  {mask_path}\n  {map_path}")
    return np.load(mask_path), np.load(map_path, allow_pickle=True).item()


def roi_bool_from_labels(mask: np.ndarray, label_map: dict, desired_names) -> np.ndarray:
    wanted = set(desired_names)
    ids = [key for key, value in label_map.items() if value in wanted]
    return np.isin(mask, ids) if ids else np.zeros_like(mask, dtype=bool)


def parse_training_image_meta(images_dir: Path, n_images: int) -> pd.DataFrame:
    rows = []
    for path in sorted(images_dir.glob("*.png"))[:n_images]:
        match = re.search(r"train-(\d+)_nsd-(\d+)\.png", path.name)
        train_index, nsd_id = (
            (int(match.group(1)), int(match.group(2))) if match else (None, None)
        )
        rows.append((path.name, train_index, nsd_id))
    return pd.DataFrame(rows, columns=["filename", "train_idx_1based", "nsd_id"])


def validate_alignment(metadata: pd.DataFrame, responses: np.ndarray) -> None:
    if len(metadata) != responses.shape[0]:
        raise ValueError(
            f"Stimulus/response row mismatch: {len(metadata)} metadata rows vs "
            f"{responses.shape[0]} response rows"
        )
    if metadata["filename"].duplicated().any():
        raise ValueError("Duplicate stimulus filenames found in subject metadata")
