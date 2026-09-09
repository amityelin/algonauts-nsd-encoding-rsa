"""Shared result-discovery and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_rois_and_subjects(results_root: Path):
    roi_dirs = [path for path in results_root.iterdir() if path.is_dir()]
    subjects = {subject.name for roi in roi_dirs for subject in roi.iterdir() if subject.is_dir()}
    return sorted(path.name for path in roi_dirs), sorted(subjects)


def ordered_rois(rois, preferred=("EBA", "FFA", "PPA")):
    rois = list(rois)
    return [roi for roi in preferred if roi in rois] + sorted(roi for roi in rois if roi not in preferred)


def save_heatmap(frame, title, output_path):
    plt.figure(figsize=(max(6, 0.6 * frame.shape[1]), max(3, 0.5 * frame.shape[0])))
    plt.imshow(frame.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(frame.shape[1]), frame.columns, rotation=45, ha="right")
    plt.yticks(range(frame.shape[0]), frame.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
