"""Validation for committed preliminary result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_ROIS = ("EBA", "FFA", "PPA")
FINDINGS_COLUMNS = (
    "result_status",
    "n_subjects",
    "roi",
    "clip_group_median_r2",
    "resnet_group_median_r2",
    "median_paired_delta_r2",
    "clip_group_median_rsa_rho",
    "resnet_group_median_rsa_rho",
    "median_paired_delta_rsa_rho",
)
PRELIMINARY_STATUS = "preliminary_FAST_MODE"


def load_validated_findings(csv_path: Path, manifest_path: Path):
    """Load result artifacts after validating their schema and shared metadata."""
    frame = pd.read_csv(csv_path)
    actual_columns = tuple(frame.columns)
    if actual_columns != FINDINGS_COLUMNS:
        raise ValueError(
            "findings.csv schema mismatch: expected columns in this order "
            f"{FINDINGS_COLUMNS}, got {actual_columns}"
        )
    if tuple(frame["roi"]) != EXPECTED_ROIS:
        raise ValueError(
            f"findings.csv must contain exactly these ROIs in order: {EXPECTED_ROIS}"
        )
    if set(frame["result_status"]) != {PRELIMINARY_STATUS}:
        raise ValueError(f"Every findings row must have status {PRELIMINARY_STATUS!r}")
    if frame.isna().any().any():
        raise ValueError("findings.csv must not contain missing values")

    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "preliminary FAST_MODE results":
        raise ValueError("Manifest must label the results as preliminary FAST_MODE results")
    if tuple(manifest.get("rois", ())) != EXPECTED_ROIS:
        raise ValueError("Manifest ROI list does not match the expected findings ROIs")
    subject_counts = set(frame["n_subjects"].astype(int))
    if subject_counts != {int(manifest.get("n_subjects", -1))}:
        raise ValueError("CSV n_subjects values do not match the manifest")
    return frame, manifest
