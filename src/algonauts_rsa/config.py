"""Portable paths and named analysis configurations.

Filesystem resolution is deterministic: explicit arguments take precedence, then
the documented environment variables, then Colab-only defaults when actually
running in Google Colab. Local runs without configured paths fail with an
actionable error.

``ALGONAUTS_DATA_ROOT``
    Directory containing ``subj01`` ... ``subj08``.
``ALGONAUTS_OUTPUT_ROOT``
    Directory for derived arrays, tables, caches, and figures.
``ALGONAUTS_RUN_MODE``
    One of ``smoke``, ``fast``, or ``full`` (default: ``fast``).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np


SEED = 42


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    seed: int
    outer_folds: int
    inner_folds: int
    alphas: np.ndarray
    n_permutations: int
    rsa_subset_size: int | None
    debug_subjects: tuple[str, ...] | None = None
    debug_rois: tuple[str, ...] | None = None
    debug_n_images: int | None = None


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    output_root: Path

    @property
    def multi_roi_dir(self) -> Path:
        return self.output_root / "multiROI"

    def model_results_dir(self, feature_model: str) -> Path:
        return self.output_root / f"encoding_rsa_random_{feature_model}"


CONFIGS = {
    "smoke": AnalysisConfig(
        name="smoke",
        seed=SEED,
        outer_folds=2,
        inner_folds=2,
        alphas=np.logspace(-2, 2, 4),
        n_permutations=10,
        rsa_subset_size=32,
        debug_subjects=("subj01",),
        debug_rois=("FFA",),
        debug_n_images=64,
    ),
    # Exact configuration used by the preliminary committed notebook outputs.
    "fast": AnalysisConfig(
        name="fast",
        seed=SEED,
        outer_folds=3,
        inner_folds=2,
        alphas=np.logspace(-2, 3, 8),
        n_permutations=200,
        rsa_subset_size=600,
    ),
    # Defined for a future approved rerun; no full-mode results are reported.
    "full": AnalysisConfig(
        name="full",
        seed=SEED,
        outer_folds=5,
        inner_folds=3,
        alphas=np.logspace(-2, 4, 12),
        n_permutations=1000,
        rsa_subset_size=None,
    ),
}


def get_config(name: str | None = None) -> AnalysisConfig:
    mode = (name or os.getenv("ALGONAUTS_RUN_MODE", "fast")).lower()
    try:
        return CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown run mode {mode!r}; choose from {sorted(CONFIGS)}") from exc


def _colab_default_data_root() -> Path:
    shortcut_id = os.getenv("ALGONAUTS_DRIVE_SHORTCUT_ID", "1DUf3nGNNFk6YjRjQtZPfAY5N105GoGJb")
    return Path("/content/drive/.shortcut-targets-by-id") / shortcut_id / "algonauts_2023_tutorial_data"


def _running_in_colab() -> bool:
    """Return true only for an active Google Colab Python runtime."""
    return "google.colab" in sys.modules


def _resolve_root(explicit, environment_name: str, colab_default: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    environment_value = os.getenv(environment_name)
    if environment_value:
        return Path(environment_value).expanduser()
    if _running_in_colab() and colab_default is not None:
        return colab_default
    raise RuntimeError(
        f"No path configured for {environment_name}. Pass it explicitly to get_paths(), "
        f"or set the {environment_name} environment variable. Colab defaults are used "
        "only inside an active Google Colab runtime."
    )


def get_paths(data_root=None, output_root=None) -> ProjectPaths:
    data_root = _resolve_root(data_root, "ALGONAUTS_DATA_ROOT", _colab_default_data_root())
    output_root = _resolve_root(
        output_root, "ALGONAUTS_OUTPUT_ROOT", Path("/content/drive/MyDrive/algonauts_outputs")
    )
    return ProjectPaths(data_root=data_root, output_root=output_root)
