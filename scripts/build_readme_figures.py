"""Regenerate the two README figures from validated preliminary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algonauts_rsa.results import load_validated_findings


ROOT = Path(__file__).resolve().parents[1]


def build_figures(csv_path: Path, manifest_path: Path, output_dir: Path) -> None:
    frame, manifest = load_validated_findings(csv_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rois = frame["roi"].tolist()
    x = np.arange(len(rois))
    width = 0.36
    n_subjects = manifest["n_subjects"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    axes[0].bar(x - width / 2, frame["clip_group_median_r2"], width, label="CLIP")
    axes[0].bar(x + width / 2, frame["resnet_group_median_r2"], width, label="ResNet")
    axes[0].set(title="Encoding performance", ylabel="Group median R²", xticks=x, xticklabels=rois)
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, frame["clip_group_median_rsa_rho"], width, label="CLIP")
    axes[1].bar(x + width / 2, frame["resnet_group_median_rsa_rho"], width, label="ResNet")
    axes[1].set(title="Representational similarity", ylabel="Group median RSA ρ", xticks=x, xticklabels=rois)
    fig.suptitle(f"Preliminary FAST_MODE group-level medians (n={n_subjects})")
    fig.savefig(output_dir / "preliminary_group_medians.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    axes[0].bar(x, frame["median_paired_delta_r2"], color="#4472C4")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="CLIP − ResNet encoding", ylabel="Median paired ΔR²", xticks=x, xticklabels=rois)
    axes[1].bar(x, frame["median_paired_delta_rsa_rho"], color="#ED7D31")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="CLIP − ResNet RSA", ylabel="Median paired Δρ", xticks=x, xticklabels=rois)
    fig.suptitle(f"Preliminary FAST_MODE median paired differences (n={n_subjects})")
    fig.savefig(output_dir / "preliminary_paired_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "findings.csv")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "results" / "preliminary_fast_mode_manifest.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    build_figures(args.csv, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
