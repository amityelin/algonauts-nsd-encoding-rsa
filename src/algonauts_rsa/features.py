"""Image loading and the feature extractor used by the analysis notebook."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_image_paths(subject: str, n_images: int, metadata_dir: Path, data_root: Path):
    metadata_path = metadata_dir / f"{subject}_train_images_meta.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")
    metadata = pd.read_csv(metadata_path).iloc[:n_images]
    image_dir = data_root / subject / "training_split" / "training_images"
    paths = [image_dir / filename for filename in metadata["filename"].tolist()]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} stimulus images; first: {missing[0]}")
    return paths, len(metadata)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    return features / np.linalg.norm(features, axis=-1, keepdims=True)


def extract_clip_features(
    paths,
    batch_size,
    tag,
    *,
    device,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    use_amp,
):
    """Extract CLIP features with the original notebook's exact execution behavior."""
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    class ImageListDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    ds = ImageListDataset(paths, preprocess)
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    dl = DataLoader(ds, **dl_kwargs)

    feats = []
    amp_ctx = torch.amp.autocast("cuda", enabled=(device == "cuda" and use_amp))
    with torch.no_grad(), amp_ctx:
        for xb in tqdm(dl, desc=f"CLIP feats [{tag}]"):
            xb = xb.to(device, non_blocking=False)
            f = model.encode_image(xb)
            f = (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()
            feats.append(f)
            if device == "cuda":
                torch.cuda.empty_cache()
    return np.vstack(feats), "CLIP_ViT-B-32"
