"""Dataset loading, driver-grouped splits and image transforms.

Key design points (each fixes a problem in the original notebook):

* **Splits are by driver.** The 22,424 training images come from only 26 drivers,
  and consecutive frames of the same driver are near-duplicates. A random split
  leaks drivers into validation and reports wildly optimistic accuracy. We hold
  out whole drivers for validation and for a final test set, and commit the split
  to ``configs/split.json`` so every run is evaluated on the same people.
* **No horizontal flips.** Flipping a "texting - right" image produces something
  that looks like "texting - left" (c1 <-> c3, c2 <-> c4) and moves the driver to
  the passenger seat, so flip augmentation silently corrupts labels.
* **One consistent normalization.** The notebook applied Caffe-style
  ``preprocess_input`` (BGR + mean subtraction) and *then* divided by 255.
  Here images are converted to [0, 1] and normalized with the backbone's stats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import CLASS_IDS
from .utils import PROJECT_ROOT, load_json, save_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "configs" / "split.json"


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    """Prefer the resized cache (``data/processed``), fall back to the raw download."""
    if data_dir is not None:
        return Path(data_dir)
    for candidate in (PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "data" / "raw"):
        if (candidate / "driver_imgs_list.csv").exists():
            return candidate
    raise FileNotFoundError("No dataset found. Run `python scripts/download_data.py` first.")


def load_metadata(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "driver_imgs_list.csv")
    df["label"] = df["classname"].map({c: i for i, c in enumerate(CLASS_IDS)}).astype(int)
    df["path"] = [str(Path(data_dir) / "imgs" / "train" / c / f) for c, f in zip(df.classname, df.img)]
    return df


def make_driver_split(df: pd.DataFrame, n_val: int = 4, n_test: int = 4, seed: int = 42) -> dict:
    subjects = np.array(sorted(df["subject"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    test = sorted(subjects[:n_test].tolist())
    val = sorted(subjects[n_test:n_test + n_val].tolist())
    train = sorted(subjects[n_test + n_val:].tolist())
    return {"seed": seed, "train": train, "val": val, "test": test}


def get_split(df: pd.DataFrame, split_path: Path = DEFAULT_SPLIT_PATH) -> dict:
    if split_path.exists():
        split = load_json(split_path)
    else:
        split = make_driver_split(df)
        save_json(split, split_path)
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"])
                | set(split["val"]) & set(split["test"])), "driver leakage between splits"
    return split


def subset(df: pd.DataFrame, split: dict, name: str) -> pd.DataFrame:
    return df[df["subject"].isin(split[name])].reset_index(drop=True)


def build_transforms(img_size: tuple[int, int], train: bool,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    h, w = img_size
    if not train:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([
        # Mild geometric jitter: camera position varies a little between cars.
        T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        T.RandomResizedCrop((h, w), scale=(0.75, 1.0), ratio=(w / h * 0.9, w / h * 1.1)),
        # NOTE: deliberately no RandomHorizontalFlip (it swaps left/right-hand classes).
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


class DriverImages(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = -1 if self.labels is None else self.labels[idx]
        return img, label
