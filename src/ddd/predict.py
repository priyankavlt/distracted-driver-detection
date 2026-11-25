"""Inference: classify images, or write a Kaggle submission file.

    # Classify individual images or folders
    python -m ddd.predict --checkpoint runs/resnet34_168x224/best.pt path/to/img.jpg path/to/folder/

    # Kaggle submission (img,c0..c9 header, calibrated & clipped probabilities)
    python scripts/download_data.py --with-test
    python -m ddd.predict --checkpoint runs/resnet34_168x224/best.pt --submission data/raw/imgs/test \\
        --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from . import CLASS_IDS, CLASS_NAMES
from .data import DriverImages, build_transforms
from .evaluate import load_checkpoint, predict_logits, softmax
from .utils import get_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Predictor:
    """Reusable wrapper: ``Predictor(ckpt).predict(pil_image) -> {class_name: prob}``."""

    def __init__(self, checkpoint: Path | str, device: str = "auto"):
        self.device = get_device(device)
        self.model, self.ckpt = load_checkpoint(Path(checkpoint), self.device)
        self.transform = build_transforms(tuple(self.ckpt["img_size"]), train=False,
                                          mean=self.ckpt["mean"], std=self.ckpt["std"])
        self.temperature = self.ckpt.get("temperature", 1.0)

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return softmax(self.model(x).float().cpu().numpy(), self.temperature)[0]

    def predict(self, image: Image.Image) -> dict[str, float]:
        probs = self.predict_proba(image)
        return {f"{c}: {n}": float(p) for c, n, p in zip(CLASS_IDS, CLASS_NAMES, probs)}


def _collect(paths) -> list[Path]:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("inputs", nargs="*", help="image files or folders")
    p.add_argument("--submission", type=Path, default=None, help="folder of Kaggle test images")
    p.add_argument("--out", type=Path, default=Path("predictions.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    files = _collect([args.submission] if args.submission else args.inputs)
    if not files:
        p.error("no input images")
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    tf = build_transforms(tuple(ckpt["img_size"]), train=False, mean=ckpt["mean"], std=ckpt["std"])
    loader = DataLoader(DriverImages(files, transform=tf), batch_size=args.batch_size, num_workers=args.workers)
    logits, _ = predict_logits(model, loader, device)
    probs = softmax(logits, ckpt.get("temperature", 1.0))

    if args.submission:
        # Kaggle format: header "img,c0,...,c9", one row per image, rows sum to 1.
        sub = pd.DataFrame(np.clip(probs, 1e-6, 1 - 1e-6), columns=CLASS_IDS)
        sub = sub.div(sub.sum(1), axis=0)
        sub.insert(0, "img", [f.name for f in files])
        sub.to_csv(args.out, index=False, float_format="%.6f")
        print(f"Wrote {len(sub):,} rows to {args.out}")
        return

    rows = []
    for f, pr in zip(files, probs):
        k = int(pr.argmax())
        rows.append({"file": str(f), "class": CLASS_IDS[k], "label": CLASS_NAMES[k], "confidence": float(pr[k]),
                     **dict(zip(CLASS_IDS, pr.round(5)))})
        print(f"{f.name:30s} {CLASS_IDS[k]} {CLASS_NAMES[k]:30s} {pr[k]:.1%}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
