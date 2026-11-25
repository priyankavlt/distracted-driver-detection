"""Grad-CAM: highlight which pixels drove the prediction.

Useful as a sanity check that the model looks at hands / phone / face rather than
memorising the driver or the car interior.

    python -m ddd.gradcam --checkpoint runs/resnet34_168x224/best.pt img1.jpg img2.jpg --out gradcam.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import CLASS_IDS, CLASS_NAMES
from .models import gradcam_target_layer
from .predict import Predictor


def gradcam(predictor: Predictor, image: Image.Image, class_idx: int | None = None):
    """Returns (heatmap HxW in [0,1] at the model's input size, probs)."""
    model, layer = predictor.model, gradcam_target_layer(predictor.model)
    feats = {}
    def hook(_m, _i, out):
        feats["a"] = out
        out.register_hook(lambda g: feats.__setitem__("g", g))

    handle = layer.register_forward_hook(hook)
    try:
        x = predictor.transform(image.convert("RGB")).unsqueeze(0).to(predictor.device)
        x.requires_grad_(True)
        logits = model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        model.zero_grad(set_to_none=True)
        logits[0, idx].backward()
        a, g = feats["a"].detach(), feats["g"].detach()
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(logits.detach().float() / predictor.temperature, 1)[0].cpu().numpy()
        return cam.cpu().numpy(), probs
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.asarray(image.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return Image.fromarray(np.uint8(255 * ((1 - alpha) * img + alpha * heat)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("gradcam.png"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    pred = Predictor(args.checkpoint, args.device)
    n = len(args.images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, args.images):
        img = Image.open(path)
        cam, probs = gradcam(pred, img)
        k = int(probs.argmax())
        ax.imshow(overlay(img, cam))
        ax.set_title(f"{CLASS_IDS[k]} {CLASS_NAMES[k]} ({probs[k]:.0%})", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
