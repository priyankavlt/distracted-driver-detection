"""Evaluation on held-out drivers: accuracy, Kaggle multi-class log loss, macro-F1,
confusion matrix, per-class and per-driver breakdowns.

    python -m ddd.evaluate --checkpoint runs/resnet34/best.pt --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CLASS_IDS, CLASS_NAMES
from .data import DriverImages, build_transforms, get_split, load_metadata, resolve_data_dir, subset
from .models import build_model
from .utils import get_device, save_json


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_logits(model, loader, device, desc="predict"):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        all_logits.append(model(x.to(device)).float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar that calibrates confidences.

    It never changes the argmax (accuracy) but can cut log loss substantially,
    which matters because log loss is the competition metric.
    """
    lg, lb = torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.detach().exp().item()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, subjects=None) -> dict:
    preds = probs.argmax(1)
    clipped = np.clip(probs, 1e-15, 1 - 1e-15)  # Kaggle's clipping
    out = {
        "n_images": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "log_loss": float(log_loss(labels, clipped, labels=list(range(len(CLASS_IDS))))),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_accuracy": {c: float((preds[labels == i] == i).mean()) for i, c in enumerate(CLASS_IDS)
                               if (labels == i).any()},
    }
    if subjects is not None:
        s = pd.Series(np.asarray(subjects))
        out["per_driver_accuracy"] = {d: float((preds[(s == d).values] == labels[(s == d).values]).mean())
                                      for d in sorted(s.unique())}
    return out


def plot_confusion(labels, preds, path: Path, title="Confusion matrix (row-normalized)"):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_IDS))), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ticks = [f"{c}: {n}" for c, n in zip(CLASS_IDS, CLASS_NAMES)]
    ax.set_xticks(range(10), CLASS_IDS)
    ax.set_yticks(range(10), ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_split(model, ckpt, df_split, device, out_dir: Path, name: str, batch_size=64, workers=4) -> dict:
    tf = build_transforms(tuple(ckpt["img_size"]), train=False, mean=ckpt["mean"], std=ckpt["std"])
    loader = DataLoader(DriverImages(df_split.path, df_split.label, tf), batch_size=batch_size,
                        num_workers=workers, shuffle=False)
    logits, labels = predict_logits(model, loader, device, desc=f"eval {name}")
    temperature = ckpt.get("temperature", 1.0)
    probs = softmax(logits, temperature)
    metrics = compute_metrics(probs, labels, df_split.subject.values)
    metrics["log_loss_uncalibrated"] = compute_metrics(softmax(logits), labels)["log_loss"]
    metrics["temperature"] = temperature
    metrics["drivers"] = sorted(df_split.subject.unique().tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(1)
    plot_confusion(labels, preds, out_dir / f"confusion_{name}.png",
                   title=f"{ckpt['arch']} — {name} drivers {', '.join(metrics['drivers'])}")
    (out_dir / f"report_{name}.txt").write_text(
        classification_report(labels, preds, target_names=[f"{c} {n}" for c, n in zip(CLASS_IDS, CLASS_NAMES)],
                              digits=3))
    save_json(metrics, out_dir / f"metrics_{name}.json")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="defaults to the checkpoint's folder")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    df = load_metadata(resolve_data_dir(args.data_dir))
    split = get_split(df)
    m = evaluate_split(model, ckpt, subset(df, split, args.split), device,
                       args.out or args.checkpoint.parent, args.split, args.batch_size, args.workers)
    print(f"[{args.split}] acc={m['accuracy']:.4f}  log_loss={m['log_loss']:.4f}  macro_f1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
