"""Train a distracted-driver classifier with a driver-grouped train/val/test split.

Examples:
    # Recommended: ImageNet-pretrained ResNet34 at 4:3 resolution
    python -m ddd.train --arch resnet34 --img-size 168 224 --epochs 8 --name resnet34

    # Baseline: the original from-scratch ResNet50 at 64x64 (bugs fixed)
    python -m ddd.train --arch resnet50_scratch --no-pretrained --img-size 64 64 \\
        --epochs 15 --lr 1e-3 --batch-size 64 --name resnet50_scratch_64

    # Quick smoke test
    python -m ddd.train --arch resnet18 --epochs 1 --limit 256 --name smoke
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CLASS_IDS, CLASS_NAMES
from .data import DriverImages, build_transforms, get_split, load_metadata, resolve_data_dir, subset
from .evaluate import evaluate_split, fit_temperature, predict_logits
from .models import build_model, model_norm_stats
from .utils import PROJECT_ROOT, get_device, save_json, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="resnet34", help="'resnet50_scratch' or any timm model name")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--img-size", type=int, nargs=2, default=(168, 224), metavar=("H", "W"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-rate", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=4, help="early stopping on val accuracy")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="subsample each split (smoke tests)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--finalize-only", action="store_true",
                   help="skip training; calibrate + evaluate an existing runs/<name>/best.pt")
    return p.parse_args(argv)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def plot_history(hist: pd.DataFrame, path: Path, title: str):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist.epoch, hist.train_loss, "o-", label="train (smoothed CE)")
    a1.plot(hist.epoch, hist.val_nll, "o-", label="val log loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(hist.epoch, hist.train_acc, "o-", label="train")
    a2.plot(hist.epoch, hist.val_acc, "o-", label="val (unseen drivers)")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    name = args.name or f"{args.arch}_{args.img_size[0]}x{args.img_size[1]}"
    out = args.runs_dir / name
    out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if not args.finalize_only:
        save_json(config, out / "config.json")

    df = load_metadata(resolve_data_dir(args.data_dir))
    split = get_split(df)
    parts = {k: subset(df, split, k) for k in ("train", "val", "test")}
    if args.limit:
        parts = {k: v.sample(min(args.limit, len(v)), random_state=args.seed).reset_index(drop=True)
                 for k, v in parts.items()}
    print(f"device={device}  " + "  ".join(f"{k}: {len(v):,} imgs / {v.subject.nunique()} drivers"
                                            for k, v in parts.items()))

    model = build_model(args.arch, pretrained=args.pretrained, drop_rate=args.drop_rate).to(device)
    mean, std = model_norm_stats(model)
    img_size = tuple(args.img_size)
    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=False)
    train_loader = DataLoader(
        DriverImages(parts["train"].path, parts["train"].label, build_transforms(img_size, True, mean, std)),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(
        DriverImages(parts["val"].path, parts["val"].label, build_transforms(img_size, False, mean, std)),
        batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    meta = {"arch": args.arch, "img_size": list(img_size), "mean": list(mean), "std": list(std),
            "class_ids": CLASS_IDS, "class_names": CLASS_NAMES, "split": split, "config": config}
    if args.finalize_only:
        history = pd.read_csv(out / "history.csv").to_dict("records")
        best_epoch = int(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["epoch"])
    else:
        optimizer = create_optimizer_v2(model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = len(train_loader)
        scheduler = cosine_with_warmup(optimizer, int(args.warmup_epochs * steps_per_epoch),
                                       args.epochs * steps_per_epoch)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_acc, best_epoch, history = -1.0, 0, []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            tot_loss = tot_correct = tot_n = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                n = y.size(0)
                tot_loss += loss.item() * n
                tot_correct += (logits.argmax(1) == y).sum().item()
                tot_n += n
                pbar.set_postfix(loss=f"{tot_loss / tot_n:.3f}", acc=f"{tot_correct / tot_n:.3f}")

            val_logits, val_labels = predict_logits(model, val_loader, device, desc="val")
            val_nll = F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(val_labels)).item()
            val_acc = float((val_logits.argmax(1) == val_labels).mean())
            row = {"epoch": epoch, "train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n,
                   "val_nll": val_nll, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"],
                   "seconds": round(time.time() - t0, 1)}
            history.append(row)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            improved = val_acc > best_acc
            print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
                  f"val logloss {val_nll:.4f} acc {val_acc:.4f} | {row['seconds']:.0f}s" + ("  *" if improved else ""),
                  flush=True)
            if improved:
                best_acc, best_epoch = val_acc, epoch
                torch.save({**meta, "state_dict": model.state_dict(), "epoch": epoch}, out / "best.pt")
            elif epoch - best_epoch >= args.patience:
                print(f"early stopping (no val improvement for {args.patience} epochs)")
                break

    hist = pd.DataFrame(history)
    plot_history(hist, out / "curves.png", f"{args.arch} @ {img_size[0]}x{img_size[1]}")

    # Reload best weights, calibrate on validation drivers, then score untouched test drivers.
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    val_logits, val_labels = predict_logits(model, val_loader, device, desc="calibrate")
    ckpt["temperature"] = fit_temperature(val_logits, val_labels)
    torch.save(ckpt, out / "best.pt")

    results = {}
    for part in ("val", "test"):
        results[part] = evaluate_split(model, ckpt, parts[part], device, out, part,
                                       args.batch_size * 2, args.workers)
    summary = {
        "arch": args.arch, "pretrained": args.pretrained, "img_size": list(img_size),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_minutes": round(hist.seconds.sum() / 60, 1),
        "temperature": ckpt["temperature"],
        "final_train_acc": float(hist.train_acc.iloc[best_epoch - 1]),
        **{f"{p}_{k}": results[p][k] for p in results for k in ("accuracy", "log_loss", "macro_f1")},
    }
    save_json(summary, out / "summary.json")
    print("\n".join(f"  {k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
