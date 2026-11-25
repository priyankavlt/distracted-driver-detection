"""Resize the raw 640x480 images once into a smaller cache.

JPEG decoding of full-size frames dominates data-loading time; the models train at
<= 224 px, so caching at 320x240 (same 4:3 aspect) makes each epoch several times
faster with no loss in accuracy.

    python -m ddd.prepare                       # data/raw -> data/processed (320x240)
    python -m ddd.prepare --size 480 360        # custom W H
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .utils import PROJECT_ROOT


def _resize(args):
    src, dst, size = args
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").resize(size, Image.BICUBIC).save(dst, quality=95)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--dst", type=Path, default=PROJECT_ROOT / "data" / "processed")
    p.add_argument("--size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    jobs = [(f, args.dst / f.relative_to(args.src), tuple(args.size))
            for f in sorted((args.src / "imgs").rglob("*.jpg"))]
    with ProcessPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(_resize, jobs, chunksize=64), total=len(jobs), desc="resizing"))
    for csv in args.src.glob("*.csv"):
        shutil.copy(csv, args.dst / csv.name)
    print(f"Cached {len(jobs):,} images at {args.size[0]}x{args.size[1]} in {args.dst}")


if __name__ == "__main__":
    main()
