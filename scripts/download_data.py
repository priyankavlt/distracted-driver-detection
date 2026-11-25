"""Download the State Farm Distracted Driver dataset.

The official source is the Kaggle competition (requires an account + accepting the
rules). For convenience this script can also pull from a public Hugging Face mirror
of the exact same archive. It reads the zip's central directory over HTTP and only
downloads the byte ranges it needs, so by default you fetch ~1 GB of training images
instead of the full 4.3 GB archive (79k unlabeled test images).

Usage:
    python scripts/download_data.py                  # train images + CSVs -> data/raw
    python scripts/download_data.py --with-test      # also the 79,726 Kaggle test images
    python scripts/download_data.py --from-kaggle    # use the Kaggle CLI instead
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

MIRROR_URL = (
    "https://huggingface.co/datasets/gymprathap/Driver-Distracted-Dataset/"
    "resolve/main/Distracted-Driver-Detection-Dataset.zip"
)
EXPECTED_TRAIN_IMAGES = 22_424
ROOT = Path(__file__).resolve().parent.parent


def _open(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": "ddd-downloader"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


class HTTPRangeFile:
    """Minimal seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        with _open(self.url, self.pos, self.pos + n - 1) as r:
            data = r.read()
        self.pos += len(data)
        return data


def download_range(url: str, start: int, end: int, dest: Path) -> None:
    """Download bytes [start, end] into dest, resuming if dest is partially present."""
    total = end - start + 1
    done = dest.stat().st_size if dest.exists() else 0
    if done >= total:
        return
    for attempt in range(10):
        try:
            with _open(url, start + done, end) as r, open(dest, "ab") as f:
                t0, last = time.time(), 0.0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 2:
                        last = time.time()
                        rate = done / max(last - t0, 1e-6) / 1e6
                        print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB  ({rate:.1f} MB/s)", end="", flush=True)
            print()
            if done >= total:
                return
        except Exception as e:  # network hiccup -> resume
            print(f"\n  retry {attempt + 1}: {e}")
            time.sleep(3)
    raise RuntimeError("Download failed after retries")


def extract_members(block: Path, block_start: int, members: list[zipfile.ZipInfo], out_dir: Path) -> None:
    with open(block, "rb") as f:
        for i, info in enumerate(members):
            f.seek(info.header_offset - block_start)
            header = f.read(30)
            sig, *_rest = struct.unpack("<IHHHHHIIIHH", header)
            if sig != 0x04034B50:
                raise ValueError(f"Bad local header for {info.filename}")
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            f.seek(name_len + extra_len, io.SEEK_CUR)
            raw = f.read(info.compress_size)
            if info.compress_type == zipfile.ZIP_DEFLATED:
                raw = zlib.decompress(raw, -15)
            elif info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Unsupported compression {info.compress_type}")
            if zlib.crc32(raw) != info.CRC:
                raise ValueError(f"CRC mismatch for {info.filename}")
            rel = info.filename.split("imgs/", 1)[-1]  # imgs/train/c0/x.jpg -> train/c0/x.jpg
            target = out_dir / "imgs" / rel if "/" in rel else out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if i % 2000 == 0:
                print(f"\r  extracted {i}/{len(members)}", end="", flush=True)
    print(f"\r  extracted {len(members)}/{len(members)}")


def from_mirror(out_dir: Path, with_test: bool) -> None:
    print(f"Reading archive index from mirror:\n  {MIRROR_URL}")
    remote = HTTPRangeFile(MIRROR_URL)
    zf = zipfile.ZipFile(remote)  # type: ignore[arg-type]
    infos = [i for i in zf.infolist() if not i.is_dir()]

    for info in infos:  # small metadata files
        if info.filename.endswith(".csv"):
            (out_dir / Path(info.filename).name).write_bytes(zf.read(info))
            print(f"  saved {info.filename}")

    splits = ["train"] + (["test"] if with_test else [])
    for split in splits:
        members = sorted((i for i in infos if f"/{split}/" in i.filename and i.filename.endswith(".jpg")),
                         key=lambda i: i.header_offset)
        start = members[0].header_offset
        # end of the last member's data (+ generous slack for its local header)
        last = members[-1]
        end = min(last.header_offset + 30 + 1024 + last.compress_size, remote.size - 1)
        block = out_dir / f".{split}_block.bin"
        print(f"Downloading {split} images ({len(members):,} files, {(end - start) / 1e9:.2f} GB)")
        download_range(MIRROR_URL, start, end, block)
        extract_members(block, start, members, out_dir)
        block.unlink()


def from_kaggle(out_dir: Path) -> None:
    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found: pip install kaggle and add ~/.kaggle/kaggle.json")
    subprocess.run(["kaggle", "competitions", "download", "-c", "state-farm-distracted-driver-detection",
                    "-p", str(out_dir)], check=True)
    with zipfile.ZipFile(out_dir / "state-farm-distracted-driver-detection.zip") as z:
        z.extractall(out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--with-test", action="store_true", help="also download the unlabeled Kaggle test set")
    p.add_argument("--from-kaggle", action="store_true", help="use the official Kaggle CLI")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_kaggle:
        from_kaggle(args.out)
    else:
        from_mirror(args.out, args.with_test)

    n = sum(1 for _ in (args.out / "imgs" / "train").rglob("*.jpg"))
    print(f"Done. {n:,} training images in {args.out / 'imgs' / 'train'}")
    if n != EXPECTED_TRAIN_IMAGES:
        print(f"WARNING: expected {EXPECTED_TRAIN_IMAGES:,} training images", file=sys.stderr)


if __name__ == "__main__":
    main()
