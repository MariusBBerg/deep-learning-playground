from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def iter_pngs(root: Path):
    for p in root.rglob("*.png"):
        yield p


def summarize(root: Path, sample: int | None):
    paths = list(iter_pngs(root))
    if not paths:
        print(f"No PNGs found under {root}")
        return

    if sample is not None and sample < len(paths):
        rng = np.random.default_rng(0)
        paths = list(rng.choice(paths, size=sample, replace=False))

    total = 0
    all_black = 0
    very_dark = 0
    very_bright = 0
    stats = []
    overexposed = 0

    for p in paths:
        arr = np.array(Image.open(p))
        if arr.ndim == 2:
            mean = float(arr.mean())
            mx = float(arr.max())
            mn = float(arr.min())
        else:
            mean = float(arr.mean())
            mx = float(arr.max())
            mn = float(arr.min())

        total += 1
        p01 = float(np.percentile(arr, 1))
        p50 = float(np.percentile(arr, 50))
        if p50 >= 250 and p01 >= 150:
            overexposed += 1
        stats.append((mean, mx, mn, p))

        if mx == 0:
            all_black += 1
        if mean < 5:
            very_dark += 1
        if mean > 245:
            very_bright += 1

    stats.sort(key=lambda x: x[0])
    print(f"Scanned: {total}")
    print(f"All-black (max=0): {all_black}")
    print(f"Very dark (mean<5): {very_dark}")
    print(f"Very bright (mean>245): {very_bright}")
    print(f"Overexposed (p50>=250 & p01>=150): {overexposed}")

    print("\n5 darkest by mean:")
    for mean, mx, mn, p in stats[:5]:
        print(f"{mean:6.2f}  min={mn:3.0f} max={mx:3.0f}  {p}")

    print("\n5 brightest by mean:")
    for mean, mx, mn, p in stats[-5:]:
        print(f"{mean:6.2f}  min={mn:3.0f} max={mx:3.0f}  {p}")


def inspect_file(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return
    arr = np.array(Image.open(path))
    mean = float(arr.mean())
    mn = float(arr.min())
    mx = float(arr.max())
    pcts = np.percentile(arr, [1, 5, 50, 95, 99]).tolist()
    print(f"{path}")
    print(f"shape={arr.shape} dtype={arr.dtype}")
    print(f"mean={mean:.2f} min={mn:.0f} max={mx:.0f}")
    print(f"p01={pcts[0]:.0f} p05={pcts[1]:.0f} p50={pcts[2]:.0f} p95={pcts[3]:.0f} p99={pcts[4]:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder to scan (e.g. tiles_vegas_rgb/train/images)")
    ap.add_argument("--sample", type=int, default=None, help="Optional random sample size")
    ap.add_argument("--file", default=None, help="Inspect a single file for stats")
    args = ap.parse_args()

    summarize(Path(args.root), args.sample)
    if args.file:
        print("\nSingle file stats:")
        inspect_file(Path(args.file))


if __name__ == "__main__":
    main()
