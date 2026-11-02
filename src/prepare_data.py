"""Scan data/raw, build train/val/test CSVs, print dataset stats and a leakage check.

Usage:  python src/prepare_data.py [--mode block|random]

IQ-OTH/NCCD contains many CT slices per patient, numbered in slice order, so
neighbouring files are near-identical. A random per-image split puts slices of
the same patient in both train and test (leakage -> inflated scores).
"block" mode (default) keeps contiguous runs of file numbers together and
snaps each cut to the nearest sharp drop in image similarity, which is most
likely a patient boundary. "random" is kept for comparison in the report.
"""
import argparse
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from config import CLASSES, RAW_DIR, SEED, SOURCE_FOLDERS, SPLIT, SPLITS_DIR

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SNAP_WINDOW = 6     # how far (in slices) a cut may move to find a boundary
TWIN_SIM = 0.97     # correlation above which two slices count as near-duplicates


def collect():
    rows = []
    for folder, label in SOURCE_FOLDERS.items():
        if label not in CLASSES:
            continue
        matches = [p for p in RAW_DIR.rglob("*") if p.is_dir() and p.name.strip() == folder]
        if not matches:
            raise SystemExit(f"Folder '{folder}' not found under {RAW_DIR}. Run download_data.py first.")
        for p in sorted(matches[0].iterdir()):
            if p.suffix.lower() in IMG_EXT:
                num = int(re.search(r"\((\d+)\)", p.name).group(1))
                rows.append({"path": p.relative_to(RAW_DIR).as_posix(), "label": label, "num": num})
    return rows


def thumb(path):
    """64x64 grayscale vector, zero-mean and unit-norm, so a dot product is a correlation."""
    a = np.asarray(Image.open(RAW_DIR / path).convert("L").resize((64, 64)), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def split_random(items, rng):
    rng.shuffle(items)
    n = len(items)
    return int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))


def block_cuts(items, fractions):
    """Sort items by file number (in place) and return one cut index per fraction,
    each snapped to the lowest neighbour similarity within SNAP_WINDOW."""
    items.sort(key=lambda r: r["num"])
    v = np.stack([thumb(r["path"]) for r in items])
    sim_prev = np.r_[1.0, (v[1:] * v[:-1]).sum(1)]   # sim_prev[k] = sim(item k-1, item k)
    cuts = []
    for f in fractions:
        target = int(len(items) * f)
        lo, hi = max(1, target - SNAP_WINDOW), min(len(items) - 1, target + SNAP_WINDOW)
        cuts.append(lo + int(np.argmin(sim_prev[lo:hi + 1])))
    return cuts


def split(rows, mode):
    rng = random.Random(SEED)
    out = []
    for label in CLASSES:
        items = [r for r in rows if r["label"] == label]
        c1, c2 = (block_cuts(items, (SPLIT[0], SPLIT[0] + SPLIT[1])) if mode == "block"
                  else split_random(items, rng))
        for i, r in enumerate(items):
            r["split"] = "train" if i < c1 else "val" if i < c2 else "test"
            out.append(r)
    return pd.DataFrame(out)


def leakage_report(df):
    """How many val/test slices have a near-duplicate in the training set."""
    v = np.stack([thumb(p) for p in df.path])
    train = v[(df.split == "train").values]
    for s in ("val", "test"):
        best = (v[(df.split == s).values] @ train.T).max(1)
        print(f"  {s}: {(best >= TWIN_SIM).sum()}/{len(best)} images have a train twin "
              f"(corr >= {TWIN_SIM}); median nearest-train corr {np.median(best):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["block", "random"], default="block")
    args = ap.parse_args()

    df = split(collect(), args.mode)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df.split == s].drop(columns="num").to_csv(SPLITS_DIR / f"{s}.csv", index=False)

    sizes = Counter(Image.open(RAW_DIR / p).size for p in df.path.head(200))
    print(f"Split mode: {args.mode}")
    print("Images per class/split:")
    print(pd.crosstab(df.label, df.split, margins=True))
    print("Most common image sizes (sample of 200):", sizes.most_common(3))
    print("Leakage check:")
    leakage_report(df)


if __name__ == "__main__":
    main()
