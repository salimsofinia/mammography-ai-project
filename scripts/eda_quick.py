#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

IMG_COL_CANDIDATES = [
    "path", "filepath", "image_path", "img_path", "file", "filename",
    "image", "imagefile", "image_file", "relpath"
]
LABEL_COL_CANDIDATES = ["label", "class", "target", "y", "diagnosis"]

def find_col(df, candidates, purpose):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a column for {purpose}. "
                     f"Tried: {candidates}. Found: {list(df.columns)}")

def to_paths(series, dataset_root: Path):
    """Turn a column into resolved Paths. Handles absolute, relative, bare filenames."""
    paths = []
    for v in series.astype(str):
        p = Path(v)
        if p.is_absolute():
            paths.append(p)
        else:
            # if v contains any path separator, treat as project-relative
            if any(sep in v for sep in ["/", "\\"]):
                paths.append((Path.cwd() / p).resolve())
            else:
                # bare filename — assume lives under dataset_root (if provided)
                if dataset_root:
                    paths.append((dataset_root / p).resolve())
                else:
                    paths.append((Path.cwd() / p).resolve())
    return paths

def safe_open_grayscale(p: Path):
    with Image.open(p) as im:
        im = im.convert("L")  # force grayscale
        return im

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_hist(data, title, xlabel, outpath):
    plt.figure()
    plt.hist(data, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_bar(categories, counts, title, outpath):
    plt.figure()
    plt.bar(categories, counts)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def make_montage(images, cols, outpng, subtitle=None):
    n = len(images)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols*2, rows*2))
    for i, im in enumerate(images, 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(im, cmap="gray")
        ax.axis("off")
    if subtitle:
        fig.suptitle(subtitle)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def compute_stats_for_split(split_name, df, outdir: Path, dataset_root: Path, montage_per_class=8, seed=42):
    rng = random.Random(seed)
    label_col = find_col(df, LABEL_COL_CANDIDATES, "label")
    img_col = find_col(df, IMG_COL_CANDIDATES, "image path")

    # Resolve paths
    paths = to_paths(df[img_col], dataset_root)
    df = df.copy()
    df["__path__"] = [str(p) for p in paths]
    df["__exists__"] = [Path(p).exists() for p in df["__path__"]]

    # Missing file report
    missing = df[~df["__exists__"]]["__path__"].tolist()
    missing_file = outdir / f"eda_missing_images_{split_name}.txt"
    if missing:
        with open(missing_file, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
    print(f"[{split_name}] total rows: {len(df)}, missing files: {len(missing)}")

    # Class distribution
    counts = df[label_col].value_counts()
    counts_png = outdir / f"eda_counts_bar_{split_name}.png"
    plot_bar(list(counts.index.astype(str)), list(counts.values),
             f"{split_name} – Class distribution", counts_png)

    # Scan images that exist
    widths, heights, aspects, means, stds, keep_paths, keep_labels = [], [], [], [], [], [], []
    for pth, lab, ok in zip(df["__path__"], df[label_col], df["__exists__"]):
        if not ok:
            continue
        p = Path(pth)
        try:
            im = safe_open_grayscale(p)
            w, h = im.size
            arr = np.asarray(im, dtype=np.uint8)
            widths.append(w)
            heights.append(h)
            aspects.append(w / h if h else np.nan)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            keep_paths.append(str(p))
            keep_labels.append(lab)
        except (UnidentifiedImageError, OSError) as e:
            # skip unreadables
            continue

    # Save image-level stats CSV
    stats_df = pd.DataFrame({
        "path": keep_paths,
        "label": keep_labels,
        "width": widths,
        "height": heights,
        "aspect": aspects,
        "mean_intensity": means,
        "std_intensity": stds
    })
    stats_csv = outdir / f"eda_image_stats_{split_name}.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Dimensions hist
    dim_png = outdir / f"eda_dim_hist_{split_name}.png"
    plt.figure()
    plt.hist(widths, bins=40, alpha=0.7, label="width")
    plt.hist(heights, bins=40, alpha=0.5, label="height")
    plt.title(f"{split_name} – Image dimensions")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dim_png, dpi=150)
    plt.close()

    # Aspect ratio hist
    aspect_png = outdir / f"eda_aspect_hist_{split_name}.png"
    plot_hist([a for a in aspects if np.isfinite(a)],
              f"{split_name} – Aspect ratio (W/H)", "Aspect ratio", aspect_png)

    # Intensity hist
    intensity_png = outdir / f"eda_intensity_hist_{split_name}.png"
    plot_hist(means, f"{split_name} – Mean intensity per image", "Mean intensity (0–255)", intensity_png)

    # Montages per class
    montage_counts = min(montage_per_class, 20)
    out_montages = []
    for lab in sorted(df[label_col].dropna().unique(), key=lambda x: str(x)):
        candidates = [Path(p) for p, l, ok in zip(df["__path__"], df[label_col], df["__exists__"]) if ok and l == lab]
        rng.shuffle(candidates)
        sel = candidates[:montage_counts]
        ims = []
        for p in sel:
            try:
                ims.append(safe_open_grayscale(p))
            except Exception:
                continue
        if ims:
            m_png = outdir / f"eda_montage_{split_name}_{str(lab)}.png"
            make_montage(ims, cols=4, outpng=m_png, subtitle=f"{split_name} – {lab}")
            out_montages.append(m_png.name)

    # Console summary
    print(f"[{split_name}] classes:\n{counts.to_string()}")
    print(f"[{split_name}] images scanned: {len(widths)}")
    print(f"[{split_name}] stats saved: {stats_csv.name}")
    return {
        "counts_png": counts_png.name,
        "dim_png": dim_png.name,
        "aspect_png": aspect_png.name,
        "intensity_png": intensity_png.name,
        "stats_csv": stats_csv.name,
        "missing_file": missing_file.name if missing else None
    }

def main():
    ap = argparse.ArgumentParser(description="Quick EDA for mammography splits.")
    ap.add_argument("--splits", type=str, default="data/splits",
                    help="Folder containing train/val/test CSVs.")
    ap.add_argument("--dataset-root", type=str, default="",
                    help="Optional root where bare filenames live (e.g., data/raw/mias).")
    ap.add_argument("--outdir", type=str, default="reports/figures",
                    help="Output directory for figures.")
    ap.add_argument("--montage-per-class", type=int, default=8,
                    help="Images per class in the montage (max 20).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits-list", type=str, nargs="*", default=["train", "val", "test"],
                    help="Which splits to process (defaults to train val test).")
    args = ap.parse_args()

    splits_dir = Path(args.splits)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Dependencies sanity note for users
    for pkg in ("pandas", "numpy", "PIL", "matplotlib"):
        pass  # placeholder; import already ensures presence

    for s in args.splits_list:
        csv_path = splits_dir / f"{s}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found; skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        try:
            results = compute_stats_for_split(
                split_name=s,
                df=df,
                outdir=outdir,
                dataset_root=dataset_root,
                montage_per_class=args.montage_per_class,
                seed=args.seed
            )
        except Exception as e:
            print(f"[ERROR] Failed on split '{s}': {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
