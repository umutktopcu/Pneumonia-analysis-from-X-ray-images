"""Stage 3a - Resize strategy study (exploratory; informs decision D6).

Compares three ways to turn non-square X-rays into 224x224 inputs:
  squash      : resize directly -> aspect ratio distorted
  pad         : keep aspect ratio, pad to a square with black borders, then resize
  center_crop : keep aspect ratio, crop the central square, then resize
For each strategy it measures whether simple, non-anatomical image statistics still
separate the classes, i.e. how much of the image-size shortcut survives preprocessing.
Also renders an aliasing demo (INTER_NEAREST vs INTER_LINEAR vs INTER_AREA).

Usage (from project root):  python src/03a_resize_study.py [--n-per-class 300]
Output: outputs/03_preprocessing/{resize_study.txt, strategies_grid.png, aliasing_demo.png}
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shortcut_checks import FEATURES, features, separability

SIZE = 224
SEED = 42


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # also converts the 283 RGB files (D1)
    if img is None:
        raise FileNotFoundError(path)
    return img


def squash(img, size=SIZE, interp=cv2.INTER_AREA):
    return cv2.resize(img, (size, size), interpolation=interp)


def pad(img, size=SIZE, interp=cv2.INTER_AREA):
    h, w = img.shape
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    square = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(square, (size, size), interpolation=interp)


def center_crop(img, size=SIZE, interp=cv2.INTER_AREA):
    h, w = img.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[top:top + s, left:left + s], (size, size), interpolation=interp)


STRATEGIES = {"squash": squash, "pad": pad, "center_crop": center_crop}


def run_study(sample: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    variants = {**STRATEGIES, "squash (INTER_LINEAR)": lambda im: squash(im, interp=cv2.INTER_LINEAR)}
    rows = []
    for rec in sample.itertuples():
        img = load_gray(data_dir / rec.relpath)
        for name, fn in variants.items():
            rows.append({"strategy": name, "label": rec.label, **features(fn(img))})
    feats = pd.DataFrame(rows)

    y_ref = (sample["label"] == "PNEUMONIA").to_numpy().astype(int)
    table = {"(raw) aspect ratio w/h": {"reference": separability((sample.width / sample.height).to_numpy(), y_ref)}}
    for name, g in feats.groupby("strategy", sort=False):
        y = (g["label"] == "PNEUMONIA").to_numpy().astype(int)
        table[name] = {c: separability(g[c].to_numpy(), y) for c in FEATURES}
    return pd.DataFrame(table).T.round(3)


def plot_strategies(sample: pd.DataFrame, data_dir: Path, out: Path, n=3):
    picks = pd.concat([g.sample(n, random_state=SEED) for _, g in sample.groupby("label")])
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    for r, rec in enumerate(picks.itertuples()):
        img = load_gray(data_dir / rec.relpath)
        panels = [("original", img)] + [(k, fn(img)) for k, fn in STRATEGIES.items()]
        for c, (title, im) in enumerate(panels):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            label = f"{rec.label}  {rec.width}x{rec.height}" if c == 0 else title
            axes[r, c].set_title(label, fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def zone_plate(n=1568, period_at_edge_px=4.0):
    """Concentric rings whose spatial frequency rises towards the edge: a classic aliasing test."""
    x = np.linspace(-1, 1, n, dtype=np.float32)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k = np.pi * (n / 2) / period_at_edge_px
    return ((0.5 + 0.5 * np.cos(k * r2)) * 255).astype(np.uint8)


def plot_aliasing(xray: np.ndarray, out: Path):
    methods = {"INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR, "INTER_AREA": cv2.INTER_AREA}
    zp = zone_plate()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(zp, cmap="gray"); axes[0, 0].set_title(f"zone plate {zp.shape[1]}x{zp.shape[0]}")
    axes[1, 0].imshow(xray, cmap="gray"); axes[1, 0].set_title(f"X-ray {xray.shape[1]}x{xray.shape[0]}")
    for c, (name, interp) in enumerate(methods.items(), start=1):
        axes[0, c].imshow(cv2.resize(zp, (SIZE, SIZE), interpolation=interp), cmap="gray", vmin=0, vmax=255)
        small = squash(xray, interp=interp)
        zoom = small[60:140, 20:100]  # right-lung / rib region, shown enlarged
        axes[1, c].imshow(zoom, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, c].set_title(f"{name} -> {SIZE}x{SIZE}")
        axes[1, c].set_title(f"{name} (zoomed region of {SIZE}px)")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/02_splits/splits.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/03_preprocessing"))
    parser.add_argument("--n-per-class", type=int, default=300)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(args.splits)
    train = splits[splits["split"] == "train"]  # study on train only: test stays untouched
    sample = pd.concat([g.sample(min(args.n_per_class, len(g)), random_state=SEED)
                        for _, g in train.groupby("label")])

    table = run_study(sample, args.data_dir)
    text = (f"Separability max(AUC, 1-AUC) of non-anatomical features, train sample "
            f"n={len(sample)} ({args.n_per_class}/class). 0.5 = no class info.\n\n{table.to_string()}")
    print(text)
    (args.out_dir / "resize_study.txt").write_text(text)

    plot_strategies(sample, args.data_dir, args.out_dir / "strategies_grid.png")
    biggest = train.loc[train["width"].idxmax()]
    plot_aliasing(load_gray(args.data_dir / biggest["relpath"]), args.out_dir / "aliasing_demo.png")
    print(f"\nFigures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
