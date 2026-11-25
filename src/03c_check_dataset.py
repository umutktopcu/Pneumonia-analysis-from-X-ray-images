"""Stage 3c - Sanity checks for src/dataset.py (run locally, needs torch/torchvision).

Checks shapes, dtypes, value ranges, label counts against splits.csv, that val/test are
deterministic and train is augmented, then saves a preview grid and times a DataLoader pass.

Usage (from project root):  python src/03c_check_dataset.py
Output: outputs/03_preprocessing/augment_preview.png
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CLASSES, ChestXrayDataset, denormalize

ROOT = Path("data/processed/sq224")
OUT = Path("outputs/03_preprocessing/augment_preview.png")


def main():
    torch.manual_seed(42)
    splits = pd.read_csv("outputs/02_splits/splits.csv")
    ds = {s: ChestXrayDataset(splits, s, ROOT, train=(s == "train")) for s in ("train", "val", "test")}

    for name, d in ds.items():
        x, y = d[0]
        expected = splits.query("split == @name")["label"].value_counts().reindex(CLASSES).tolist()
        counts = torch.bincount(d.labels, minlength=2).tolist()
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32, x.shape
        assert counts == expected, (name, counts, expected)
        print(f"{name:5s} n={len(d):4d} labels(N,P)={counts}  x range=[{x.min():.2f}, {x.max():.2f}]")

    assert torch.equal(ds["val"][0][0], ds["val"][0][0]), "val must be deterministic"
    assert not torch.equal(ds["train"][0][0], ds["train"][0][0]), "train should be augmented"
    print("determinism checks passed (val fixed, train augmented)")

    # Preview: one original (val pipeline) next to 4 augmented draws, per class
    eval_view = ChestXrayDataset(splits, "train", ROOT, train=False)
    idx = [int((eval_view.labels == c).nonzero()[0]) for c in range(2)]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for r, i in enumerate(idx):
        axes[r, 0].imshow(denormalize(eval_view[i][0]), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{CLASSES[r]} (no augmentation)", fontsize=9)
        for c in range(1, 5):
            axes[r, c].imshow(denormalize(ds["train"][i][0]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"augmented #{c}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print(f"preview saved to {OUT}")

    for workers in (0, 4):
        loader = DataLoader(ds["train"], batch_size=64, shuffle=True, num_workers=workers)
        start = time.time()
        for xb, yb in loader:
            pass
        print(f"one train epoch of loading, num_workers={workers}: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
