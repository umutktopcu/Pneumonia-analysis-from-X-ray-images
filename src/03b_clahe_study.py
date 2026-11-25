"""Stage 3b - CLAHE study (exploratory; informs decision D7).

Compares the sq224 and sq224_clahe caches on the TRAIN split:
  - visual preview of the same images with and without CLAHE
  - separability of non-anatomical statistics (does CLAHE amplify or remove class cues?)

Usage (from project root, after building both caches with 03_preprocess.py):
    python src/03b_clahe_study.py
Output: outputs/03_preprocessing/{clahe_study.txt, clahe_preview.png}
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shortcut_checks import FEATURES, features, separability

SEED = 42
VARIANTS = ["sq224", "sq224_clahe"]


def read(root: Path, relpath: str):
    return cv2.imread(str(root / Path(relpath).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/02_splits/splits.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/03_preprocessing"))
    args = parser.parse_args()

    train = pd.read_csv(args.splits).query("split == 'train'")
    y = (train["label"] == "PNEUMONIA").to_numpy().astype(int)

    table = {}
    for v in VARIANTS:
        feats = pd.DataFrame([features(read(args.processed / v, p)) for p in train["relpath"]])
        table[v] = {c: separability(feats[c].to_numpy(), y) for c in FEATURES}
    table = pd.DataFrame(table).T.round(3)
    text = (f"Separability max(AUC, 1-AUC) on the full train split (n={len(train)}). "
            f"0.5 = no class info.\n\n{table.to_string()}")
    print(text)
    (args.out_dir / "clahe_study.txt").write_text(text)

    picks = pd.concat([g.sample(3, random_state=SEED) for _, g in train.groupby("label")])
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6.4))
    for c, rec in enumerate(picks.itertuples()):
        for r, v in enumerate(VARIANTS):
            axes[r, c].imshow(read(args.processed / v, rec.relpath), cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(f"{rec.label} / {v}", fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(args.out_dir / "clahe_preview.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
