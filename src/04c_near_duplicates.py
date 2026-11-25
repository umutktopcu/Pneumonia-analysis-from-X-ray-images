"""Stage 4c - Near-duplicate / leakage check between splits (perceptual hash).

MD5 (stage 2) only finds byte-identical files. A difference hash (dHash) also matches re-encoded or
slightly altered copies: shrink to (H+1)xH, record whether each pixel is brighter than its right
neighbour -> H*H bits; similar images have a small Hamming distance.

Lesson learned: an 8x8 hash (64 bits) is useless here - chest X-rays all look alike at that size
(random pairs differ by only ~11 bits). We use 16x16 (256 bits) and calibrate against random pairs.

Test: for every val image and every test image, the distance to its NEAREST train image.
Test patients are independent of train (after D2), so test acts as a control: if val were leaking
(same patients/images as train), val images would sit much closer to train than test images do.

Usage (from project root): python src/04c_near_duplicates.py
Output: outputs/04_baselines/{near_duplicates.txt, nearest_train_neighbours.csv, near_duplicates.png}
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("data/processed/sq224")
OUT = Path("outputs/04_baselines")
HASH = 16
SEED = 42


def dhash(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (HASH + 1, HASH), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def nearest(h_query: np.ndarray, h_ref: np.ndarray, chunk: int = 200):
    """Index and Hamming distance of the nearest reference hash for every query hash."""
    idx, dist = np.empty(len(h_query), int), np.empty(len(h_query), int)
    for i in range(0, len(h_query), chunk):
        d = (h_query[i:i + chunk, None, :] != h_ref[None, :, :]).sum(axis=2)
        idx[i:i + chunk], dist[i:i + chunk] = d.argmin(axis=1), d.min(axis=1)
    return idx, dist


def main():
    sp = pd.read_csv("outputs/02_splits/splits.csv").query("split != 'excluded'").reset_index(drop=True)
    H = np.stack([dhash(ROOT / Path(r).with_suffix(".png")) for r in sp["relpath"]])
    split = sp["split"].to_numpy()
    train = np.flatnonzero(split == "train")

    rng = np.random.default_rng(SEED)
    a, b = rng.choice(len(sp), 400, replace=False), rng.choice(len(sp), 400, replace=False)
    ref = (H[a][:, None, :] != H[b][None, :, :]).sum(axis=2)[a[:, None] != b[None, :]]
    lines = [f"dHash {HASH}x{HASH} = {HASH * HASH} bits",
             f"random pairs: min {ref.min()}, 1% {np.percentile(ref, 1):.0f}, median {np.median(ref):.0f}"]

    rows = []
    for s in ("val", "test"):
        q = np.flatnonzero(split == s)
        j, d = nearest(H[q], H[train])
        lines.append(f"{s:4s} -> nearest train: min {d.min()}, 1% {np.percentile(d, 1):.0f}, "
                     f"median {np.median(d):.0f}, below random-pair minimum: {(d < ref.min()).sum()}")
        rows.append(pd.DataFrame({"split": s, "relpath": sp.loc[q, "relpath"].to_numpy(),
                                  "label": sp.loc[q, "label"].to_numpy(),
                                  "nearest_train": sp.loc[train[j], "relpath"].to_numpy(),
                                  "nearest_train_label": sp.loc[train[j], "label"].to_numpy(), "dist": d}))
    nn = pd.concat(rows).sort_values("dist")
    OUT.mkdir(parents=True, exist_ok=True)
    nn.to_csv(OUT / "nearest_train_neighbours.csv", index=False)
    (OUT / "near_duplicates.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    show = nn.head(6)
    fig, axes = plt.subplots(len(show), 2, figsize=(6, 3 * len(show)))
    for r, row in enumerate(show.itertuples()):
        for c, (path, lab) in enumerate(((row.relpath, row.label), (row.nearest_train, row.nearest_train_label))):
            axes[r, c].imshow(cv2.imread(str(ROOT / Path(path).with_suffix(".png")), 0), cmap="gray")
            axes[r, c].set_title(f"{path.split('/')[0]}/{lab}  {Path(path).stem}\nd={row.dist}", fontsize=8)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "near_duplicates.png", dpi=100)


if __name__ == "__main__":
    main()
