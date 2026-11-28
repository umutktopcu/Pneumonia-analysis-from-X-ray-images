"""Stage 8a - Visual audit of the KV detector on TRAIN/VAL NORMAL images.

Why: the claim "the KV label never occurs on train/val NORMAL images" (Stage 7) rests on a detector whose
threshold (0.47) was validated on test images only. Train NORMAL scores reach 0.459, just under it.
Here we look at the highest-scoring train/val NORMAL corners by eye, next to known KV examples.

Usage (project root): python src/08a_kv_train_review.py
Output: outputs/08_review/kv_review_{1,2}.png  (50 crops each, rank + score + file name)
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data/chest_xray")
OUT = Path("outputs/08_review")
N_TOP, PER_SHEET, COLS = 100, 50, 5
TILE = (300, 190)  # (w, h) of one tile incl. caption


def text_region(relpath: str) -> np.ndarray:
    """Bottom 20 % x right 30 % of the original image: the full region the 07a detector searches."""
    img = cv2.imread(str(DATA / relpath), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    crop = img[int(0.80 * h):, int(0.70 * w):]
    return cv2.resize(crop, (TILE[0], TILE[1] - 30), interpolation=cv2.INTER_AREA)


def tile(relpath: str, caption: str) -> np.ndarray:
    t = np.full((TILE[1], TILE[0]), 255, np.uint8)
    t[30:] = text_region(relpath)
    cv2.putText(t, caption, (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=128)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kv = pd.read_csv("outputs/07_gradcam/kv_labels.csv")
    normals = kv[kv.split.isin(["train", "val"]) & (kv.label == "NORMAL")]
    top = normals.sort_values("kv_score", ascending=False).head(N_TOP).reset_index(drop=True)
    refs = kv[(kv.split == "test") & (kv.label == "NORMAL") & kv.kv_label].sort_values("kv_score").head(COLS)
    ref_row = [tile(r.relpath, f"REF KV test {r.kv_score:.2f}") for r in refs.itertuples()]
    for s in range(N_TOP // PER_SHEET):
        chunk = top.iloc[s * PER_SHEET:(s + 1) * PER_SHEET]
        tiles = [tile(r.relpath, f"#{i + 1} {r.kv_score:.3f} {Path(r.relpath).stem[:14]}")
                 for i, r in zip(chunk.index, chunk.itertuples())]
        rows = [np.hstack(ref_row)] + [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
        cv2.imwrite(str(OUT / f"kv_review_{s + 1}.png"), np.vstack(rows))
    top[["relpath", "split", "kv_score"]].to_csv(OUT / "kv_review_top100.csv", index=True, index_label="rank0")
    print(f"wrote {N_TOP // PER_SHEET} sheets; score range of reviewed images: "
          f"{top.kv_score.min():.3f}-{top.kv_score.max():.3f}")


if __name__ == "__main__":
    main()
