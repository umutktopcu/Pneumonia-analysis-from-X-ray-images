"""Stage 7a - Detect the burned-in "KV.." exposure label (bottom-right corner) in every image.

Grad-CAM of ResNet-18 pointed at the bottom-right corner of NORMAL test images that all six models
call PNEUMONIA; those corners contain text like "KV59" (an exposure setting printed on the film).
This script measures how often that label occurs per split and class.

Method (classic CV, course Hafta 6: thresholding, template matching):
  corner  = bottom 20 % x right 30 % of the ORIGINAL image, resized to 480 x 300, binarized (>= 200)
  template= binarized "KV" glyphs cut from ONE test NORMAL image whose corner holds only the text on a
            black background (no clean crop could be cut automatically from train images: bright bone
            and tissue edges fall in the same box). Only the glyph shapes are used, not the image's label
            or any model output; the template image is excluded from the validation below.
  score   = max normalized cross-correlation over 7 template scales; label = score > THRESHOLD
Validation: hand labels of 53 test NORMAL images (outputs/07_gradcam/kv_manual_labels.csv, template excluded).
Spot check of detections: outputs/07_gradcam/kv_detector_check.png (12/12 random hits were real labels;
2 of the 6 lowest-scoring hits were not, so counts may be slightly inflated).

Usage (from project root): python src/07a_kv_labels.py
Output: outputs/07_gradcam/{kv_labels.csv, kv_template.png}; kv_labels.csv also stores the match box
in 224x224 processed-image coordinates (for the masking experiment in 07b).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data/chest_xray")
OUT = Path("outputs/07_gradcam")
TEMPLATE_IMAGE = "test/NORMAL/NORMAL2-IM-0221-0001.jpeg"
CORNER = (0.80, 0.70)          # start of the corner crop: 80 % of height, 70 % of width
CORNER_SIZE = (480, 300)       # (width, height) after resizing the crop
SCALES = (0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.7)
THRESHOLD = 0.47


def corner(path: str):
    img = cv2.imread(str(DATA / path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    y0, x0 = int(CORNER[0] * h), int(CORNER[1] * w)
    crop = cv2.resize(img[y0:, x0:], CORNER_SIZE, interpolation=cv2.INTER_AREA)
    return crop, (h, w, y0, x0)


def make_template() -> np.ndarray:
    crop, _ = corner(TEMPLATE_IMAGE)
    sub = crop[150:, 240:]                       # lower-right quarter of the corner: where the text sits
    ys, xs = np.nonzero(sub >= 200)
    box = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return (box[:, : box.shape[1] // 2] >= 200).astype(np.float32)   # keep "KV", drop the digits


TEMPLATE = make_template()


def detect(path: str) -> dict:
    crop, (h, w, y0, x0) = corner(path)
    binary = (crop >= 200).astype(np.float32)
    best = {"kv_score": -1.0, "box": None}
    for s in SCALES:
        t = cv2.resize(TEMPLATE, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        if t.shape[0] >= binary.shape[0] or t.shape[1] >= binary.shape[1] or t.sum() == 0:
            continue
        res = cv2.matchTemplate(binary, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(np.nan_to_num(res))  # all-black crop -> NaN scores
        if score > best["kv_score"]:
            # box covering "KV" plus the digits to its right (~2.2x template width), in crop coordinates
            best = {"kv_score": float(score), "box": (x, y, int(2.2 * t.shape[1]), t.shape[0])}
    if best["box"] is None:  # no usable scale at all
        return {"kv_score": 0.0, "box_x0": np.nan, "box_y0": np.nan, "box_x1": np.nan, "box_y1": np.nan}
    x, y, bw, bh = best["box"]
    sx, sy = (w - x0) / CORNER_SIZE[0], (h - y0) / CORNER_SIZE[1]    # crop -> original pixels
    to224 = lambda v, full: v * 224 / full                            # original -> squashed 224
    return {"kv_score": best["kv_score"],
            "box_x0": to224(x0 + x * sx, w), "box_y0": to224(y0 + y * sy, h),
            "box_x1": to224(x0 + (x + bw) * sx, w), "box_y1": to224(y0 + (y + bh) * sy, h)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "kv_template.png"), (TEMPLATE * 255).astype(np.uint8))
    sp = pd.read_csv("outputs/02_splits/splits.csv").query("split != 'excluded'").reset_index(drop=True)
    with ProcessPoolExecutor(4) as ex:
        found = pd.DataFrame(list(ex.map(detect, sp["relpath"], chunksize=64)))
    sp = pd.concat([sp[["relpath", "split", "label"]], found], axis=1)
    sp["kv_label"] = sp["kv_score"] > THRESHOLD
    sp.to_csv(OUT / "kv_labels.csv", index=False)

    manual = pd.read_csv(OUT / "kv_manual_labels.csv").merge(sp[["relpath", "kv_score"]], on="relpath")
    manual = manual[manual["relpath"] != TEMPLATE_IMAGE]
    print(f"template {TEMPLATE.shape} from {TEMPLATE_IMAGE}")
    print("validation on hand-labelled test images (kv_score):")
    print(manual.groupby("kv_visible")["kv_score"].agg(["count", "min", "median", "max"]).round(3).to_string())
    agree = ((manual["kv_score"] > THRESHOLD) == manual["kv_visible"]).mean()
    print(f"agreement with hand labels at threshold {THRESHOLD}: {agree:.3f}\n")
    print(pd.crosstab([sp["split"], sp["label"]], sp["kv_label"]).rename(columns={True: "KV", False: "no KV"}))


if __name__ == "__main__":
    main()
