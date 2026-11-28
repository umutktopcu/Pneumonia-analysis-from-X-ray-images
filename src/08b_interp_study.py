"""Stage 8b - Does the resize interpolation leak the original image size into the pixels?

The CNNs only see 224x224 images, yet image size alone gives test AUC 0.911. One candidate route: the
downsampling factor (larger NORMAL images are shrunk more) changes fine texture/noise. We measure, for
three interpolations, (1) how well fine texture separates the classes and (2) how strongly fine texture
follows the original image height WITHIN each class (a pure size leak, independent of the label).

Usage (project root): python src/08b_interp_study.py
Output: outputs/08_review/interp_study.csv
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import roc_auc  # noqa: E402

DATA = Path("data/chest_xray")
N_PER_CLASS, SEED, SIZE = 300, 42, 224
METHODS = {"INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR, "INTER_AREA": cv2.INTER_AREA}


def hf_energy(img: np.ndarray) -> float:
    """Mean absolute Laplacian of the 224x224 image: fine texture + noise."""
    return float(np.abs(cv2.Laplacian(img.astype(np.float32) / 255.0, cv2.CV_32F)).mean())


def main() -> None:
    sp = pd.read_csv("outputs/02_splits/splits.csv")
    train = sp[sp.split == "train"]
    sample = pd.concat([g.sample(N_PER_CLASS, random_state=SEED) for _, g in train.groupby("label")])
    y = (sample.label == "PNEUMONIA").to_numpy().astype(int)
    feats = {m: [] for m in METHODS}
    for rel in sample.relpath:
        img = cv2.imread(str(DATA / rel), cv2.IMREAD_GRAYSCALE)
        for name, flag in METHODS.items():
            feats[name].append(hf_energy(cv2.resize(img, (SIZE, SIZE), interpolation=flag)))
    rows = []
    for name, vals in feats.items():
        v = np.asarray(vals)
        auc = roc_auc(y, v)
        d = sample.assign(hf=v)
        rho = {lab: d[d.label == lab][["hf", "height"]].corr(method="spearman").iloc[0, 1] for lab in ["NORMAL", "PNEUMONIA"]}
        rows.append({"method": name, "class_separability": round(max(auc, 1 - auc), 3),
                     "rho_height_NORMAL": round(rho["NORMAL"], 2), "rho_height_PNEUMONIA": round(rho["PNEUMONIA"], 2)})
    out = pd.DataFrame(rows)
    Path("outputs/08_review").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/08_review/interp_study.csv", index=False)
    print(f"train sample: {N_PER_CLASS} per class, seed {SEED}; separability = max(AUC, 1-AUC), 0.5 = none")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
