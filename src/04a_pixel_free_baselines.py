"""Stage 4a - Pixel-free baselines: how far can we get WITHOUT looking at the lungs?

  majority      : always predict PNEUMONIA (score = 1 for every image)
  metadata_only : logistic regression on the ORIGINAL image size (log width, log height)

Model fitted on train, decision threshold chosen on val (Youden), metrics reported on val and test.
The metadata model measures how much class information the image-size shortcut carries; it is an
alarm level for later CNN results, not a strict lower bound (the CNN never sees the raw size).

Usage (from project root): python src/04a_pixel_free_baselines.py
Output: outputs/04_baselines/pixel_free_baselines.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import report, youden_threshold

OUT = Path("outputs/04_baselines/pixel_free_baselines.csv")


def size_features(df: pd.DataFrame) -> np.ndarray:
    # log(w/h) = log w - log h is a linear combination of these two, so the aspect ratio is
    # already representable; adding it as a third feature would be redundant (collinear).
    return np.column_stack([np.log(df["width"]), np.log(df["height"])])


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr=0.1, epochs=5000, l2=1e-3):
    """Plain batch gradient descent on the mean binary cross-entropy (+ small L2 penalty)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))          # sigmoid -> P(PNEUMONIA)
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)  # gradient of BCE w.r.t. w is X^T (p - y) / n
        b -= lr * (p - y).mean()
    return w, b


def main():
    splits = pd.read_csv("outputs/02_splits/splits.csv")
    data = {s: splits[splits["split"] == s] for s in ("train", "val", "test")}
    y = {s: (d["label"] == "PNEUMONIA").to_numpy().astype(int) for s, d in data.items()}

    # Standardize with TRAIN statistics only (using val/test statistics would leak information).
    X = {s: size_features(d) for s, d in data.items()}
    mu, sd = X["train"].mean(axis=0), X["train"].std(axis=0)
    X = {s: (x - mu) / sd for s, x in X.items()}
    w, b = fit_logistic_regression(X["train"], y["train"])
    print(f"logistic regression weights (standardized log width, log height): {w.round(3)}, bias {b:.3f}")

    scores = {
        "majority": {s: np.ones(len(d)) for s, d in data.items()},
        "metadata_only": {s: 1 / (1 + np.exp(-(X[s] @ w + b))) for s in data},
    }
    rows = []
    for model, sc in scores.items():
        t = youden_threshold(y["val"], sc["val"]) if model != "majority" else 0.5
        for split in ("val", "test"):
            rows.append({"model": model, "split": split, **report(y[split], sc[split], t)})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    cols = ["model", "split", "threshold", "accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
