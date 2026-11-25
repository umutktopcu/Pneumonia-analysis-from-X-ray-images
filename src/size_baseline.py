"""Image-size-only baseline (logistic regression on original width/height), shared by
04a_pixel_free_baselines.py and 05_compare_runs.py.

Its mistakes define the "shortcut-conflicting" images: cases where image size points to the wrong
class, so a model can only be right there if it uses something other than the size cue.
"""
import numpy as np
import pandas as pd

from metrics import youden_threshold


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


def fit_size_model(splits: pd.DataFrame):
    """Fit on train (features standardized with train statistics), threshold on val (Youden).
    Returns (scores per split, threshold)."""
    data = {s: splits[splits["split"] == s] for s in ("train", "val", "test")}
    y_train = (data["train"]["label"] == "PNEUMONIA").to_numpy().astype(int)
    X = {s: size_features(d) for s, d in data.items()}
    mu, sd = X["train"].mean(axis=0), X["train"].std(axis=0)
    X = {s: (x - mu) / sd for s, x in X.items()}
    w, b = fit_logistic_regression(X["train"], y_train)
    scores = {s: pd.Series(1 / (1 + np.exp(-(X[s] @ w + b))), index=data[s]["relpath"].to_numpy())
              for s in data}
    y_val = (data["val"]["label"] == "PNEUMONIA").to_numpy().astype(int)
    return scores, youden_threshold(y_val, scores["val"].to_numpy())


def conflicting_relpaths(splits: pd.DataFrame, split: str) -> set:
    """Images of `split` that the size-only model misclassifies at its val-chosen threshold."""
    scores, t = fit_size_model(splits)
    d = splits[splits["split"] == split]
    wrong = (scores[split].to_numpy() >= t).astype(int) != (d["label"] == "PNEUMONIA").to_numpy().astype(int)
    return set(d["relpath"][wrong])
