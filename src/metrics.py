"""Binary classification metrics. Positive class = PNEUMONIA (label 1).

Written with numpy only so every formula is visible. Run `python src/metrics.py` to execute
the self-test against hand-computed values.
"""
import numpy as np
import pandas as pd


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def confusion(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),  # pneumonia, called pneumonia
        "fn": int(((y == 1) & (p == 0)).sum()),  # pneumonia, MISSED  (most costly clinically)
        "tn": int(((y == 0) & (p == 0)).sum()),  # normal, called normal
        "fp": int(((y == 0) & (p == 1)).sum()),  # normal, false alarm
    }


def roc_auc(y_true, scores) -> float:
    """P(score of a random positive > score of a random negative); ties count 1/2 (Mann-Whitney U)."""
    y = np.asarray(y_true).astype(int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank().to_numpy()  # average ranks handle ties
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(y_true, scores, threshold: float) -> dict:
    """All metrics at one decision threshold: predict PNEUMONIA when score >= threshold."""
    c = confusion(y_true, np.asarray(scores) >= threshold)
    sens = _div(c["tp"], c["tp"] + c["fn"])
    spec = _div(c["tn"], c["tn"] + c["fp"])
    prec = _div(c["tp"], c["tp"] + c["fp"])
    return {
        "threshold": float(threshold),
        **c,
        "accuracy": _div(c["tp"] + c["tn"], sum(c.values())),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": _div(2 * prec * sens, prec + sens),
        "balanced_acc": (sens + spec) / 2,
        "roc_auc": roc_auc(y_true, scores),
    }


def youden_threshold(y_true, scores) -> float:
    """Threshold maximizing sensitivity + specificity - 1. Choose it on VALIDATION data only."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    candidates = np.unique(s)
    j = [((s >= t) & (y == 1)).sum() / y.sum() + ((s < t) & (y == 0)).sum() / (len(y) - y.sum()) - 1
         for t in candidates]
    return float(candidates[int(np.argmax(j))])


def _self_test():
    # y=[1,1,0,0], scores=[.9,.4,.6,.1], threshold .5 -> TP=1 FN=1 TN=1 FP=1
    r = report([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert (r["tp"], r["fn"], r["tn"], r["fp"]) == (1, 1, 1, 1)
    assert r["sensitivity"] == r["specificity"] == r["precision"] == r["f1"] == r["accuracy"] == 0.5
    # positive-negative pairs: (.9>.6) (.9>.1) (.4<.6) (.4>.1) -> 3/4
    assert r["roc_auc"] == 0.75
    assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5        # all ties -> chance
    assert roc_auc([1, 1, 0], [0.9, 0.8, 0.1]) == 1.0                 # perfect ranking
    assert youden_threshold([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]) == 0.8  # lowest t separating perfectly
    always_pos = report([1, 1, 0], [1, 1, 1], 0.5)                     # majority-style classifier
    assert always_pos["sensitivity"] == 1 and always_pos["specificity"] == 0
    print("metrics self-test passed")


if __name__ == "__main__":
    _self_test()
