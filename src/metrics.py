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


def roc_curve(y_true, scores):
    """False-positive rate and true-positive rate for every distinct threshold (highest first)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(s)[::-1]
    tpr = np.array([((s >= t) & (y == 1)).sum() for t in thresholds]) / y.sum()
    fpr = np.array([((s >= t) & (y == 0)).sum() for t in thresholds]) / (len(y) - y.sum())
    return np.r_[0.0, fpr], np.r_[0.0, tpr], thresholds


def threshold_for_sensitivity(y_true, scores, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity is still >= target (maximizes specificity under that
    constraint). Choose it on VALIDATION data only."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = np.sort(s[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))  # need at least k positives at or above the threshold
    return float(pos[k - 1])


def _stratified_indices(y, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]


def bootstrap_auc_ci(y_true, scores, n: int = 2000, seed: int = 42):
    """95% CI of ROC-AUC; positives and negatives resampled separately (class balance kept)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = [roc_auc(y[i], s[i]) for i in (_stratified_indices(y, rng) for _ in range(n))]
    return np.percentile(aucs, [2.5, 97.5])


def paired_bootstrap_auc_diff(y_true, scores_a, scores_b, n: int = 2000, seed: int = 42):
    """AUC(a) - AUC(b) on the SAME resampled images each round. Returns (diff, ci_low, ci_high,
    fraction of rounds where a <= b). A CI that excludes 0 indicates a consistent difference."""
    y = np.asarray(y_true).astype(int)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = _stratified_indices(y, rng)
        diffs.append(roc_auc(y[i], a[i]) - roc_auc(y[i], b[i]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(roc_auc(y, a) - roc_auc(y, b)), float(lo), float(hi), float((diffs <= 0).mean())


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
    # ROC curve: area under (fpr, tpr) must equal the rank-based AUC
    rng = np.random.default_rng(0)
    yy, ss = rng.integers(0, 2, 300), rng.random(300).round(2)  # rounding creates ties
    fpr, tpr, _ = roc_curve(yy, ss)
    assert abs(np.trapezoid(tpr, fpr) - roc_auc(yy, ss)) < 1e-9
    # sensitivity target: positives scored .9 .8 .7 .6 -> 75 % sensitivity needs threshold .7
    assert threshold_for_sensitivity([1, 1, 1, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5], 0.75) == 0.7
    # paired bootstrap of a model against itself: difference exactly 0
    d = paired_bootstrap_auc_diff(yy, ss, ss, n=50)
    assert d[0] == d[1] == d[2] == 0.0
    print("metrics self-test passed")


if __name__ == "__main__":
    _self_test()
