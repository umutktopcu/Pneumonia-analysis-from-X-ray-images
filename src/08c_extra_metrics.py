"""Stage 8c - Metrics we did not report before: Wilson CIs at the operating point, calibration, subtypes.

All numbers on TEST at the D13 threshold (from outputs/06_evaluation/thresholds.csv, chosen on val).
These are exploratory/descriptive additions: nothing here is used to choose a model or a threshold.

- Wilson 95 % interval for sensitivity, specificity and precision (binomial proportions; better than
  p +/- 1.96*SE when p is close to 0 or 1, which is exactly our case).
- Calibration: Brier score (mean squared error of the probability) and ECE (expected calibration error,
  10 equal-width bins) on val and test, plus a reliability diagram for test.
- Sensitivity per pneumonia subtype (bacterial / viral).

Usage (project root): python src/08c_extra_metrics.py
Output: outputs/08_review/{extra_metrics.csv, subtype_sensitivity.csv, reliability_test.png}
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from runs import RUNS, load_predictions  # noqa: E402

OUT = Path("outputs/08_review")
MODELS = ["smallcnn_noweight", "resnet18_frozen", "resnet18_ft", "resnet18_ft_clahe",
          "densenet121_ft", "efficientnet_b0_ft"]
N_BINS = 10


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion k/n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def calibration(y: np.ndarray, p: np.ndarray) -> tuple:
    """Brier score, ECE and the per-bin (mean predicted, observed rate, count) table."""
    brier = float(np.mean((p - y) ** 2))
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    table = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ece = float((table.n * (table.p - table.y).abs()).sum() / len(y))
    return brier, ece, table


def _self_test() -> None:
    lo, hi = wilson(194, 234)
    assert 0.77 < lo < 0.79 and 0.87 < hi < 0.88, (lo, hi)
    b, e, _ = calibration(np.array([0, 1]), np.array([0.0, 1.0]))
    assert b == 0 and e == 0


def main() -> None:
    _self_test()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv("outputs/02_splits/splits.csv")[["relpath", "subtype"]]
    thr = pd.read_csv("outputs/06_evaluation/thresholds.csv")
    thr = thr[thr.rule == "val_sens95"].set_index("model").threshold
    rows, sub_rows = [], []
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect calibration")
    for m in MODELS:
        pr = load_predictions(RUNS / m).merge(sp, on="relpath", how="left")
        assert pr.subtype.notna().all(), m
        val, te = pr[pr.split == "val"], pr[pr.split == "test"]
        y, s = te.label.to_numpy(), te.score.to_numpy()
        pred = (s >= thr[m]).astype(int)
        tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
        tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
        b_val, e_val, _ = calibration(val.label.to_numpy(), val.score.to_numpy())
        b_te, e_te, table = calibration(y, s)
        rows.append({"model": m, "threshold": round(float(thr[m]), 4),
                     "sensitivity [95% Wilson]": fmt(tp, tp + fn), "specificity [95% Wilson]": fmt(tn, tn + fp),
                     "precision [95% Wilson]": fmt(tp, tp + fp),
                     "brier_val": round(b_val, 4), "brier_test": round(b_te, 4),
                     "ece_val": round(e_val, 4), "ece_test": round(e_te, 4),
                     "share_test_scores_<0.05_or_>0.95": round(float(np.mean((s < 0.05) | (s > 0.95))), 3)})
        for st, g in te[te.label == 1].groupby("subtype"):
            k = int((g.score >= thr[m]).sum())
            sub_rows.append({"model": m, "subtype": st, "detected": k, "n": len(g), "sensitivity [95% Wilson]": fmt(k, len(g))})
        line, = ax.plot(table.p, table.y, lw=1.0, alpha=0.7, label=m)
        ax.scatter(table.p, table.y, s=8 + 2 * np.sqrt(table.n) * 4, color=line.get_color(), alpha=0.8)
    ax.set(xlabel="mean predicted P(pneumonia) in bin", ylabel="observed share of pneumonia",
           title="Reliability diagram, official test set (10 bins)\nmarker area ~ images in bin; middle bins hold few images")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_test.png", dpi=150)
    pd.DataFrame(rows).to_csv(OUT / "extra_metrics.csv", index=False)
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(OUT / "subtype_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(sub.pivot(index="model", columns="subtype", values="sensitivity [95% Wilson]").to_string())


if __name__ == "__main__":
    main()
