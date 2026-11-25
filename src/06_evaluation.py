"""Stage 6 - Evaluation beyond a single AUC (no retraining; uses saved predictions).

1. Paired bootstrap: is AUC(a) - AUC(b) consistent across resamples of the SAME test images?
2. Confusion matrices at the val-chosen (Youden) threshold.
3. Error overlap: do different models fail on the same NORMAL test images?
4. Threshold rules: val Youden, val sensitivity >= 95 %, and test Youden (ORACLE: uses test labels,
   reported only to show what the val-chosen threshold costs; never a legitimate choice).

Usage (from project root): python src/06_evaluation.py
Output: outputs/06_evaluation/{paired_bootstrap.csv, thresholds.csv, error_overlap.csv,
        confusion_matrices.png, roc_test.png}
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import (paired_bootstrap_auc_diff, report, roc_curve, threshold_for_sensitivity,
                     youden_threshold)
from runs import RUNS, load_predictions

MODELS = ["smallcnn_noweight", "resnet18_frozen", "resnet18_ft", "resnet18_ft_clahe",
          "densenet121_ft", "efficientnet_b0_ft"]
OUT = Path("outputs/06_evaluation")


def load_all():
    """Wide tables (one column of scores per model), rows aligned on relpath."""
    wide = {}
    for split in ("val", "test"):
        frames = []
        for m in MODELS:
            pr = load_predictions(RUNS / m).query("split == @split")
            frames.append(pr.set_index("relpath")[["score"]].rename(columns={"score": m}))
            labels = pr.set_index("relpath")["label"]
        df = pd.concat(frames, axis=1, join="inner")
        assert len(df) == len(labels), f"{split}: models do not share the same images"
        df["label"] = labels.loc[df.index]
        wide[split] = df
    return wide


def paired_table(test: pd.DataFrame) -> pd.DataFrame:
    y = test["label"].to_numpy()
    rows = []
    for a, b in combinations(MODELS, 2):
        diff, lo, hi, p_le = paired_bootstrap_auc_diff(y, test[a], test[b])
        rows.append({"model_a": a, "model_b": b, "auc_diff": diff, "ci_low": lo, "ci_high": hi,
                     "consistent": "yes" if lo > 0 or hi < 0 else "no", "frac_a_le_b": p_le})
    return pd.DataFrame(rows)


def threshold_table(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        rules = {
            "val_youden": youden_threshold(val["label"], val[m]),
            "val_sens95": threshold_for_sensitivity(val["label"], val[m], 0.95),
            "ORACLE_test_youden": youden_threshold(test["label"], test[m]),
        }
        for rule, t in rules.items():
            r = report(test["label"], test[m], t)
            rows.append({"model": m, "rule": rule, "threshold": t, "test_sens": r["sensitivity"],
                         "test_spec": r["specificity"], "test_acc": r["accuracy"],
                         "fn": r["fn"], "fp": r["fp"]})
    return pd.DataFrame(rows)


def error_overlap(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """For each test NORMAL image: how many models call it PNEUMONIA at their val-Youden threshold."""
    normal = test[test["label"] == 0]
    wrong = pd.DataFrame({m: normal[m] >= youden_threshold(val["label"], val[m]) for m in MODELS})
    return wrong.assign(n_models_wrong=wrong.sum(axis=1))


def plot_confusions(val, test, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, m in zip(axes.ravel(), MODELS):
        r = report(test["label"], test[m], youden_threshold(val["label"], val[m]))
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NORMAL", "PNEUMONIA"],
               yticklabels=["NORMAL", "PNEUMONIA"], xlabel="predicted", ylabel="true",
               title=f"{m}\nsens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")
    plt.suptitle("Test confusion matrices at the val-chosen (Youden) threshold")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def plot_roc(val, test, out):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(test["label"], test[m])
        r = report(test["label"], test[m], youden_threshold(val["label"], val[m]))
        line, = ax.plot(fpr, tpr, label=f"{m} (AUC {r['roc_auc']:.3f})")
        ax.plot(1 - r["specificity"], r["sensitivity"], "o", color=line.get_color())
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(xlabel="1 - specificity (false positive rate)", ylabel="sensitivity (true positive rate)",
           title="Test ROC curves; dots = operating point at the val-chosen threshold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wide = load_all()
    val, test = wide["val"], wide["test"]

    paired = paired_table(test)
    paired.to_csv(OUT / "paired_bootstrap.csv", index=False)
    print("Paired bootstrap, test AUC(a) - AUC(b), 95% CI (2000 resamples):")
    print(paired.round(4).to_string(index=False))

    thr = threshold_table(val, test)
    thr.to_csv(OUT / "thresholds.csv", index=False)
    print("\nTest sensitivity/specificity under different threshold rules:")
    print(thr.round(3).to_string(index=False))

    ov = error_overlap(val, test)
    ov.to_csv(OUT / "error_overlap.csv")
    print(f"\nTest NORMAL images (n={len(ov)}) by number of models (of {len(MODELS)}) calling them PNEUMONIA:")
    print(ov["n_models_wrong"].value_counts().sort_index().to_string())

    plot_confusions(val, test, OUT / "confusion_matrices.png")
    plot_roc(val, test, OUT / "roc_test.png")


if __name__ == "__main__":
    main()
