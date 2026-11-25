"""Stage 5 - Compare all training runs under the same protocol (D10, D12).

Reads outputs/runs/<run>/{predictions.csv, metrics.csv} and reports per run:
  VAL  (used for model selection): ROC-AUC, AUC on the shortcut-conflicting subset, BCE loss
  TEST (reported only)           : AUC with 95% bootstrap CI, sensitivity/specificity at the val
                                   threshold, AUC on the shortcut-conflicting subset

Usage (from project root): python src/05_compare_runs.py
Output: outputs/05_transfer/comparison.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import bootstrap_auc_ci, roc_auc
from runs import RUNS, load_predictions
from size_baseline import conflicting_relpaths

OUT = Path("outputs/05_transfer/comparison.csv")


def bce(y, p, eps=1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    splits = pd.read_csv("outputs/02_splits/splits.csv")
    conflict = {s: conflicting_relpaths(splits, s) for s in ("val", "test")}
    rows = []
    for run_dir in sorted(p for p in RUNS.iterdir() if (p / "predictions.csv").exists()):
        pr, met = load_predictions(run_dir), pd.read_csv(run_dir / "metrics.csv").set_index("split")
        row = {"run": run_dir.name}
        for s in ("val", "test"):
            d = pr[pr["split"] == s]
            assert len(d) == (splits["split"] == s).sum(), f"{run_dir.name}: {s} size mismatch"
            y, sc = d["label"].to_numpy(), d["score"].to_numpy()
            c = d[d["relpath"].isin(conflict[s])]
            assert len(c) == len(conflict[s]), f"{run_dir.name}: conflicting subset not matched"
            row[f"{s}_auc"] = roc_auc(y, sc)
            row[f"{s}_conflict_auc"] = roc_auc(c["label"].to_numpy(), c["score"].to_numpy())
            if s == "val":
                row["val_loss"] = bce(y, sc)
            else:
                lo, hi = bootstrap_auc_ci(y, sc)
                row["test_auc_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                row["test_sens"], row["test_spec"] = met.loc["test", "sensitivity"], met.loc["test", "specificity"]
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    n_c = {s: len(v) for s, v in conflict.items()}
    print(f"shortcut-conflicting images: val {n_c['val']}, test {n_c['test']}\n")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
