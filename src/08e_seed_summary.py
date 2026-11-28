"""Stage 8e - How much do results move with the training seed? (declared before the runs)

Purpose: measure training randomness, which the bootstrap CIs of Stages 5-6 do not contain. Nothing here
selects a model or turns CLAHE on: D7 requires a VAL improvement for CLAHE, and that rule stands.
Groups (seed 42 = the original Stage 5 run, seeds 1 and 2 = new runs, same settings):
  resnet18_ft, resnet18_ft_clahe, densenet121_ft          (early stopping on val AUC, as before)
  resnet18_ft_last                                        (fixed 10 epochs, final epoch kept: --select last)
For each run: val AUC, test AUC, test sensitivity/specificity at the D13 rule (val sensitivity >= 95 %).

Usage (project root): python src/08e_seed_summary.py
Output: outputs/08_review/seed_runs.csv, seed_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import report, roc_auc, threshold_for_sensitivity  # noqa: E402
from runs import RUNS, load_predictions  # noqa: E402

OUT = Path("outputs/08_review")
GROUPS = {
    "resnet18_ft": ["resnet18_ft", "resnet18_ft_s1", "resnet18_ft_s2"],
    "resnet18_ft_clahe": ["resnet18_ft_clahe", "resnet18_ft_clahe_s1", "resnet18_ft_clahe_s2"],
    "densenet121_ft": ["densenet121_ft", "densenet121_ft_s1", "densenet121_ft_s2"],
    "resnet18_ft_last": ["resnet18_ft_last_s42", "resnet18_ft_last_s1", "resnet18_ft_last_s2"],
}


def one_run(run: str) -> dict:
    pr = load_predictions(RUNS / run)
    val, te = pr[pr.split == "val"], pr[pr.split == "test"]
    thr = threshold_for_sensitivity(val.label.to_numpy(), val.score.to_numpy(), 0.95)
    r = report(te.label.to_numpy(), te.score.to_numpy(), thr)
    return {"run": run, "val_auc": roc_auc(val.label.to_numpy(), val.score.to_numpy()),
            "test_auc": r["roc_auc"], "test_sens": r["sensitivity"], "test_spec": r["specificity"],
            "test_acc": r["accuracy"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, runs in GROUPS.items():
        for seed, run in zip([42, 1, 2], runs):
            if not (RUNS / run / "predictions.csv").exists():
                print(f"missing (not run yet?): {run}")
                continue
            rows.append({"group": group, "seed": seed, **one_run(run)})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(OUT / "seed_runs.csv", index=False)
    summ = df.groupby("group")[["val_auc", "test_auc", "test_sens", "test_spec", "test_acc"]].agg(["mean", "std", "min", "max"])
    summ.round(4).to_csv(OUT / "seed_summary.csv")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False), "\n")
    print(summ.round(4).to_string(), "\n")
    w = df.pivot(index="seed", columns="group", values="test_auc")
    if {"resnet18_ft", "resnet18_ft_clahe"} <= set(w.columns):
        d = (w["resnet18_ft_clahe"] - w["resnet18_ft"]).dropna()
        print(f"CLAHE - plain test AUC per seed: {d.round(4).to_dict()} (Stage 6 single-seed: +0.010)")


if __name__ == "__main__":
    main()
