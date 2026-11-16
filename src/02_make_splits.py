"""Stage 2 - Leakage-aware train/val split (decision D2 in DECISIONS.md).

- The official test set is kept untouched (comparability + third-expert-checked labels).
- Training pool = official train + official val (16 images, distinct patients).
- Excluded from the pool:
    patient_in_test : images of patients whose ID also appears in the test set
    duplicate_md5   : exact copies of an image already kept (test copies take priority)
- The pool is split into train/val at PATIENT level, stratified by class.

Usage (from project root):
    python src/02_make_splits.py [--val-frac 0.2] [--seed 42]

Output (outputs/02_splits/):
    splits.csv   all images; `split` in {train, val, test, excluded} + `exclude_reason`
    summary.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 140)


def mark_exclusions(df: pd.DataFrame) -> pd.Series:
    reason = pd.Series("", index=df.index)
    is_test = df["source_split"] == "test"

    leaked = ~is_test & df["patient_id"].isin(set(df.loc[is_test, "patient_id"]))
    reason[leaked] = "patient_in_test"

    # Sort so test copies come first: the first copy of each file is kept, later pool copies dropped.
    order = df.assign(_rank=(~is_test).astype(int)).sort_values(["_rank", "relpath"])
    later_copy = order.duplicated("md5", keep="first").reindex(df.index)
    reason[later_copy & ~is_test & (reason == "")] = "duplicate_md5"
    return reason


def patient_split(pool: pd.DataFrame, val_frac: float, seed: int) -> pd.Series:
    """Assign whole patients to val (per class) until ~val_frac of that class's images is reached."""
    rng = np.random.default_rng(seed)
    val_patients = set()
    for _, group in pool.groupby("label"):
        counts = group.groupby("patient_id").size()   # sorted index -> deterministic before shuffling
        # permutation returns a copy; shuffling counts.index.to_numpy() in place would
        # corrupt the index itself (it is a view) and mismatch patients with their counts.
        patients = rng.permutation(counts.index.to_numpy())
        cum = counts.loc[patients].cumsum()
        before = cum.shift(fill_value=0)              # images already in val before adding this patient
        val_patients.update(cum.index[before < val_frac * len(group)])
    return pd.Series(np.where(pool["patient_id"].isin(val_patients), "val", "train"), index=pool.index)


def run_checks(df: pd.DataFrame, val_frac: float) -> None:
    kept = df[df["split"] != "excluded"]
    assert (kept.groupby("patient_id")["label"].nunique() == 1).all(), "a patient has two labels"
    assert (kept.groupby("patient_id")["split"].nunique() == 1).all(), "a patient appears in two splits"
    assert (kept.groupby("md5")["split"].nunique() == 1).all(), "an identical file appears in two splits"
    test = df[df["source_split"] == "test"]
    assert (test["split"] == "test").all(), "official test set was modified"
    assert not ((df["source_split"] != "test") & (df["split"] == "test")).any(), "pool image moved to test"
    # each class's val share must be within one patient of the target
    pool = df[df["split"].isin(["train", "val"])]
    max_imgs = pool.groupby("patient_id").size().max()
    for label, g in pool.groupby("label"):
        n_val = (g["split"] == "val").sum()
        assert abs(n_val - val_frac * len(g)) <= max_imgs, f"{label}: val size {n_val} far from target"


def summarize(df: pd.DataFrame) -> str:
    kept = df[df["split"] != "excluded"]
    order = ["train", "val", "test"]
    counts = pd.crosstab(kept["split"], kept["label"]).reindex(order)
    counts["total"] = counts.sum(axis=1)
    counts["pneumonia_ratio"] = (counts["PNEUMONIA"] / counts["total"]).round(3)
    counts["patients"] = kept.groupby("split")["patient_id"].nunique().reindex(order)
    parts = [
        "=== Exclusions from training pool ===", str(df["exclude_reason"].replace("", "kept").value_counts()),
        "\n=== Final splits ===", str(counts),
        "\n=== Subtypes per split ===", str(pd.crosstab(kept["split"], kept["subtype"]).reindex(order)),
        "\nAll leakage checks passed.",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=Path("outputs/01_exploration/metadata.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/02_splits"))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata).rename(columns={"split": "source_split"})
    if "md5" not in df:
        raise ValueError("metadata.csv has no md5 column: re-run stage 1 without --no-hash")
    # Unparsed filenames become their own single-image "patient" (none expected after stage 1 fixes).
    df["patient_id"] = df["patient_id"].fillna("file:" + df["relpath"])

    df["exclude_reason"] = mark_exclusions(df)
    df["split"] = "excluded"
    df.loc[df["source_split"] == "test", "split"] = "test"
    pool = df[(df["source_split"] != "test") & (df["exclude_reason"] == "")]
    df.loc[pool.index, "split"] = patient_split(pool, args.val_frac, args.seed)

    run_checks(df, args.val_frac)
    summary = summarize(df)
    print(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["relpath", "split", "source_split", "exclude_reason", "label", "subtype",
            "patient_id", "width", "height", "mode", "md5"]
    df[cols].to_csv(args.out_dir / "splits.csv", index=False)
    (args.out_dir / "summary.txt").write_text(summary)
    print(f"\nSaved {args.out_dir / 'splits.csv'}")


if __name__ == "__main__":
    main()
