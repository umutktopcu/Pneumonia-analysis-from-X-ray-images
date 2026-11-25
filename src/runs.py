"""Reading saved training runs (outputs/runs/<run>/predictions.csv)."""
from pathlib import Path

import pandas as pd

RUNS = Path("outputs/runs")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """predictions.csv plus `relpath` matching splits.csv (works for sq224 and sq224_clahe caches)."""
    pr = pd.read_csv(Path(run_dir) / "predictions.csv")
    pr["relpath"] = (pr["path"].str.replace(r"^.*?/sq224(_clahe)?/", "", regex=True)
                     .str.replace(".png", ".jpeg", regex=False))
    return pr
