"""Stage 3 - Build the preprocessed image cache used for training and evaluation.

Reads outputs/02_splits/splits.csv, preprocesses every non-excluded image with
src/preprocessing.py and writes lossless PNGs, keeping the original relative paths:
    data/processed/<variant>/<split folder>/<CLASS>/<name>.png
Variants: sq224 (default) and sq224_clahe (--clahe). Existing files are skipped
unless --overwrite, so the script can be re-run safely.

Usage (from project root):
    python src/03_preprocess.py            # -> data/processed/sq224/
    python src/03_preprocess.py --clahe    # -> data/processed/sq224_clahe/
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from preprocessing import IMG_SIZE, preprocess


def cached_path(out_root: Path, relpath: str) -> Path:
    return out_root / Path(relpath).with_suffix(".png")


def build_cache(df: pd.DataFrame, data_dir: Path, out_root: Path, clahe: bool, overwrite: bool) -> int:
    written = 0
    for i, relpath in enumerate(df["relpath"], start=1):
        dst = cached_path(out_root, relpath)
        if dst.exists() and not overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), preprocess(data_dir / relpath, clahe=clahe)):
            raise IOError(f"could not write {dst}")
        written += 1
        if i % 1000 == 0:
            print(f"  {i}/{len(df)}")
    return written


def verify_cache(df: pd.DataFrame, out_root: Path) -> None:
    """Every expected file exists and is a single-channel IMG_SIZE x IMG_SIZE uint8 image."""
    for relpath in df["relpath"]:
        img = cv2.imread(str(cached_path(out_root, relpath)), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"missing or unreadable: {relpath}"
        assert img.shape == (IMG_SIZE, IMG_SIZE) and img.dtype == "uint8", f"bad image: {relpath} {img.shape}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--splits", type=Path, default=Path("outputs/02_splits/splits.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    df = splits[splits["split"] != "excluded"]
    out_root = args.out_dir / (f"sq{IMG_SIZE}" + ("_clahe" if args.clahe else ""))

    start = time.time()
    written = build_cache(df, args.data_dir, out_root, args.clahe, args.overwrite)
    verify_cache(df, out_root)
    print(f"{out_root}: {written} written, {len(df)} verified ({time.time() - start:.0f}s)")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
