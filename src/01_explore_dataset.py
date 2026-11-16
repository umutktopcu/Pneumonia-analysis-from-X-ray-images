"""Stage 1 - Dataset exploration for Kaggle "Chest X-Ray Images (Pneumonia)".

Usage (from project root):
    python src/01_explore_dataset.py --data-dir data/chest_xray

Expects: <data-dir>/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

Outputs (outputs/01_exploration/):
    metadata.csv            one row per image; reused in later stages for splitting
    summary.txt             class counts, image properties, patient overlap, duplicates
    class_distribution.png
    image_sizes.png
    samples.png
    intensity_by_class.png
"""
import argparse
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMG_EXTS = {".jpeg", ".jpg", ".png"}
SEED = 42

# Filename conventions of this dataset. patient_id is a heuristic derived from the name,
# not official metadata: verify it before relying on it for patient-level splitting.
PNEUMONIA_RE = re.compile(r"^person(\d+)_(bacteria|virus)_\d+(?:_\d+)?$", re.IGNORECASE)
NORMAL_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)(?:-\d+){1,2}$", re.IGNORECASE)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def parse_filename(stem: str, label: str):
    """Return (patient_id, subtype) or (None, 'unknown') if the name is unrecognised."""
    if label == "PNEUMONIA":
        m = PNEUMONIA_RE.match(stem)
        if m:
            return f"P{m.group(1)}", m.group(2).lower()
    else:
        m = NORMAL_RE.match(stem)
        if m:
            prefix = "N2" if m.group(1) else "N"
            return f"{prefix}-{m.group(2)}", "normal"
    return None, "unknown"


def scan_dataset(data_dir: Path, compute_hash: bool) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in CLASSES:
            folder = data_dir / split / label
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing folder: {folder}\n--data-dir must point at the folder that "
                    "directly contains train/ val/ test/ (the Kaggle zip has a nested copy)."
                )
            for path in sorted(folder.iterdir()):
                # skip macOS metadata such as .DS_Store and ._*.jpeg
                if path.name.startswith(".") or path.suffix.lower() not in IMG_EXTS:
                    continue
                patient_id, subtype = parse_filename(path.stem, label)
                with Image.open(path) as img:  # lazy: reads only the header, not pixels
                    width, height = img.size
                    mode = img.mode
                row = {
                    "relpath": path.relative_to(data_dir).as_posix(),
                    "split": split,
                    "label": label,
                    "patient_id": patient_id,
                    "subtype": subtype,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
                if compute_hash:
                    row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    lines = []

    def section(title, obj):
        lines.extend([f"\n=== {title} ===", str(obj)])

    counts = pd.crosstab(df["split"], df["label"], margins=True).reindex([*SPLITS, "All"])
    counts["pneumonia_ratio"] = (counts["PNEUMONIA"] / counts["All"]).round(3)
    section("Class counts per split", counts)
    section("Subtypes per split", pd.crosstab(df["split"], df["subtype"]).reindex(SPLITS))

    unknown = df[df["subtype"] == "unknown"]
    section(
        "Filenames not matching known patterns",
        f"{len(unknown)} files" + (f", e.g. {unknown['relpath'].head(5).tolist()}" if len(unknown) else ""),
    )

    section("Image size by class (pixels)",
            df.groupby("label")[["width", "height"]].agg(["min", "median", "max"]))
    section("Color mode by class", pd.crosstab(df["label"], df["mode"]))

    known = df.dropna(subset=["patient_id"])
    per_patient = known.groupby(["label", "patient_id"]).size()
    section("Images per patient by class",
            per_patient.groupby("label").describe()[["count", "mean", "50%", "max"]].round(2))

    ids = {s: set(known.loc[known["split"] == s, "patient_id"]) for s in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    section("Patient IDs shared between splits (leakage risk)",
            {f"{a} & {b}": len(ids[a] & ids[b]) for a, b in pairs})

    if "md5" in df:
        dup = df[df.duplicated("md5", keep=False)]
        groups = dup.groupby("md5")
        section("Exact duplicate files (MD5)", {
            "duplicate_groups": groups.ngroups,
            "groups_spanning_splits": int(sum(g["split"].nunique() > 1 for _, g in groups)),
            "groups_with_conflicting_labels": int(sum(g["label"].nunique() > 1 for _, g in groups)),
        })

    return "\n".join(lines)


def load_gray(path: Path, size=None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        if size:
            img = img.resize(size)
        return np.asarray(img, dtype=np.float32) / 255.0


def plot_class_distribution(df, out):
    counts = pd.crosstab(df["split"], df["label"]).reindex(SPLITS)
    ax = counts.plot(kind="bar", figsize=(7, 4), rot=0)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution per split")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_image_sizes(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, g in df.groupby("label"):
        axes[0].scatter(g["width"], g["height"], s=4, alpha=0.4, label=label)
        axes[1].hist(g["width"] / g["height"], bins=50, alpha=0.6, density=True, label=label)
    axes[0].set(xlabel="width (px)", ylabel="height (px)", title="Image dimensions")
    axes[1].set(xlabel="aspect ratio (w/h)", ylabel="density", title="Aspect ratio by class")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_samples(df, data_dir, out, n=5):
    train = df[df["split"] == "train"]
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(3 * n, 6.5), squeeze=False)
    for r, label in enumerate(CLASSES):
        group = train[train["label"] == label]
        picks = group.sample(min(n, len(group)), random_state=SEED)
        for c, (_, row) in enumerate(picks.iterrows()):
            axes[r, c].imshow(load_gray(data_dir / row["relpath"]), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{label} / {row['subtype']}\n{row['width']}x{row['height']}", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_intensity(df, data_dir, out, n_per_class=300, size=(256, 256)):
    """Mean image and per-image brightness per class: a quick check for shortcut cues."""
    train = df[df["split"] == "train"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, label in enumerate(CLASSES):
        group = train[train["label"] == label]
        picks = group.sample(min(n_per_class, len(group)), random_state=SEED)
        imgs = np.stack([load_gray(data_dir / p, size) for p in picks["relpath"]])
        axes[i].imshow(imgs.mean(axis=0), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Mean image: {label} (n={len(imgs)})")
        axes[i].axis("off")
        axes[2].hist(imgs.mean(axis=(1, 2)), bins=40, alpha=0.6, density=True, label=label)
    axes[2].set(xlabel="mean pixel intensity per image", ylabel="density", title="Brightness by class")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/01_exploration"))
    parser.add_argument("--no-hash", action="store_true", help="skip MD5 duplicate check (faster)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(args.data_dir, compute_hash=not args.no_hash)
    df.to_csv(args.out_dir / "metadata.csv", index=False)

    summary = summarize(df)
    print(summary)
    (args.out_dir / "summary.txt").write_text(summary)

    plot_class_distribution(df, args.out_dir / "class_distribution.png")
    plot_image_sizes(df, args.out_dir / "image_sizes.png")
    plot_samples(df, args.data_dir, args.out_dir / "samples.png")
    plot_intensity(df, args.data_dir, args.out_dir / "intensity_by_class.png")
    print(f"\nSaved {len(df)} rows and figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
