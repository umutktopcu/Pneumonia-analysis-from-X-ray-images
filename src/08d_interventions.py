"""Stage 8d - Cleaner interventions on the KV subgroup (declared before running; no retraining).

Problem with Stage 7b: painting the KV box black also ADDS an artificial black patch, and black patches
alone raise pneumonia scores on control images (e.g. ResNet-18 FT, NORMAL controls: median 0.117 -> 0.348).
"Erasing the text changed little" was therefore confounded. Three arms, all on the official test set,
all at the D13 threshold, all on already-trained checkpoints:

  erase_bg : NORMAL + KV images; only the bright glyph pixels are removed and filled from their
             surroundings (OpenCV Telea inpainting) -> no new black patch.
  paste    : NORMAL images WITHOUT KV; the glyph pixels of a KV image (donor, random, seed 42) are
             overlaid (pixel-wise maximum) at the donor's box position. Only glyph pixels change.
  darken   : NORMAL + KV images; intensities scaled so each image's mean equals the median mean of the
             NORMAL test images without KV (tests the "brighter subgroup" part of the hypothesis).

Declared expectations:
  - If the TEXT drives the error: erase_bg lowers calls on KV normals clearly, paste raises calls on clean
    normals clearly (say > 10 % of them flip for the fine-tuned models).
  - If the subgroup's BRIGHTNESS drives it: darken lowers calls on KV normals clearly.
  - If neither moves much, the cause is something else in that acquisition setting (field of view,
    sharpness, label noise) that this data cannot isolate.
Small n (18 KV normals): results are descriptive, not tests; they choose nothing.

Usage (project root, venv active):
  python src/08d_interventions.py --dry-run   # builds the edited images + example panel only (no torch)
  python src/08d_interventions.py             # scores every model, writes the summary
Output: outputs/08_review/{interventions.csv, interventions_summary.txt, interventions_examples.png}
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import apply_clahe  # noqa: E402

OUT = Path("outputs/08_review")
ROOT = Path("data/processed/sq224")
RUNS = Path("outputs/runs")
MODELS = ["smallcnn_noweight", "resnet18_frozen", "resnet18_ft", "resnet18_ft_clahe",
          "densenet121_ft", "efficientnet_b0_ft"]
MARGIN, GLYPH_LEVEL, SEED = 3, 0.3, 42


def load(relpath: str) -> np.ndarray:
    return cv2.imread(str(ROOT / Path(relpath).with_suffix(".png")), cv2.IMREAD_GRAYSCALE)


def box_slice(r) -> tuple:
    y0, y1 = max(0, int(r.box_y0) - MARGIN), min(224, int(np.ceil(r.box_y1)) + MARGIN)
    x0, x1 = max(0, int(r.box_x0) - MARGIN), min(224, int(np.ceil(r.box_x1)) + MARGIN)
    return slice(y0, y1), slice(x0, x1)


def glyph_mask(patch: np.ndarray) -> np.ndarray:
    """Bright text pixels in a box: above background + 30 % of the (max - background) range."""
    bg = np.median(patch)
    return patch > bg + GLYPH_LEVEL * (patch.max() - bg)


def erase_bg(img: np.ndarray, r) -> np.ndarray:
    """Remove only the glyph pixels and fill them from their surroundings (Telea inpainting)."""
    sl = box_slice(r)
    mask = np.zeros(img.shape, np.uint8)
    mask[sl] = cv2.dilate(glyph_mask(img[sl]).astype(np.uint8), np.ones((3, 3), np.uint8))  # + anti-aliased edges
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def paste(img: np.ndarray, donor_img: np.ndarray, donor_row) -> np.ndarray:
    out = img.copy()
    sl = box_slice(donor_row)
    src = donor_img[sl]
    m = cv2.dilate(glyph_mask(src).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    out[sl] = np.where(m, np.maximum(out[sl], src), out[sl])  # burned-in text adds brightness; never darkens
    return out


def darken(img: np.ndarray, target_mean: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * (target_mean / (img.mean() / 255.0)), 0, 255).astype(np.uint8)


def build() -> pd.DataFrame:
    """One row per (image, arm) with the original and the edited 224x224 uint8 array."""
    kv = pd.read_csv("outputs/07_gradcam/kv_labels.csv").query("split == 'test'")
    kv_norm = kv[(kv.label == "NORMAL") & kv.kv_label].reset_index(drop=True)
    clean = kv[(kv.label == "NORMAL") & ~kv.kv_label].reset_index(drop=True)
    donors = kv[kv.kv_label].reset_index(drop=True)
    target_mean = float(np.median([load(p).mean() / 255.0 for p in clean.relpath]))
    rng = np.random.default_rng(SEED)
    rows = []
    for r in kv_norm.itertuples():
        img = load(r.relpath)
        rows.append({"relpath": r.relpath, "arm": "erase_bg", "orig": img, "edit": erase_bg(img, r)})
        rows.append({"relpath": r.relpath, "arm": "darken", "orig": img, "edit": darken(img, target_mean)})
    for r, d in zip(clean.itertuples(), rng.integers(0, len(donors), len(clean))):
        dr = donors.iloc[d]
        img = load(r.relpath)
        rows.append({"relpath": r.relpath, "arm": "paste", "orig": img, "edit": paste(img, load(dr.relpath), dr),
                     "donor": dr.relpath})
    print(f"target mean intensity for darken: {target_mean:.3f}; "
          f"rows: {pd.Series([x['arm'] for x in rows]).value_counts().to_dict()}")
    return pd.DataFrame(rows)


def panel(df: pd.DataFrame, scores: pd.DataFrame | None = None) -> None:
    """Two examples per arm, original vs edited, zoomed to the bottom-right corner where the text sits."""
    ex = pd.concat([g.head(2) for _, g in df.groupby("arm", sort=False)])
    fig, axes = plt.subplots(2, len(ex) * 2, figsize=(2.2 * len(ex) * 2, 4.6), squeeze=False)
    for c, r in enumerate(ex.itertuples()):
        for row, (im, name) in enumerate(((r.orig, "original"), (r.edit, r.arm))):
            axes[row, 2 * c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[row, 2 * c + 1].imshow(im[180:, 160:], cmap="gray", vmin=0, vmax=255)
            title = name
            if scores is not None:
                s = scores[(scores.relpath == r.relpath) & (scores.arm == r.arm) & (scores.run == "resnet18_ft")]
                title += f"\nResNet-18 P={s['score_orig' if row == 0 else 'score_edit'].iloc[0]:.2f}"
            axes[row, 2 * c].set_title(title, fontsize=7)
            axes[row, 2 * c + 1].set_title("corner zoom", fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "interventions_examples.png", dpi=110)


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    import torch
    from dataset import normalize
    from models import build_model

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    thr = pd.read_csv("outputs/06_evaluation/thresholds.csv").query("rule == 'val_sens95'").set_index("model")["threshold"]

    @torch.no_grad()
    def score(model, arrays, clahe):
        out = []
        for i in range(0, len(arrays), 64):
            batch = [apply_clahe(a) if clahe else a for a in arrays[i:i + 64]]
            x = torch.stack([normalize(torch.from_numpy(a)[None]) for a in batch]).to(device)
            out.append(torch.sigmoid(model(x)).flatten().cpu().numpy())
        return np.concatenate(out)

    res = []
    for run in MODELS:
        cfg = json.loads((RUNS / run / "config.json").read_text())
        model = build_model(cfg["model"], pretrained=False) if cfg["model"] != "small_cnn" else build_model("small_cnn")
        model.load_state_dict(torch.load(Path("checkpoints") / f"{run}.pt", map_location="cpu"))
        model = model.to(device).eval()
        clahe = cfg["variant"] == "sq224_clahe"
        res.append(df[["relpath", "arm"]].assign(run=run, threshold=thr[run],
                                                  score_orig=score(model, list(df.orig), clahe),
                                                  score_edit=score(model, list(df.edit), clahe)))
        print(f"done: {run}")
    return pd.concat(res, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only build the edited images and the panel")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    if args.dry_run:
        panel(df)
        print("dry run: wrote interventions_examples.png")
        return
    sc = score_all(df)
    sc["called_orig"] = sc.score_orig >= sc.threshold
    sc["called_edit"] = sc.score_edit >= sc.threshold
    sc.to_csv(OUT / "interventions.csv", index=False)
    summary = (sc.groupby(["arm", "run"])
               .agg(n=("relpath", "size"), median_orig=("score_orig", "median"), median_edit=("score_edit", "median"),
                    called_pneumonia_orig=("called_orig", "sum"), called_pneumonia_edit=("called_edit", "sum"))
               .round(3))
    text = ("Official test set, NORMAL images only, D13 threshold. 'called_pneumonia' = predicted PNEUMONIA "
            "(all are false alarms).\n\n" + summary.to_string())
    (OUT / "interventions_summary.txt").write_text(text)
    print(text)
    panel(df, sc)


if __name__ == "__main__":
    main()
