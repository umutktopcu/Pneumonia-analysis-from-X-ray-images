"""Stage 7b - Intervention: does erasing the "KV.." label change the models' predictions?

Correlation (07a) is not causation. Here we change ONLY the label region and re-score the same test
images with the already-trained models (no retraining):
  intervention : KV-labelled images -> paint the detected text box (+3 px margin) black (the text sits
                 on a black background, so the result looks like plain background)
  control      : images without a KV label -> paint their best-matching (text-free) box black the same way
If predictions on KV images change much more than on controls, the label itself drives the decision.
Decision threshold = D13 operating point (val sensitivity >= 95 %) from outputs/06_evaluation/thresholds.csv.

Usage (from project root, venv active): python src/07b_kv_intervention.py
Output: outputs/07_gradcam/{kv_intervention.csv, kv_intervention_summary.txt, kv_intervention_examples.png}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image

from dataset import normalize
from models import build_model
from runs import RUNS

MODELS = ["smallcnn_noweight", "resnet18_frozen", "resnet18_ft", "resnet18_ft_clahe",
          "densenet121_ft", "efficientnet_b0_ft"]
OUT = Path("outputs/07_gradcam")
MARGIN = 3


def masked(img: torch.Tensor, box) -> torch.Tensor:
    x0, y0, x1, y1 = box
    out = img.clone()
    out[:, max(0, int(y0) - MARGIN):int(np.ceil(y1)) + MARGIN, max(0, int(x0) - MARGIN):int(np.ceil(x1)) + MARGIN] = 0
    return out


@torch.no_grad()
def score(model, imgs, device) -> np.ndarray:
    x = torch.stack([normalize(i) for i in imgs]).to(device)
    return torch.sigmoid(model(x)).cpu().numpy()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kv = pd.read_csv(OUT / "kv_labels.csv").query("split == 'test'").reset_index(drop=True)
    kv = kv.dropna(subset=["box_x0"])
    thr = pd.read_csv("outputs/06_evaluation/thresholds.csv").query("rule == 'val_sens95'").set_index("model")["threshold"]

    rows = []
    for run in MODELS:
        cfg = json.loads((RUNS / run / "config.json").read_text())
        model = build_model(cfg["model"], pretrained=False) if cfg["model"] != "small_cnn" else build_model("small_cnn")
        model.load_state_dict(torch.load(Path("checkpoints") / f"{run}.pt", map_location="cpu"))
        model = model.to(device).eval()
        root = Path("data/processed") / cfg["variant"]
        orig, edit = [], []
        for r in kv.itertuples():
            img = read_image(str(root / Path(r.relpath).with_suffix(".png")), mode=ImageReadMode.GRAY)
            orig.append(img)
            edit.append(masked(img, (r.box_x0, r.box_y0, r.box_x1, r.box_y1)))
        s0 = np.concatenate([score(model, orig[i:i + 64], device) for i in range(0, len(orig), 64)])
        s1 = np.concatenate([score(model, edit[i:i + 64], device) for i in range(0, len(edit), 64)])
        rows.append(kv.assign(run=run, score_orig=s0, score_masked=s1, threshold=thr[run]))
        print(f"done: {run}")

    res = pd.concat(rows)
    res["pred_orig"] = res["score_orig"] >= res["threshold"]
    res["pred_masked"] = res["score_masked"] >= res["threshold"]
    res["group"] = res["label"] + np.where(res["kv_label"], " + KV", " (no KV, control)")
    res.to_csv(OUT / "kv_intervention.csv", index=False)

    summary = (res.groupby(["run", "group"])
               .agg(n=("relpath", "size"),
                    median_score_orig=("score_orig", "median"), median_score_masked=("score_masked", "median"),
                    called_pneumonia_orig=("pred_orig", "sum"), called_pneumonia_masked=("pred_masked", "sum"))
               .round(3))
    text = ("Test images, D13 threshold. 'called_pneumonia' = number of images predicted PNEUMONIA.\n\n"
            + summary.to_string())
    (OUT / "kv_intervention_summary.txt").write_text(text)
    print(text)

    # Example panel: KV NORMAL images before/after masking, ResNet-18 FT scores
    ex = res[(res["run"] == "resnet18_ft") & (res["group"] == "NORMAL + KV")].head(6)
    fig, axes = plt.subplots(2, len(ex), figsize=(2.6 * len(ex), 5.6), squeeze=False)
    for c, r in enumerate(ex.itertuples()):
        img = read_image(str(Path("data/processed/sq224") / Path(r.relpath).with_suffix(".png")), mode=ImageReadMode.GRAY)
        box = (r.box_x0, r.box_y0, r.box_x1, r.box_y1)
        for row, (im, sc) in enumerate(((img, r.score_orig), (masked(img, box), r.score_masked))):
            axes[row, c].imshow(im[0], cmap="gray")
            axes[row, c].set_title(f"{'original' if row == 0 else 'KV masked'}\nP(pneu)={sc:.2f}", fontsize=8)
            axes[row, c].axis("off")
    plt.suptitle("ResNet-18 FT on NORMAL test images with a KV label", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "kv_intervention_examples.png", dpi=100)


if __name__ == "__main__":
    main()
