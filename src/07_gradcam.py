"""Stage 7 - Grad-CAM: which image regions drive the PNEUMONIA score?

Grad-CAM (Selvaraju et al., 2017) for a single-logit classifier:
  A_k   = feature maps of the last convolutional stage (k = channel, e.g. 512 x 7 x 7 in ResNet-18)
  a_k   = mean over space of d(logit)/d(A_k)          -> importance of channel k
  CAM   = ReLU( sum_k a_k * A_k )                      -> keep only evidence FOR pneumonia
  upsample to 224 x 224, scale to [0, 1] per image (so maps show WHERE, not HOW STRONG)

Analysis on the official test set (diagnosis only, nothing is selected with it):
  - consistency check: scores recomputed locally must match predictions.csv from Kaggle
  - border fraction: share of the map in the outer 12 % frame (uniform map -> ~42 %)
  - groups: NORMAL misclassified by all 6 models / by none / PNEUMONIA
  - sanity check (Adebayo et al., 2018): maps of a randomly initialized network must differ

Usage (from project root, venv active): python src/07_gradcam.py
Output: outputs/07_gradcam/{cam_stats.csv, summary.txt, gradcam_<run>.png}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

from dataset import normalize
from models import build_model
from runs import RUNS, load_predictions

RUNS_TO_EXPLAIN = {"resnet18_ft": "resnet18", "efficientnet_b0_ft": "efficientnet_b0"}
TARGET_LAYER = {"resnet18": lambda m: m.backbone.layer4, "efficientnet_b0": lambda m: m.backbone.features}
ROOT = Path("data/processed/sq224")
OUT = Path("outputs/07_gradcam")
BORDER = 0.12
SEED = 42


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._keep_activation)

    def _keep_activation(self, module, inputs, output):
        self.acts = output
        output.register_hook(self._keep_gradient)  # called during backward with d(logit)/d(output)

    def _keep_gradient(self, grad):
        self.grads = grad

    def __call__(self, x: torch.Tensor):
        """x: 1x3xHxW normalized image. Returns (cam HxW in [0, 1], logit)."""
        self.model.zero_grad()
        logit = self.model(x)
        logit.sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)              # a_k
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))     # ReLU(sum_k a_k A_k)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), float(logit.item())


def border_fraction(cam: np.ndarray, frac: float = BORDER) -> float:
    h, w = cam.shape
    bh, bw = int(frac * h), int(frac * w)
    inner = cam[bh:h - bh, bw:w - bw].sum()
    return float(1 - inner / (cam.sum() + 1e-8))


def load_image(relpath: str) -> torch.Tensor:
    return read_image(str(ROOT / Path(relpath).with_suffix(".png")), mode=ImageReadMode.GRAY)


def test_groups() -> pd.DataFrame:
    test = pd.read_csv("outputs/02_splits/splits.csv").query("split == 'test'")[["relpath", "label"]]
    overlap = pd.read_csv("outputs/06_evaluation/error_overlap.csv", index_col=0)["n_models_wrong"]
    test["n_models_wrong"] = test["relpath"].map(overlap)
    test["group"] = np.select(
        [test["label"] == "PNEUMONIA", test["n_models_wrong"] == 6, test["n_models_wrong"] == 0],
        ["PNEUMONIA", "NORMAL_wrong_by_all_6", "NORMAL_right_by_all_6"], default="NORMAL_mixed")
    return test.reset_index(drop=True)


def load_model(run: str, arch: str, device, random_weights: bool = False):
    model = build_model(arch, pretrained=False)
    if not random_weights:
        model.load_state_dict(torch.load(Path("checkpoints") / f"{run}.pt", map_location="cpu"))
    return model.to(device).eval()  # eval: BatchNorm uses stored statistics, Dropout off


def plot_examples(test, cams, run, out, n=6):
    groups = ["NORMAL_wrong_by_all_6", "NORMAL_right_by_all_6", "PNEUMONIA"]
    fig, axes = plt.subplots(len(groups), n, figsize=(2.6 * n, 2.9 * len(groups)))
    rng = np.random.default_rng(SEED)
    for r, g in enumerate(groups):
        idx = rng.choice(test.index[test["group"] == g], n, replace=False)
        for c, i in enumerate(idx):
            img = load_image(test.loc[i, "relpath"])[0].numpy()
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].imshow(cams[i], cmap="jet", alpha=0.4, vmin=0, vmax=1)
            axes[r, c].set_title(f"{g.replace('_by_all_6', '')}\nP(pneu)={test.loc[i, 'score']:.2f} "
                                 f"border={test.loc[i, 'border_frac']:.2f}", fontsize=7)
            axes[r, c].axis("off")
    plt.suptitle(f"Grad-CAM ({run}): evidence for PNEUMONIA", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    base = test_groups()
    lines, all_stats = [f"uniform map border fraction (outer {BORDER:.0%} frame): "
                        f"{1 - (1 - 2 * BORDER) ** 2:.3f}"], []

    for run, arch in RUNS_TO_EXPLAIN.items():
        model = load_model(run, arch, device)
        cam_fn = GradCAM(model, TARGET_LAYER[arch](model))
        test, cams = base.copy(), {}
        scores, borders = [], []
        for i, relpath in enumerate(test["relpath"]):
            x = normalize(load_image(relpath)).unsqueeze(0).to(device)
            cams[i], logit = cam_fn(x)
            scores.append(1 / (1 + np.exp(-logit)))
            borders.append(border_fraction(cams[i]))
        test["score"], test["border_frac"], test["run"] = scores, borders, run

        saved = load_predictions(RUNS / run).query("split == 'test'").set_index("relpath")["score"]
        max_diff = float(np.abs(test.set_index("relpath")["score"] - saved.loc[test["relpath"]].to_numpy()).max())
        lines.append(f"\n[{run}] max |local score - Kaggle score| = {max_diff:.2e} "
                     f"({'OK, same model' if max_diff < 1e-2 else 'MISMATCH - wrong checkpoint?'})")
        summary = test.groupby("group")[["score", "border_frac"]].agg(["count", "median"]).round(3)
        lines.append(summary.to_string())

        # Sanity check: same architecture with random weights should give unrelated maps.
        rand_model = load_model(run, arch, device, random_weights=True)
        rand_cam = GradCAM(rand_model, TARGET_LAYER[arch](rand_model))
        sample = np.random.default_rng(SEED).choice(len(test), 30, replace=False)
        corr = [np.corrcoef(cams[i].ravel(), rand_cam(normalize(load_image(test.loc[i, "relpath"]))
                                                       .unsqueeze(0).to(device))[0].ravel())[0, 1]
                for i in sample]
        lines.append(f"sanity check: mean correlation trained vs random-weight maps = {np.nanmean(corr):.3f} "
                     f"(near 0 expected)")

        plot_examples(test, cams, run, OUT / f"gradcam_{run}.png")
        all_stats.append(test)

    pd.concat(all_stats).to_csv(OUT / "cam_stats.csv", index=False)
    (OUT / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
