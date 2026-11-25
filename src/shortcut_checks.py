"""Cheap, non-anatomical image statistics used to check how much class information
survives preprocessing (a CNN could exploit any of them without looking at the lungs)."""
import cv2
import numpy as np
import pandas as pd

FEATURES = ["black_frac", "side_band_mean", "mean_intensity", "hf_energy"]


def features(img: np.ndarray) -> dict:
    f = img.astype(np.float32) / 255.0
    band = img.shape[1] // 14  # 16-px strips at the left/right edges for 224-px images
    return {
        "black_frac": float((img < 10).mean()),
        "side_band_mean": float(np.concatenate([f[:, :band], f[:, -band:]], axis=1).mean()),
        "mean_intensity": float(f.mean()),
        "hf_energy": float(np.abs(cv2.Laplacian(f, cv2.CV_32F)).mean()),  # fine texture / sharpness
    }


def separability(score: np.ndarray, y: np.ndarray) -> float:
    """max(AUC, 1-AUC): 0.5 = no class information in this feature, 1.0 = perfect separation."""
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))
