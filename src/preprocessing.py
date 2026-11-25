"""Deterministic preprocessing shared by the caching script and the training code.

Every image (train, val and test) goes through exactly the same steps (decisions D1, D6, D7):
    1. read as single-channel grayscale        (D1: removes the RGB-only-in-PNEUMONIA shortcut)
    2. squash to IMG_SIZE x IMG_SIZE, INTER_AREA (D6: keeps the whole lung field, anti-aliased)
    3. optional CLAHE, applied AFTER resizing   (D7: identical tile size for every image)
Random augmentation is NOT done here; it belongs to the training loop (train split only).
"""
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 224


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_squash(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(img)


def preprocess(path: Path, size: int = IMG_SIZE, clahe: bool = False) -> np.ndarray:
    img = resize_squash(load_gray(path), size)
    return apply_clahe(img) if clahe else img
