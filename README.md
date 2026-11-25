# Pneumonia Classification from Pediatric Chest X-Rays

Deep learning study for NORMAL vs PNEUMONIA classification on the Kaggle
*Chest X-Ray Images (Pneumonia)* dataset (Kermany et al., 2018), with an emphasis on
**leakage-free evaluation**, **shortcut detection** and **explainability (Grad-CAM)**.

> Educational project (MTH416). Not a clinically validated diagnostic system.

## Status

| Stage | Topic | Status |
|---|---|---|
| 1 | Dataset exploration | ✅ |
| 2 | Leakage-aware patient-level split | ✅ |
| 3 | Preprocessing & augmentation | ✅ |
| 4 | Baselines (majority class, metadata-only, small CNN) | ✅ |
| 5 | Transfer learning | ⏳ |
| 6 | Evaluation (sensitivity, specificity, ROC-AUC, threshold) | ⏳ |
| 7 | Grad-CAM explainability | ⏳ |

## Key findings so far
- **Class imbalance:** 74 % pneumonia in train, 62.5 % in test → always-"pneumonia" scores 62.5 % test accuracy.
- **Probable patient leakage in the official split:** 170 patient IDs appear in both train and test.
  We keep the official test set and remove those patients' 457 images from training ([D2](DECISIONS.md)).
- **Shortcuts in the data:** RGB images occur only in the pneumonia class; image width alone reaches
  test AUC 0.95; laterality markers sit in consistent positions. See [DECISIONS.md](DECISIONS.md).
- **Preprocessing chosen by measurement, not by habit:** squash to 224x224 with anti-aliased `INTER_AREA`
  (pad/crop left stronger non-anatomical cues or cut the lungs), CLAHE off by default because it amplified
  those cues (kept as an ablation), augmentation limited to realistic, label-preserving transforms
  (no flips, no blur). Decisions D6–D9.

## Results so far (test set = official 624 images; threshold chosen on val)

| Model | Test ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Always "pneumonia" | 0.500 | 0.625 | 1.000 | 0.000 |
| Logistic regression on original image size only (no pixels) | 0.911 | 0.833 | 0.879 | 0.756 |
| SmallCNN from scratch (0.39 M params) | 0.945 | 0.843 | 0.992 | 0.594 |

The CNN reaches val AUC 0.998 but test AUC 0.945; test NORMAL images are often called pneumonia, and CNN
scores correlate with original image height — see Stage 4 findings in [DECISIONS.md](DECISIONS.md).

## Data splits (stage 2)

| Split | NORMAL | PNEUMONIA | Patients |
|---|---|---|---|
| train | 1,078 | 2,723 | 2,123 |
| val | 270 | 682 | 568 |
| test (official) | 234 | 390 | 427 |

## Reproduce
0. Create an environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
1. Download the dataset from Kaggle (`paultimothymooney/chest-xray-pneumonia`) and unzip so that
   `data/chest_xray/train`, `data/chest_xray/val` and `data/chest_xray/test` exist.
2. Run from the project root:
   ```bash
   python src/01_explore_dataset.py   # -> outputs/01_exploration/
   python src/02_make_splits.py       # -> outputs/02_splits/splits.csv
   python src/03_preprocess.py        # -> data/processed/sq224/        (224x224 grayscale PNG cache)
   python src/03_preprocess.py --clahe  # -> data/processed/sq224_clahe/ (ablation variant)
   python src/03c_check_dataset.py    # sanity checks for the PyTorch Dataset
   python src/04a_pixel_free_baselines.py   # majority + image-size-only baselines
   python src/train.py --run-name smallcnn_noweight   # SmallCNN (~25 s/epoch on an M5 MacBook Air, MPS)
   python src/04c_near_duplicates.py        # perceptual-hash leakage check
   ```

## Repository layout
```
src/            numbered scripts per stage (03a/03b = studies behind decisions D6/D7)
                + shared modules: preprocessing.py, dataset.py, shortcut_checks.py
outputs/        generated summaries, figures and split files
DECISIONS.md    every methodological decision with rationale and rejected alternatives
```

## Limitations
Single hospital (Guangzhou), children aged 1–5. Results are internal performance estimates
and do not establish generalisation to other hospitals or adult patients.
