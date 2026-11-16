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
| 3 | Preprocessing & augmentation | ⏳ |
| 4 | Baselines (majority class, metadata-only, small CNN) | ⏳ |
| 5 | Transfer learning | ⏳ |
| 6 | Evaluation (sensitivity, specificity, ROC-AUC, threshold) | ⏳ |
| 7 | Grad-CAM explainability | ⏳ |

## Key findings so far
- **Class imbalance:** 74 % pneumonia in train, 62.5 % in test → always-"pneumonia" scores 62.5 % test accuracy.
- **Probable patient leakage in the official split:** 170 patient IDs appear in both train and test.
  We keep the official test set and remove those patients' 457 images from training ([D2](DECISIONS.md)).
- **Shortcuts in the data:** RGB images occur only in the pneumonia class; image width alone reaches
  test AUC 0.95; laterality markers sit in consistent positions. See [DECISIONS.md](DECISIONS.md).

## Data splits (stage 2)

| Split | NORMAL | PNEUMONIA | Patients |
|---|---|---|---|
| train | 1,078 | 2,723 | 2,123 |
| val | 270 | 682 | 568 |
| test (official) | 234 | 390 | 427 |

## Reproduce
1. Download the dataset from Kaggle (`paultimothymooney/chest-xray-pneumonia`) and unzip so that
   `data/chest_xray/train`, `data/chest_xray/val` and `data/chest_xray/test` exist.
2. Run from the project root:
   ```bash
   pip install -r requirements.txt
   python src/01_explore_dataset.py   # -> outputs/01_exploration/
   python src/02_make_splits.py       # -> outputs/02_splits/splits.csv
   ```

## Repository layout
```
src/            numbered scripts, one per stage
outputs/        generated summaries, figures and split files
DECISIONS.md    every methodological decision with rationale and rejected alternatives
```

## Limitations
Single hospital (Guangzhou), children aged 1–5. Results are internal performance estimates
and do not establish generalisation to other hospitals or adult patients.
