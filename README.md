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
| 5 | Transfer learning | ✅ |
| 6 | Evaluation (sensitivity, specificity, ROC-AUC, threshold) | ✅ |
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

## Results (official 624-image test set; decision threshold chosen on val)

| Model | Val AUC | Test ROC-AUC [95% bootstrap CI] | Test AUC on shortcut-conflicting images | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Always "pneumonia" | – | 0.500 | – | 1.000 | 0.000 |
| Logistic regression on original image size only (no pixels) | 0.901 | 0.911 | – | 0.879 | 0.756 |
| SmallCNN from scratch (0.39 M params) | 0.998 | 0.945 [0.923, 0.964] | 0.628 | 0.992 | 0.594 |
| ResNet-18, frozen backbone | 0.990 | 0.952 [0.933, 0.968] | 0.724 | 0.967 | 0.786 |
| ResNet-18, fine-tuned | 0.999 | 0.971 [0.955, 0.984] | 0.847 | 0.997 | 0.688 |
| ResNet-18, fine-tuned, CLAHE input | 0.999 | 0.981 [0.971, 0.989] | 0.828 | 0.997 | 0.697 |
| DenseNet-121, fine-tuned | 0.999 | 0.975 [0.961, 0.986] | 0.757 | 0.997 | 0.739 |
| EfficientNet-B0, fine-tuned (**selected by the pre-stated val rule, D12**) | 1.000 | 0.936 [0.912, 0.958] | 0.759 | 0.995 | 0.701 |

**Key finding:** validation data from the same distribution as training cannot rank models under the
distribution shift present in the official test set — the val-selected model is the weakest pretrained
model on test. Models are therefore reported as a benchmark, not ranked by test score (that would be
selection on the test set). "Shortcut-conflicting" = images that the size-only model misclassifies.
Details: Stage 4/5 findings and D10–D12 in [DECISIONS.md](DECISIONS.md).

### Operating point (D13: threshold = highest val threshold with val sensitivity ≥ 95 %)

| Model | Test sensitivity | Test specificity | Missed pneumonia (FN / 390) | False alarms (FP / 234) |
|---|---|---|---|---|
| SmallCNN | 0.985 | 0.714 | 6 | 67 |
| ResNet-18, frozen | 0.972 | 0.765 | 11 | 55 |
| ResNet-18, fine-tuned | 0.977 | 0.829 | 9 | 40 |
| ResNet-18, fine-tuned, CLAHE | 0.974 | 0.855 | 10 | 34 |
| DenseNet-121, fine-tuned | 0.974 | 0.829 | 10 | 40 |
| EfficientNet-B0, fine-tuned | 0.962 | 0.825 | 15 | 41 |

Paired bootstrap on test: ResNet-18 FT, ResNet-18 FT + CLAHE and DenseNet-121 FT are consistently better
(AUC) than SmallCNN, frozen ResNet-18 and EfficientNet-B0; they are not reliably different from each other.
27 of 234 test NORMAL images are misclassified by all six models — mostly small original images.

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
   # Stage 5 runs were trained on Kaggle (T4 GPU) with src/train.py, e.g.
   #   python src/train.py --run-name resnet18_ft --model resnet18 --lr 1e-4 --scheduler cosine --epochs 15 --num-workers 4
   python src/05_compare_runs.py            # comparison table with bootstrap CIs
   python src/06_evaluation.py              # paired bootstrap, thresholds, confusion matrices, error overlap
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
