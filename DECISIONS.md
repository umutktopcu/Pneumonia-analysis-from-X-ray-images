# Decision Log

Each entry: what we decided, why, what we rejected, and what it costs us.
Evidence for all numbers: `outputs/01_exploration/summary.txt`, `outputs/02_splits/summary.txt`.

---

## D1 — Convert every image to single-channel grayscale (Stage 1, 2026-11-15)
**Why:** 283 images are stored as RGB and *all* of them are in train/PNEUMONIA. The ones we
checked are gray images saved with 3 identical channels. Without conversion a model could
learn "3 channels ⇒ pneumonia" (shortcut learning).
**Consequence:** For ImageNet-pretrained models we convert to grayscale first and only then
replicate to 3 channels, so every image goes through the identical path.

## D2 — Test set handling: keep official test, drop leaked patients from training (Stage 2, 2026-11-16)
**Problem:** 170 patient IDs (from `personX_` filenames) occur in both train and test; they own
320/390 test pneumonia images and 457 train images. Patient IDs share one numbering range across
splits and per-subtype image numbers never collide → strong hint that same ID = same child.
But the dataset authors (Kermany et al., Mendeley Data) state train/test contain independent
patients, and our thumbnail-similarity check was inconclusive. Status: **probable, unproven**.

**Options considered**
- A. Use the splits as given → comparable to literature, but test scores possibly inflated.
- B. Pool everything and re-split by patient (done by e.g. arXiv:2511.00456) → cleanest, but
  loses comparability and mixes the 3rd-expert-checked test labels with train labels.
- **C. Chosen:** keep the official test set untouched; remove from the training pool every
  image of a patient whose ID also appears in test.

**Why C:** robust to both possibilities — if leakage is real, C removes it; if it is not, we only
lost ~12% of the majority (pneumonia) class. Test labels were additionally checked by a third
expert, so the official test is the highest-quality part of the data. Results stay comparable
with the many papers that report on the 624-image test set.

**Cost / limitations:** 457 fewer training images. Test results remain *internal* (same hospital,
children aged 1–5) and say nothing about generalisation to other hospitals or adults.

## D3 — Patient-level, class-stratified train/val split, val = 20 %, seed 42 (Stage 2)
**Why patient level:** pneumonia patients have up to 31 images (mean 2.55). An image-level random
split puts the same child in train and val, inflating validation scores. Several papers on this
dataset make exactly this mistake (stratified *image-level* 80/20 or 80/10/10 splits).
**Why 20 %:** val drives early stopping, threshold selection and model comparison; with ~1,350
normal images, 20 % gives ~270 normals, enough for a reasonably stable specificity estimate.
**Why our own splitter instead of sklearn `StratifiedGroupKFold`:** every patient has a single
label, so stratifying by class = splitting patients class by class. Simple, transparent, and
no extra dependency at this stage.
**Official val (16 images):** merged into the pool — too small to be useful, distinct patients.

## D4 — Exact duplicates (Stage 2)
Pool copies of an already-kept file are dropped (22 images); test copies take priority over pool
copies. The test set's own internal duplicates (6 groups) are left as-is to keep the official test
intact. MD5 only catches byte-identical files, not resized/re-encoded near-duplicates.

## D5 — Tooling (2026-11-15)
PyTorch. EDA and data preparation run locally (MacBook Air M5, 24 GB). GPU training on Kaggle
Notebooks: the dataset is hosted there and long runs would throttle the fanless laptop.

---

## Known open issues (to address in later stages)
- **Image-size shortcut:** NORMAL images are much larger (median 1654x1323 vs 1160x776). Width
  alone gives test AUC 0.95; an aspect-ratio rule gets 68.6 % test accuracy without pixels.
  → Stage 3: choose resizing deliberately; Stage 4: report a metadata-only baseline;
  Stage 7: check Grad-CAM for non-lung focus.
- **Laterality markers ("R")** in consistent positions, visible in the NORMAL mean image.
