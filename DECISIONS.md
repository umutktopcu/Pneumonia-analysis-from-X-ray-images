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

## D6 — Resize by squashing to 224x224 with INTER_AREA (Stage 3, 2026-11-16)
**Options compared** (`src/03a_resize_study.py`, `outputs/03_preprocessing/`): squash, pad to square
(black borders), center square crop. Measured how well non-anatomical statistics separate the classes
after each strategy (max(AUC,1-AUC), train sample n=600; 0.5 = no information):

| strategy | black_frac | side_band_mean | mean_intensity |
|---|---|---|---|
| squash | 0.71 | 0.62 | 0.55 |
| pad | 0.78 | 0.55 | 0.74 |
| center_crop | 0.79 | 0.65 | 0.65 |

**Why squash:** keeps the whole lung field (center crop cuts the lateral lungs of wide pneumonia images;
peripheral pneumonia would be lost irrecoverably), shows the weakest non-anatomical cues overall, and is
the simplest option. Pad produces black bars whose thickness directly encodes the aspect ratio (pneumonia
images are wider) and wastes ~40 % of the input on those images.
**Why INTER_AREA:** downsampling is ~7x; INTER_AREA averages all source pixels under each output pixel
(acts as an anti-aliasing low-pass filter). INTER_NEAREST/INTER_LINEAR create moiré/false texture
(`aliasing_demo.png`), and because NORMAL images are larger (downsampled more), aliasing would be
class-dependent: texture separability rose 0.876 → 0.898 with INTER_LINEAR.
**Why 224:** native input size of the ImageNet-pretrained backbones planned for Stage 5.
**Known risk:** squashing distorts wider (mostly pneumonia) images more; the model could learn the
distortion. Check with Grad-CAM (Stage 7). Fallback: content-aware body-box crop (Otsu + largest contour).

## D7 — CLAHE off by default; kept as an ablation (Stage 3, 2026-11-16)
**Applied after resizing** when used, so the 8x8 tile grid covers the same relative area in every image
(before resizing, tile size would depend on original resolution, which differs by class).
**Evidence** (`src/03b_clahe_study.py`, full train split n=3801): CLAHE *increases* the separability of
non-anatomical cues (black_frac 0.70 → 0.74, side_band_mean 0.62 → 0.66, mean_intensity 0.52 → 0.58).
Visually it also boosts laterality markers and bowel gas, and raises local contrast inside hazy lungs —
plausibly weakening the very opacity signal that marks pneumonia (hypothesis, not tested).
**Decision:** baseline pipeline without CLAHE; train one model on `sq224_clahe` in Stage 5 as an ablation
and keep CLAHE only if it improves validation performance.

## D8 — Training augmentation: small, realistic, label-preserving only (Stage 3, 2026-11-16)
**Rule used:** an augmentation is kept only if (1) the label is unchanged and (2) the result could occur
in real acquisitions.
**Kept** (train split only, `src/dataset.py`): rotation ±10°, translation ±5 %, scale 0.95–1.05
(patient positioning), brightness/contrast ±10 % (exposure).
**Rejected:** horizontal flip (label-invariant, but produces mirrored anatomy — heart on the right,
mirrored "R" markers — never seen at test time; lungs are anatomically asymmetric; course slide Hafta 4
#40), vertical flip / large rotations (unrealistic), blur (NORMAL images are measurably sharper, so blur
can push a NORMAL image towards the PNEUMONIA look — effectively label noise), random erasing (can delete
the lesion). Brightness kept small because opacity = brighter lung regions is part of the disease signal.

## D9 — Normalization: /255, replicate to 3 channels, ImageNet mean/std (Stage 3, 2026-11-16)
Pretrained filters were learned on inputs standardized with ImageNet statistics; matching that input
distribution is what makes the transferred features usable. No per-image min-max (course slide #60):
it rescales each image by its own extremes and erases absolute brightness, which carries the opacity
signal.

## D10 — Evaluation protocol (Stage 4, 2026-11-25)
- Positive class = PNEUMONIA. Always report sensitivity AND specificity (plus precision, F1, accuracy,
  balanced accuracy, ROC-AUC, confusion counts) — `src/metrics.py`, numpy-only, self-tested.
- Primary comparison metric: ROC-AUC (threshold-free). Decision threshold: chosen on VAL only
  (Youden J = sensitivity + specificity − 1 by default); TEST is only reported, never used for choices.
- Every model is compared against two pixel-free baselines (`src/04a_pixel_free_baselines.py`):
  majority class (test acc 0.625, AUC 0.5) and a logistic regression on original image size
  (test AUC 0.911, acc 0.833). A CNN that does not clearly beat the size model has not demonstrably
  learned anything beyond acquisition differences.
- F1 with PNEUMONIA as positive is inflated by the majority class (always-PNEUMONIA scores F1 0.769
  on test), so F1 alone is never used to compare models.

## D11 — Class imbalance: no loss weighting; threshold chosen on val (Stage 4, 2026-11-25)
Compared on SmallCNN (`src/train.py`, seed 42), identical except for the loss:

| run | val-chosen threshold | test AUC | test sensitivity | test specificity |
|---|---|---|---|---|
| (a) plain BCE | 0.834 | 0.945 | 0.992 | 0.594 |
| (b) BCE, pos_weight = 1078/2723 ≈ 0.40 | 0.957 | 0.945 | 0.995 | 0.603 |

Weighting shifted the scores (and hence the threshold) but not their ranking (identical AUC), as expected.
The specificity difference is 2 of 234 NORMAL images, well inside noise (SE ≈ √(0.6·0.4/234) ≈ 0.03, single
seed). Decision: (a), the simpler option. The low test specificity is a distribution-shift problem, which
loss weighting cannot fix.

## Stage 4 findings (2026-11-25)
- **SmallCNN (0.39 M params, from scratch): val AUC 0.998, test AUC 0.945** vs metadata-only baseline test AUC
  0.911. It beats the size shortcut, but by a modest margin; the near-perfect val score does not transfer.
- **Test NORMAL images look "pneumonia-like" to the model:** median score 0.033 on val NORMAL vs 0.719 on test
  NORMAL; at the val-chosen threshold 95/234 test NORMAL images are false positives (specificity 0.594).
- **Evidence consistent with size-shortcut use:** within NORMAL images the CNN score correlates with the
  original image height (Spearman −0.55 test, −0.47 val); false-positive NORMAL images are shorter (median
  height 1016 vs 1489 px). Correlation only — shorter images may also differ clinically or by device.
  The test shift is not explained by height alone (test vs val NORMAL median height 1318 vs 1349 px).
- **No near-duplicate leakage** (`src/04c_near_duplicates.py`, 256-bit dHash): nearest-train distances are
  similar for val (median 34) and test (median 32), so val is no closer to train than the independent test
  set; the closest pairs (distance 16; random pairs ≥ 19) are visibly different children. An 8x8 hash was
  useless (random X-ray pairs differ by only ~11/64 bits) — perceptual hashes must be calibrated on the data.
- Val loss spikes (e.g. 1.09 at one epoch, back to 0.04 the next) while val AUC stays ≈0.99: instability /
  calibration jumps, not overfitting (the lower envelope of val loss keeps falling; no train-val AUC gap).
  Likely BatchNorm running statistics + constant LR (hypothesis). Model selection uses val AUC, so unaffected.

## D12 — Model-selection rule when validation is saturated (Stage 5, 2026-11-25)
**Problem:** pretrained models reach val AUC ≈ 0.999 within 1–3 epochs; val (same distribution as train)
can no longer separate models or epochs.
**Rule (proposed in the session before the Stage 5 comparison table was computed; the user had already
seen the raw test lines printed by train.py):** rank by (1) val ROC-AUC, then (2) val AUC on the
*shortcut-conflicting* subset (val images the image-size-only model misclassifies: 194 images, only 23
NORMAL — noisy), then (3) val BCE loss; if still indistinguishable, (4) prefer the simpler model.
Test is never used for selection.
**Outcome:** the rule selects **EfficientNet-B0** (best on all three val criteria) — which has the
**lowest** test AUC of the pretrained models (0.936). See Stage 5 findings.

## Stage 5 findings (2026-11-25) — `src/05_compare_runs.py`, `outputs/05_transfer/comparison.csv`
All runs: seed 42, AdamW, cosine LR (except SmallCNN), ≤15 epochs, early stopping on val AUC (patience 5),
Kaggle T4. "conflict" = shortcut-conflicting subset (val 194 / test 104 images).

| run | val AUC | val conflict AUC | val loss | test AUC [95% bootstrap CI] | test conflict AUC | test sens | test spec |
|---|---|---|---|---|---|---|---|
| efficientnet_b0_ft | 0.9996 | 0.995 | 0.023 | 0.936 [0.912, 0.958] | 0.759 | 0.995 | 0.701 |
| densenet121_ft | 0.9994 | 0.993 | 0.073 | 0.975 [0.961, 0.986] | 0.757 | 0.997 | 0.739 |
| resnet18_ft | 0.9993 | 0.990 | 0.031 | 0.971 [0.955, 0.984] | 0.847 | 0.997 | 0.688 |
| resnet18_ft_clahe | 0.9992 | 0.987 | 0.050 | 0.981 [0.971, 0.989] | 0.828 | 0.997 | 0.697 |
| smallcnn_noweight | 0.9980 | 0.966 | 0.070 | 0.945 [0.923, 0.964] | 0.628 | 0.992 | 0.594 |
| resnet18_frozen | 0.9901 | 0.916 | 0.134 | 0.952 [0.933, 0.968] | 0.724 | 0.967 | 0.786 |

- **Main finding: in-distribution validation does not predict performance on the official test set.**
  The val ranking (EfficientNet first) is essentially unrelated to the test ranking (EfficientNet last among
  pretrained models). No val-only criterion we have — AUC, conflict-subset AUC, loss — fixes this, because the
  test set differs from train/val in ways val does not contain (Stage 4: test NORMAL images look
  pneumonia-like to every model).
- Pretrained fine-tuning beats SmallCNN on test AUC and is much better on shortcut-conflicting test images
  (0.76–0.85 vs 0.59–0.63) → less reliance on the image-size cue, not none.
- Test AUC differences among DenseNet-121, ResNet-18 and ResNet-18+CLAHE are within overlapping CIs; picking the
  "best" of them by test AUC would be selection on the test set. They are reported as a benchmark, not ranked.
- The frozen ResNet-18 (feature extraction) has the lowest val AUC but the highest test specificity (0.786).
- Early stopping on a saturated val AUC chooses the "best" epoch among near-ties (DenseNet: epoch 1). This is a
  symptom of the same problem, not its cause.
- Every additional experiment judged on test increases the risk of adapting to the test set (test reuse);
  further experiments must be few and declared in advance.

## D13 — Primary operating point: sensitivity ≥ 95 % on validation (Stage 6, 2026-11-25)
**Decision (user + assistant):** the reported operating point of every model is the highest threshold whose
VAL sensitivity is ≥ 95 % (`metrics.threshold_for_sensitivity`); val-Youden results are reported as a
secondary table. **Why:** pneumonia detection is a screening task — missing a case is costlier than a false
alarm — so the clinically meaningful question is "catch ≥ 95 % of cases, then minimize false alarms".
Youden weights both errors equally. The rule was proposed before its test results were computed; it is a
clinical choice, not a test-optimized one. **Cost:** on test, more missed cases than Youden (e.g. ResNet-18
FT: FN 1 → 9) for far fewer false alarms (FP 73 → 40).

## Stage 6 findings (2026-11-25) — `src/06_evaluation.py`, `outputs/06_evaluation/`
- **Paired bootstrap (same resampled test images for both models, 2000 rounds, class-stratified):**
  ResNet-18 FT, ResNet-18 FT + CLAHE and DenseNet-121 FT are consistently better than SmallCNN, frozen
  ResNet-18 and EfficientNet-B0 (CIs of the AUC difference exclude 0, e.g. DenseNet − EfficientNet
  +0.039 [0.019, 0.061]). DenseNet vs ResNet-18 FT: no consistent difference. ResNet-18 + CLAHE vs ResNet-18:
  +0.010 [0.002, 0.021] on test, but (a) 15 pairwise comparisons were made, so one or two "consistent"
  results are expected by chance; (b) the bootstrap only covers test-sample uncertainty, not training
  randomness (single seed); (c) D7 required CLAHE to improve VAL (val AUC 0.9992 vs 0.9993: no). → CLAHE stays
  off; the test hint is reported as exploratory.
- **Threshold rules (thresholds always from val; the "oracle" uses test labels and is shown only for scale):**
  val Youden → test sens 0.97–1.00, spec 0.59–0.79. Val "sensitivity ≥ 95 %" → spec improves for every
  fine-tuned model (e.g. ResNet-18: FP 73 → 40) at the cost of more misses (FN 1 → 9). Oracle test-Youden reaches
  spec ≈ 0.85–0.92: the gap is the price of the val→test shift. Hypothesis "Youden picks the low end of a
  plateau of tied thresholds" was checked and rejected (max J is unique for every model).
- **Shared errors:** of 234 test NORMAL images, 117 are misclassified by no model and 27 by all six (48 by ≥5).
  Errors are concentrated in the same images and follow image size: median original height 1484 px (0 models
  wrong), 1275 (1–2), 1087 (3–4), 942 px (5–6). The failure is a property of those images (acquisition /
  size cue, possibly label issues), not of one architecture. Candidates for Grad-CAM in Stage 7.

## Stage 7 findings (2026-11-25) — Grad-CAM, KV-label detector, intervention
Files: `src/07_gradcam.py`, `src/07a_kv_labels.py`, `src/07b_kv_intervention.py`, `outputs/07_gradcam/`.
- **Checks:** checkpoints reproduce the Kaggle test scores on the Mac (max |diff| 5e-6 / 1e-5) → the explained
  models are the evaluated ones. Randomization sanity check (Adebayo et al., 2018): trained vs random-weight
  maps correlate 0.03 (ResNet-18) and −0.20 (EfficientNet) → maps depend on the learned weights.
- **Grad-CAM (hypothesis generator):** ResNet-18's evidence for PNEUMONIA in NORMAL images that all six
  models misclassify often sits in the bottom-right corner, where a burned-in exposure label ("KV55/58/59/68/70")
  is printed. EfficientNet's evidence sits more on the lateral chest wall / frame (median border fraction on true
  pneumonia 0.55 vs 0.41 for ResNet-18; uniform map 0.42). Maps of confidently NORMAL images are noise after
  per-image normalization (weak evidence rescaled to [0, 1]) — not interpretable.
- **KV label prevalence** (template-matching detector, validated 53/53 on hand-labelled test images; first two
  detector attempts failed validation and were discarded): train NORMAL 0/1078, train PNEUMONIA 143/2723,
  val NORMAL 0/270, val PNEUMONIA 45/682, **test NORMAL 18/234**, test PNEUMONIA 15/390. In train/val the label
  occurs only in pneumonia images; in test it also appears on normals, and 13 of those 18 are misclassified by
  all six models (mean 5.5/6 models wrong vs 1.5/6 for other test normals). Val cannot reveal this failure.
- **Intervention (erase only the text, no retraining):** predictions barely change (e.g. ResNet-18 FT calls
  16 → 12 of the 18 KV normals pneumonia; EfficientNet 11 → 11; SmallCNN 16 → 16) → the text itself is NOT the
  main cause. KV images form an acquisition subgroup: brighter (median mean intensity 0.57 NORMAL+KV vs 0.48)
  and smaller (median height 1060 vs 1326 px). Hypothesis: the models associate this acquisition appearance with
  pneumonia because the subgroup has no normal examples in training (n = 18 test normals → hypothesis only).
- **Lessons:** Grad-CAM localizes coarsely (32x32-px cells in ResNet-18) and produced a plausible but wrong
  causal story; only the intervention could reject it. Visible black patches (control group) raised false alarms
  of frozen ResNet-18 from 39 to 61 → models are fragile to out-of-distribution artifacts.
- **Decision (D14):** no further experiment. Body-box cropping or ensembling do not target the identified cause
  (missing normal examples from an acquisition subgroup; all models fail on the same images). The fix is data.
  Avoids another look at the test set.

## Comparison with the literature (searched 2026-11-25)
- Kermany et al. (2018), the dataset's original paper, reported ~92.8 % test accuracy on the official test set
  (as cited by arXiv:2601.00837). Our legitimately thresholded models (D13) reach 0.920–0.929 accuracy with test
  AUC 0.971–0.981 → same range.
- Much higher published numbers (96–99.5 %) usually measure something easier, not a better model:
  * re-splitting the data so the test set comes from the training distribution — e.g. a new 80/10/10 split of the
    training folder (arXiv:2601.00837, ~99 %), merged patient-level re-split (arXiv:2511.00456, 96–98 %), or
    cross-validation; our own in-distribution val accuracy is ~0.99 too;
  * image-level random splits (patient leakage, see D3);
  * choosing threshold / epoch / model on the test set: our oracle analysis shows test-tuned thresholds add up to
    ~5 accuracy points (e.g. ResNet-18 FT 0.881 → 0.934);
  * genuinely different methods that may help: lung-region cropping at high resolution (Rajaraman et al., 2018,
    atlas-based lung ROI at 1024 px, 96.2 %), ensembles (Chouhan et al., 2020, 96.4 % on unseen data). Their
    validation protocols were not re-checked here.
- A bias analysis of public pneumonia X-ray datasets (Catalá et al., IEEE Access 2021) also concludes that
  published results on such datasets may be optimistic.

## Known open issues (to address in later stages)
- **Image-size shortcut:** NORMAL images are much larger (median 1654x1323 vs 1160x776). Width
  alone gives test AUC 0.95; an aspect-ratio rule gets 68.6 % test accuracy without pixels.
  → Stage 3: choose resizing deliberately; Stage 4: report a metadata-only baseline;
  Stage 7: check Grad-CAM for non-lung focus.
- **Laterality markers ("R")** in consistent positions, visible in the NORMAL mean image.
- **Texture/sharpness differs by class** (`hf_energy`, separability 0.86 after preprocessing): NORMAL
  images are sharper even at matched original width (0.81 in the 1100–1500 px band), so it is not a
  resizing artifact. Could be genuine disease signal (opacities blur vessels/ribs) or acquisition
  differences; this dataset cannot tell them apart.
