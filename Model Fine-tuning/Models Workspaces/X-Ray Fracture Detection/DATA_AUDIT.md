# Fracture detection data audit

Audit of the datasets behind the fracture classifier, run on 2026-10-15. It explains the
cleaning that `data_preprocessing.py` performs.

| Name in code | Kaggle dataset | Files | Not fractured / fractured |
|---|---|---|---|
| `A` | [bmadushanirodrigo/fracture-multi-region-x-ray-data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) | 10,563 | 5,400 / 5,163 (ships its own train/val/test) |
| `B` | [orvile/bone-fracture-dataset](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset) | 2,127 | 127 / 2,000 |
| `fracatlas` | [tommyngx/fracatlas](https://www.kaggle.com/datasets/tommyngx/fracatlas) | 4,083 | 3,366 / 717 |

## How duplicates were detected

- **Exact copies:** identical MD5.
- **Near copies:** candidate pairs come from a perceptual hash that ignores rotation, flips and
  intensity inversion. A pair counts as the same radiograph when the normalised
  cross-correlation of 96x96 thumbnails is at least 0.97 for one of the 16
  rotation/flip/inversion variants. Visual checks of sampled pairs set that threshold:
  pHash alone matches different patients in the same standard view.
- Label conflicts use a stricter cut-off of NCC ≥ 0.99.

## Findings

### 1. Dataset A is mostly copies of a few hundred radiographs
- 10,563 files contain only 3,871 distinct MD5s and about 2,300 distinct radiographs.
  If augmentation families are merged as well (`12-rotated1-rotated3 - Copy (1).jpg` comes from
  image `12`), A collapses to 494 groups, so 95.3% of its files are redundant.
- A's own split leaks: **738 of 819 val images and 502 of 506 test images have a copy in A/train**.
  873 MD5 groups span more than one split.
- 229 A images are copies of FracAtlas images.

### 2. The old notebook's split leaked test images into training
After MD5 dedup and a random split, as the old notebook did, 119 of 871 test images (13.7%) still
had a near copy in train (A 31/69, B 52/189, FracAtlas 36/613). The test set also held only
20 "not fractured" images from A and B. The deployed checkpoint's **85.9% validation accuracy**
(epoch 12) was measured on a split like this, so it overstates real performance.

### 3. Some radiographs carry both labels
There are 63 pairs of the same radiograph with opposite labels: 52 inside B, 6 between A and FracAtlas,
and 5 inside FracAtlas. For example, FracAtlas `IMG0003034` (fractured) is the same image as
`IMG0004314` and `IMG0004341` (not fractured). The cleaning step drops these clusters.

### 4. Dataset B is unsuitable
- 94% "fracture". The 127 normal files are only 34 distinct radiographs, and the 2,000 fracture
  files are about 909.
- 13 of its "fracture" files are copies of its own normal images.
- It is excluded by default. Pass it in `roots` to `build_manifest` to include it.

### 5. FracAtlas has a hardware shortcut
97 of the 99 FracAtlas images with surgical hardware are labelled fractured, so a model can learn
"metal implant means fracture". The notebook reports test metrics with these images held out separately.
Fracture rate also differs by region: hand 28.5%, leg 11.6%, hip 18.6%, shoulder 18.1%.

### 6. Datasets have very different fracture rates
Dataset A is about 45% fractured and FracAtlas about 17%. If the classes are balanced globally, the
network can partly learn "which dataset is this" instead of "is there a fracture". The training
sampler therefore balances classes within each source.

### 7. Dataset A's labels can be predicted from image style
Found after the first clean training run, on 2026-10-16. A's two classes come from visibly different sources:
- **"Not fractured":** rotated copies of a handful of wrist X-rays. The rotations are by
  arbitrary angles and leave black corners, and the same "RA95" marker repeats.
  These copies don't match at 90-degree steps, so they survive dedup as separate images.
  They still stay in one split, because they share an augmentation family.
- **"Fractured":** images collected from the web, including textbook and journal figures with arrows and
  captions, CT diagrams with text labels, and even MRI.

In the clean A test split, all 117 rotated-copy files are "not fractured". The first clean model scored
AUC 0.9996 on A against 0.91 on FracAtlas, which means it learned the style rather than the
fracture. **A's results say nothing about real performance, and A teaches spurious cues, so it should
be dropped from training.** FracAtlas, which is real clinical radiographs, is the benchmark to trust.

### 8. Preprocessing bugs in the old notebook
- `AugmentedDatasetWrapper` called `ToPILImage()` on tensors that were already ImageNet-normalised.
  The values wrapped around, so the augmented images were scrambled. In a test, the pixel row
  `[0 32 63 96 127 160 191 224]` came out as `[42 165 26 149 14 141 7 129]`.
- Validation and test used `Resize(256)` followed by `CenterCrop(224)`, which cuts off the ends of
  long-bone X-rays where fractures often are. The new pipeline pads to a square instead.

## GRAZPEDWRI-DX: second clinical dataset, used from run 2
[jasonroggy/grazpedwri-dx](https://www.kaggle.com/datasets/jasonroggy/grazpedwri-dx) is a Kaggle copy of
GRAZPEDWRI-DX (Nagy et al., *Scientific Data* 2022): 20,327 paediatric wrist trauma radiographs of
6,091 patients from University Hospital Graz, 2008–2018, annotated by paediatric radiologists. Its
`dataset.csv` gives one row per image with `patient_id`, `fracture_visible`, `cast`, `metal`,
`osteopenia`, `initial_exam`, `diagnosis_uncertain` and the AO fracture classification.

**Label:** fracture visible on this image (`fracture_visible == 1`). The metadata shows several shortcuts:

| Flag | Images | Share fractured | Handling |
|---|---|---|---|
| plaster cast | 5,776 | 96% | dropped (a cast is visible and predicts the label) |
| `diagnosis_uncertain` | 537 | 19% | dropped (unreliable label) |
| no visible fracture, but the study has an AO fracture classification | 773 | (labelled 0) | dropped: the fracture exists but not in this view, so it is not a clean negative |
| metal implant | 708 | 84% | kept as `hardware`; excluded from training and reported separately |
| follow-up exam (`initial_exam` empty) | 9,466 | 95% | kept: healing fractures are still fractures, and the patient split prevents leakage |
| osteopenia | 2,473 | 97% | kept; mostly disuse osteopenia after casting, a real finding but correlated |

After filtering, 13,528 images from 5,546 patients remain, 58.5% fractured.

**No leakage:** the split is **by patient**, so every image of a patient across studies and projections
lands in one split. Images are also deduplicated against each other and against FracAtlas.

**Known limitation:** FracAtlas publishes no patient IDs. Two different views of the same FracAtlas
patient can still fall into different splits; only identical or near-identical images are caught.

## Result: clean split, run 2 (FracAtlas + GRAZPEDWRI-DX)
This split was used by Kaggle run `xcv-fracture-fracatlas-graz` v1 (seed 42). Dataset A is no longer used. FracAtlas keeps exactly
run 1's assignment, so its 596 test images are identical in both runs. GRAZPEDWRI-DX is split by patient.

| Split | FracAtlas not fx | FracAtlas fx | GRAZ not fx | GRAZ fx | Total |
|---|---|---|---|---|---|
| train | 2,297 | 418 | 3,762 | 5,388 | 11,865 |
| val | 477 | 100 | 906 | 1,155 | 2,638 |
| test | 492 | 104 | 887 | 1,133 | 2,616 |

- 17,611 files reduce to 17,482 distinct radiographs in 9,500 split groups.
- The duplicate search found 140 near-duplicate pairs and 15 exact-copy files, all inside FracAtlas; GRAZPEDWRI-DX has none.
- No duplicate crosses the two datasets.
- 4 label-conflicting radiographs (9 files) were dropped.
- 0 split groups and 0 patients are shared between train and val/test.
- The 363 hardware images in train were removed from training. Val and test keep them, and
  they are reported separately: 17 FracAtlas and 69 GRAZ images in test.

### Run 2 results (test set, no hardware)
DenseNet121 at 512px, best val AUC 0.978 at epoch 7, early stop at epoch 12. Train accuracy was 97% against
94% val; in run 1 it was 99.6% against 86%.

| Model | FracAtlas AUC (n=579) | GRAZ AUC (n=1,951) |
|---|---|---|
| run 2 | **0.930** | **0.982** |
| run 1 (A + FracAtlas, 384px) | 0.914 | 0.706 |
| deployed model (trained on part of FracAtlas test) | 0.907 | 0.472 |

- **At matched specificity on FracAtlas, run 2 has the highest sensitivity:** at 90% specificity it is 0.86,
  against 0.80 for run 1 and 0.78 for the deployed model. The AUC gain alone is within noise, since FracAtlas test has only 88 fractures.
- **GRAZ shortcut check:** on initial exams without osteopenia (n=1,515) the AUC is still 0.981, so the
  model does not rely on the follow-up or osteopenia shortcuts.
- **Hardware images are not learned as "fracture":** they were excluded from training. FracAtlas hardware AUC is 0.25 (n=17, only 1 negative) and GRAZ hardware AUC is 0.80.
- **Threshold:** the threshold tuned on the combined val set (0.468) is set mostly by GRAZ and misses 25% of FracAtlas
  fractures. Youden on FracAtlas val gives 0.241, the same value as the mean Youden over both datasets.
  At 0.241, FracAtlas test reaches sens 0.81 / spec 0.95 and GRAZ test reaches sens 0.96 / spec 0.91.
  A 90%-sensitivity threshold (0.044) would cut FracAtlas specificity to 0.63.
- **Deployment path:** scoring the original files with the Pipeline preprocessing gives the same probabilities
  as the training-time cache (correlation 1.000 on FracAtlas and 0.9996 on GRAZ).

## Result: clean split, run 1 (A + FracAtlas)
This is the split used by the first Kaggle training run (`xcv-fracture-clean-training` v1, seed 42), built from A and FracAtlas.
Every copy of a radiograph lands in the same split, and no split group is shared between train
and val/test. The run's `clean_manifest.csv` holds the exact assignment.

| Split | A not fx | A fx | FracAtlas not fx | FracAtlas fx | Total |
|---|---|---|---|---|---|
| train | 968 | 798 | 2,297 | 484 | 4,547 |
| val | 122 | 153 | 477 | 100 | 852 |
| test | 122 | 56 | 492 | 104 | 774 |

14,664 files reduce to 6,173 distinct radiographs in 4,368 split groups. 4 label-conflicting
radiographs (15 files) were dropped. A images that copy FracAtlas images are kept as their
FracAtlas originals. Kaggle's mounted copy of A has 10,581 files, 18 more than the downloaded zip
used for the audit numbers above, so the exact split depends on the dataset version.
