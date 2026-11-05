# Fracture localization benchmark: RT-DETR vs YOLO26 vs Faster R-CNN

All three detectors are trained on the same leakage-free split and scored on the same test set by one COCO evaluator (pycocotools). Each model gets the same compute: 1.25 hours of training on one Kaggle T4, 640 px input, and the same X-ray augmentation.

## Results (test split: 399 images, 241 fracture boxes)

| Model | mAP@50 | mAP@50-95 | Best F1 (conf) | Image sens. / spec. at best F1 | Latency (T4, batch 1) | Params | Epochs in budget |
|---|---|---|---|---|---|---|---|
| **Faster R-CNN** R50-FPN v2 | **0.410** | **0.127** | **0.48** (0.70) | 0.70 / 0.93 | 65.5 ms | 43.3 M | 11 |
| YOLO26m | 0.200 | 0.065 | 0.30 (0.08) | 0.39 / 0.96 | **30.9 ms** | 20.4 M | 24 |
| RT-DETR-l | 0.112 | 0.040 | 0.19 (0.35) | 0.39 / 0.98 | 70.2 ms | 32.0 M | 14 |

- **mAP:** COCO mAP from pycocotools, with predictions at confidence ≥ 0.001 and up to 100 boxes per image.
- **Best F1:** box-level F1 at IoU 0.5, at the confidence threshold that maximises it. That threshold was picked on the test set, so treat this column as optimistic. The ranking is the same at a fixed 0.25 threshold: F1 is 0.36, 0.06 and 0.12 respectively. YOLO26's scores run low, so 0.25 cuts most of its boxes.
- **Image sens. / spec.:** sensitivity is whether a fractured image gets at least one box; specificity is whether a normal image gets none.
- **Latency:** includes preprocessing and NMS, measured after warm-up.

### Reading these numbers

- **Equal compute, not convergence.** All three models were still improving when the 1.25 h budget ran out. RT-DETR is hit hardest, because transformer detectors usually need far longer schedules. The table answers one question: which detector learns this task best on a small budget. It does not say what each model reaches once fully trained.
- **This is a harder test set than the old one.** The old RT-DETR score (0.71 mAP@50 on 423 images in `../RTDETR/main.ipynb`) was measured on a split with train/test leakage. It should no longer be quoted.
- **Ultralytics skips a few FracAtlas JPEGs.** A few FracAtlas JPEGs have damaged file endings. Ultralytics tries to repair them in place, which fails on Kaggle's read-only input, so it drops them: 13 of 3,922 train images, 4 val and 2 test. The YOLO and RT-DETR runs therefore trained on 0.3% fewer images, and their checkpoints were selected on 395 val images. The test table is unaffected: every model is scored on all 399 test images by the same evaluator. The one image OpenCV cannot decode at all (`fracatlas__IMG0004286.jpg`) gets no YOLO or RT-DETR boxes, so it counts as a miss for them. This costs them well under a point.

## Data

| Source | Kaggle dataset | Used |
|---|---|---|
| Roboflow "Bone Fracture Detection" v4 | `pkdarabi/bone-fracture-detection-computer-vision-project` | All 4,147 images. 7 fracture classes merged into 1, and 2,388 polygon labels converted to boxes. |
| FracAtlas | `tommyngx/fracatlas` | The 717 fractured images (YOLO boxes), plus 717 randomly sampled non-fractured images as backgrounds. |

### Why the sources' own splits were not used

The Roboflow export ships several augmented copies of each radiograph. In its official split, **54 of 169 test images (32%) share a radiograph with the train split**, and so do 108 of 348 validation images (31%). `prep_data.py` regroups the images before splitting:

1. **Same Roboflow source stem.** Copies share the file name before `.rf.`; this added 2,806 links.
2. **Same pixels.** A 64-bit pHash under all 8 flips and rotations (Hamming distance ≤ 10), confirmed by normalised cross-correlation ≥ 0.92 on 64×64 thumbnails. This added 114 merges.

The result is 5,582 images in 2,662 groups. Whole groups go to one split, 70/15/15, stratified by source and by whether the image has boxes. Val and test keep one image per group, so a radiograph is never scored twice; this drops 862 duplicates. The script asserts that no group is shared between splits.

| Split | Images | Fracture boxes | Bone fx / bg | FracAtlas fx / bg |
|---|---|---|---|---|
| train | 3,922 | 2,330 | 1,448 / 1,467 | 504 / 503 |
| val | 399 | 270 | 113 / 85 | 102 / 99 |
| test | 399 | 241 | 110 / 88 | 102 / 99 |

## Training setup

| | RT-DETR-l | YOLO26m | Faster R-CNN R50-FPN v2 |
|---|---|---|---|
| Framework | Ultralytics 8.4.41 | Ultralytics 8.4.41 | torchvision |
| Init | COCO | COCO | COCO |
| Optimizer | AdamW, lr 1e-4 | Ultralytics default (auto) | SGD, lr 0.01, momentum 0.9 |
| Schedule | cosine, `time=1.25` | cosine, `time=1.25` | cosine, epoch count fitted to the budget after timing epoch 1 |
| Batch | 8 | 16 | 8 (AMP) |
| Augmentation | flips (h 0.5, v 0.3), ±10° rotation, brightness ±30%, mosaic 0.5 | same | flips (h 0.5, v 0.3), brightness ±30% |
| Best checkpoint | val fitness | val fitness | val COCO mAP@50 (epoch 8) |

## Files

| File | Purpose |
|---|---|
| `bench_common.py` | Label parsing (polygons converted to boxes), COCO ground truth, the evaluator, and an evaluator sanity check (ground truth scored against itself must give 1.0) that runs before any training. |
| `prep_data.py` | Builds the leakage-free split and writes `split_manifest.csv` and `prep_report.json`. |
| `train_ultralytics.py` | Trains and scores RT-DETR-l or YOLO26m, set by `MODEL_KEY`. |
| `train_faster_rcnn.py` | Trains and scores Faster R-CNN. |
| `stage_kernels.py` | Writes the four Kaggle kernels. Each is `bench_common.py` pasted above its script. |
| `results/` | Each model's `results_<model>.json` from its Kaggle run (test metrics, epochs, latency; Faster R-CNN also has its per-epoch val history), plus the split's `prep_report.json`. |

## Reproduce

```bash
python stage_kernels.py ./kaggle_detbench 1.25
```

1. Push `kaggle_detbench/xcv-detbench-prep` first. It runs on CPU, takes about 10 minutes, and needs no internet.
2. Then push the three GPU kernels, which read the prep kernel's output through `kernel_sources`. Kaggle runs at most two GPU sessions at a time.

Each GPU kernel writes `results_<model>.json` (metrics, epochs, latency) and `test_dets_<model>.json` (every test detection), so the table can be recomputed without retraining.

Kaggle kernels used (private, account `seifeldinmoheb`):

- `xcv-detbench-prep`
- `xcv-detbench-rtdetr`
- `xcv-detbench-yolo`
- `xcv-detbench-frcnn`
