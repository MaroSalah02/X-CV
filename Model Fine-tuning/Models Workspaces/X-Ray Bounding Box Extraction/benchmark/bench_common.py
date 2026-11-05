"""Shared code for the fracture-localization benchmark (RT-DETR vs YOLO vs Faster R-CNN).

All three detectors are trained on the same leakage-free split built by `prep_data.py`
and scored on the same test set with the same COCO evaluator (pycocotools), so their
numbers can be compared directly.

Kaggle kernels are single files, so `stage_kernels.py` pastes this module on top of each
kernel script. Keep it free of heavy imports at module level (cv2, torch, ultralytics).
"""
import json
import os
import time
from pathlib import Path

import numpy as np

SEED = 42
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# at least one Roboflow JPEG is cut short by a few bytes; decode what is there
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------------------------------------------------------- paths
def find_dir(root, marker, is_file=False):
    """First directory under `root` that contains `marker` (a file or folder name).

    Kaggle has changed where inputs are mounted more than once, so search instead of
    hard-coding /kaggle/input/<slug>.
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        names = filenames if is_file else dirnames
        if marker in names:
            return Path(dirpath)
        # skip leaf folders that only hold thousands of images / label files
        dirnames[:] = [d for d in dirnames if d not in ('labels', 'Fractured', 'Non_fractured', 'PASCAL VOC')]
    raise FileNotFoundError(f"'{marker}' not found under {root}")


# ----------------------------------------------------------------------------- labels
def parse_yolo_label(path):
    """YOLO label file -> list of [cx, cy, w, h] (normalised), every class mapped to 'fracture'.

    Some Roboflow exports mix boxes (5 values) with polygons (class + x1 y1 x2 y2 ...).
    Polygons are turned into their bounding box instead of being dropped, which would
    silently make a fractured image look like a background image.
    """
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                cx, cy, w, h = map(float, parts[1:])
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                pts = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            else:
                continue
            x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
            if x1 - x0 > 1e-4 and y1 - y0 > 1e-4:
                boxes.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    return boxes


def read_split_labels(split_dir):
    """{image file name: [[cx, cy, w, h], ...]} for a prepared split folder."""
    split_dir = Path(split_dir)
    out = {}
    for img in sorted((split_dir / 'images').iterdir()):
        if img.suffix.lower() in IMG_EXTS:
            out[img.name] = parse_yolo_label(split_dir / 'labels' / (img.stem + '.txt'))
    return out


# ----------------------------------------------------------------------------- COCO eval
def build_coco_gt(split_dir):
    """COCO ground-truth dict for a prepared split (category 1 = fracture)."""
    from PIL import Image
    split_dir = Path(split_dir)
    images, anns = [], []
    for i, (name, boxes) in enumerate(read_split_labels(split_dir).items(), start=1):
        with Image.open(split_dir / 'images' / name) as im:
            W, H = im.size
        images.append({'id': i, 'file_name': name, 'width': W, 'height': H})
        for cx, cy, w, h in boxes:
            bw, bh = w * W, h * H
            anns.append({'id': len(anns) + 1, 'image_id': i, 'category_id': 1,
                         'bbox': [(cx - w / 2) * W, (cy - h / 2) * H, bw, bh],
                         'area': bw * bh, 'iscrowd': 0})
    return {'images': images, 'annotations': anns,
            'categories': [{'id': 1, 'name': 'fracture'}]}


def coco_evaluate(gt, dets):
    """gt: COCO dict; dets: list of {'file_name', 'bbox_xyxy', 'score'} in pixels.

    Returns mAP@50, mAP@50-95, AR@100 (pycocotools) plus precision / recall / F1 at
    confidence 0.25 and IoU 0.5, the operating point the app uses.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    name_to_id = {im['file_name']: im['id'] for im in gt['images']}
    res = []
    for d in dets:
        x0, y0, x1, y1 = d['bbox_xyxy']
        res.append({'image_id': name_to_id[d['file_name']], 'category_id': 1,
                    'bbox': [x0, y0, x1 - x0, y1 - y0], 'score': float(d['score'])})
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = json.loads(json.dumps(gt))
        coco_gt.createIndex()
        if res:
            coco_dt = coco_gt.loadRes(res)
            ev = COCOeval(coco_gt, coco_dt, 'bbox')
            ev.evaluate(); ev.accumulate(); ev.summarize()
            stats = ev.stats
        else:
            stats = np.zeros(12)
    out = {'mAP50': float(stats[1]), 'mAP50_95': float(stats[0]), 'AR100': float(stats[8])}
    out.update(pr_at_threshold(gt, dets, conf=0.25, iou_thr=0.5))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pr_at_threshold(gt, dets, conf=0.25, iou_thr=0.5):
    """Box-level precision/recall/F1 at one confidence, plus image-level sensitivity and
    specificity (does the image get at least one box?)."""
    id_to_name = {im['id']: im['file_name'] for im in gt['images']}
    gt_boxes = {im['file_name']: [] for im in gt['images']}
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        gt_boxes[id_to_name[a['image_id']]].append([x, y, x + w, y + h])
    pred = {n: [] for n in gt_boxes}
    for d in dets:
        if d['score'] >= conf:
            pred[d['file_name']].append((d['score'], d['bbox_xyxy']))
    tp = fp = fn = 0
    img_tp = img_fn = img_tn = img_fp = 0
    for name, gts in gt_boxes.items():
        ps = sorted(pred[name], key=lambda t: -t[0])
        used = [False] * len(gts)
        for _, pb in ps:
            best, bj = 0.0, -1
            for j, gb in enumerate(gts):
                if not used[j]:
                    v = _iou(pb, gb)
                    if v > best:
                        best, bj = v, j
            if best >= iou_thr:
                used[bj] = True; tp += 1
            else:
                fp += 1
        fn += used.count(False)
        if gts:
            img_tp += bool(ps); img_fn += not ps
        else:
            img_fp += bool(ps); img_tn += not ps
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {'precision@0.25': p, 'recall@0.25': r,
            'F1@0.25': 2 * p * r / (p + r) if p + r else 0.0,
            'image_sensitivity@0.25': img_tp / (img_tp + img_fn) if img_tp + img_fn else 0.0,
            'image_specificity@0.25': img_tn / (img_tn + img_fp) if img_tn + img_fp else 0.0}


def sanity_check_evaluator(gt):
    """Ground truth scored against itself must give mAP 1.0; run before any training so
    a broken evaluator fails in seconds instead of after hours of GPU time."""
    id_to_name = {im['id']: im['file_name'] for im in gt['images']}
    dets = []
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        dets.append({'file_name': id_to_name[a['image_id']], 'bbox_xyxy': [x, y, x + w, y + h], 'score': 0.99})
    m = coco_evaluate(gt, dets)
    assert m['mAP50'] > 0.99 and m['recall@0.25'] > 0.99, f"evaluator sanity check failed: {m}"
    print('evaluator sanity check ok:', {k: round(v, 3) for k, v in m.items()})


def time_budget_guard(start, hours):
    """True while there is time left in the training budget."""
    return (time.time() - start) < hours * 3600


def find_bench_root():
    """Prepared dataset: /kaggle/input/... when run on Kaggle, else env BENCH_ROOT."""
    if os.environ.get('BENCH_ROOT'):
        return Path(os.environ['BENCH_ROOT'])
    return find_dir('/kaggle/input', 'split_manifest.csv', is_file=True)
