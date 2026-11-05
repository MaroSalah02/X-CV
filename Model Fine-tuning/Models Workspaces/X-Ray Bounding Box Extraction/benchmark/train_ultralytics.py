"""Train and score an Ultralytics detector (RT-DETR or YOLO) on the prepared benchmark.

`stage_kernels.py` sets MODEL_KEY for each kernel. Every model gets the same wall-clock
training budget (BUDGET_HOURS) on one T4, the same 640px input and the same X-ray
augmentation, then is scored by the shared COCO evaluator on the held-out test split.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from bench_common import (SEED, find_bench_root, build_coco_gt, coco_evaluate,
                              sanity_check_evaluator)
except ImportError:
    pass

MODEL_KEY = os.environ.get('MODEL_KEY', 'yolo26m')       # replaced by stage_kernels.py
BUDGET_HOURS = float(os.environ.get('BUDGET_HOURS', '1.5'))
ULTRALYTICS_VERSION = '8.4.41'                           # version the deployed RT-DETR used
WORK = Path(os.environ.get('WORK_DIR', '/kaggle/working'))

MODELS = {
    # RT-DETR settings follow the notebook that produced the deployed model
    # (AdamW, lr 1e-4: higher rates diverged to NaN there).
    'rtdetr-l': dict(cls='RTDETR', weights='rtdetr-l.pt', batch=8,
                     extra=dict(optimizer='AdamW', lr0=1e-4, lrf=0.01, weight_decay=1e-4,
                                warmup_epochs=3)),
    'yolo26m': dict(cls='YOLO', weights='yolo26m.pt', batch=16, extra=dict()),
}
AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.3, degrees=10, mosaic=0.5)


SMOKE = os.environ.get('SMOKE') == '1'                   # tiny CPU run for local testing


def main():
    try:
        import ultralytics
        assert ultralytics.__version__ == ULTRALYTICS_VERSION
    except (ImportError, AssertionError):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        f'ultralytics=={ULTRALYTICS_VERSION}'], check=True)
    import torch
    import ultralytics
    from ultralytics import RTDETR, YOLO
    print('ultralytics', ultralytics.__version__, '| torch', torch.__version__,
          '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')

    bench = find_bench_root()
    test_gt = build_coco_gt(bench / 'test')
    sanity_check_evaluator(test_gt)

    data_yaml = WORK / 'data.yaml'
    data_yaml.write_text(f'path: {bench}\ntrain: train/images\nval: val/images\n'
                         'test: test/images\nnc: 1\nnames:\n- fracture\n')

    cfg = MODELS[MODEL_KEY]
    Model = RTDETR if cfg['cls'] == 'RTDETR' else YOLO
    model = Model(os.environ.get('WEIGHTS', cfg['weights']))
    run = dict(imgsz=640, epochs=300, time=BUDGET_HOURS, batch=cfg['batch'], workers=4, cache='ram')
    if SMOKE:
        run = dict(imgsz=128, epochs=1, batch=2, workers=0, cache=False, device='cpu')
    t0 = time.time()
    model.train(data=str(data_yaml), patience=20, seed=SEED, cos_lr=True,
                project=str(WORK / 'runs'), name=MODEL_KEY, exist_ok=True, plots=not SMOKE,
                **run, **AUG, **cfg['extra'])
    train_hours = (time.time() - t0) / 3600

    save_dir = Path(model.trainer.save_dir)              # Ultralytics may nest the project dir
    best = save_dir / 'weights' / 'best.pt'
    model = Model(str(best))

    # detections on the test split, low confidence so the PR curve is complete
    test_dir = bench / 'test' / 'images'
    dets = []
    for r in model.predict(source=str(test_dir), imgsz=640, conf=0.001, max_det=100,
                           stream=True, verbose=False):
        name = Path(r.path).name
        for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            dets.append({'file_name': name, 'bbox_xyxy': b, 'score': s})
    metrics = coco_evaluate(test_gt, dets)

    # latency: one image at a time, after warm-up, preprocessing + NMS included
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    imgs = sorted(test_dir.iterdir())[:120]
    n_warm = 2 if SMOKE else 20
    for p in imgs[:n_warm]:
        model.predict(str(p), imgsz=640, conf=0.25, verbose=False)
    sync()
    t = time.time()
    for p in imgs[n_warm:]:
        model.predict(str(p), imgsz=640, conf=0.25, verbose=False)
    sync()
    metrics['latency_ms_per_image'] = (time.time() - t) / len(imgs[n_warm:]) * 1000

    # Ultralytics' own validator, for reference only (different AP interpolation)
    uv = model.val(data=str(data_yaml), split='test', imgsz=640, conf=0.001, plots=False)
    metrics['ultralytics_mAP50'] = float(uv.box.map50)
    metrics['ultralytics_mAP50_95'] = float(uv.box.map)

    import csv
    with open(save_dir / 'results.csv') as f:
        epochs_run = sum(1 for _ in csv.DictReader(f))
    out = dict(model=MODEL_KEY, params_M=sum(p.numel() for p in model.model.parameters()) / 1e6,
               epochs=epochs_run, train_hours=train_hours, budget_hours=BUDGET_HOURS,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
               n_test_images=len(test_gt['images']),
               n_test_boxes=len(test_gt['annotations']), test=metrics)
    with open(WORK / f'results_{MODEL_KEY}.json', 'w') as f:
        json.dump(out, f, indent=2)
    with open(WORK / f'test_dets_{MODEL_KEY}.json', 'w') as f:
        json.dump(dets, f)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
