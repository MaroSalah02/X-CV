"""Train and score torchvision Faster R-CNN (ResNet-50 FPN v2, COCO-pretrained) on the
prepared benchmark, with the same time budget, input size and evaluator as the
Ultralytics models.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

try:
    from bench_common import (SEED, find_bench_root, build_coco_gt, coco_evaluate,
                              sanity_check_evaluator, read_split_labels)
except ImportError:
    pass

BUDGET_HOURS = float(os.environ.get('BUDGET_HOURS', '1.5'))
WORK = Path(os.environ.get('WORK_DIR', '/kaggle/working'))
BATCH = int(os.environ.get('BATCH', '8'))
IMG_SIZE = int(os.environ.get('IMG_SIZE', '640'))
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', '40'))
PATIENCE = 6
SMOKE = os.environ.get('SMOKE') == '1'                   # tiny CPU run for local testing


def main():
    import torch
    import torchvision
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import torchvision.transforms.functional as TF
    from PIL import Image

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch', torch.__version__, '| torchvision', torchvision.__version__, '|', dev)

    bench = find_bench_root()
    test_gt = build_coco_gt(bench / 'test')
    val_gt = build_coco_gt(bench / 'val')
    sanity_check_evaluator(test_gt)

    class DetSet(Dataset):
        def __init__(self, split, train):
            self.dir = bench / split / 'images'
            self.items = list(read_split_labels(bench / split).items())
            self.train = train

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            name, boxes = self.items[i]
            img = Image.open(self.dir / name).convert('RGB')
            W, H = img.size
            b = torch.tensor([[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
                              for cx, cy, w, h in boxes], dtype=torch.float32).reshape(-1, 4)
            if self.train:   # same family of augmentation as the Ultralytics runs
                if random.random() < 0.5:
                    img = TF.hflip(img); b[:, [0, 2]] = W - b[:, [2, 0]]
                if random.random() < 0.3:
                    img = TF.vflip(img); b[:, [1, 3]] = H - b[:, [3, 1]]
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            t = TF.to_tensor(img)
            return t, {'boxes': b, 'labels': torch.ones(len(b), dtype=torch.int64)}, name

    collate = lambda batch: tuple(zip(*batch))
    train_ds = DetSet('train', True)
    if SMOKE:
        train_ds.items = train_ds.items[:4]
    nw = 0 if SMOKE else 4
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=nw,
                          collate_fn=collate, drop_last=True, persistent_workers=nw > 0)

    weights = None if SMOKE else 'DEFAULT'
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=IMG_SIZE, max_size=IMG_SIZE)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    model.to(dev)

    @torch.no_grad()
    def predict(split, limit=None):
        model.eval()
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 100
        ds = DetSet(split, False)
        if limit:
            ds.items = ds.items[:limit]
        dl = DataLoader(ds, batch_size=BATCH, num_workers=nw, collate_fn=collate)
        dets = []
        for imgs, _, names in dl:
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                outs = model([im.to(dev) for im in imgs])
            for o, n in zip(outs, names):
                for b, s in zip(o['boxes'].float().tolist(), o['scores'].float().tolist()):
                    dets.append({'file_name': n, 'bbox_xyxy': b, 'score': s})
        model.train()
        return dets

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01 * BATCH / 8, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    iters_per_epoch = len(train_dl)
    warmup = min(500, iters_per_epoch)

    # Time the first epoch, then fix the epoch count so the cosine schedule ends inside the
    # budget (same idea as Ultralytics' `time=` argument).
    t0 = time.time()
    total_epochs = MAX_EPOCHS
    total_iters = total_epochs * iters_per_epoch
    best_ap, best_epoch, history, it = -1.0, 0, [], 0
    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        te = time.time()
        loss_sum = 0.0
        for imgs, targets, _ in train_dl:
            lr_scale = (it + 1) / warmup if it < warmup else \
                0.5 * (1 + math.cos(math.pi * min(1.0, (it - warmup) / max(1, total_iters - warmup))))
            for g in opt.param_groups:
                g['lr'] = 0.01 * BATCH / 8 * max(lr_scale, 0.01)
            imgs = [im.to(dev) for im in imgs]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            loss_sum += loss.item(); it += 1
        val = coco_evaluate(val_gt, predict('val', limit=4 if SMOKE else None)) if not SMOKE else {'mAP50': 0.0}
        epoch_s = time.time() - te
        if epoch == 1:
            budget_s = BUDGET_HOURS * 3600
            total_epochs = max(2, min(MAX_EPOCHS, int(budget_s // epoch_s)))
            total_iters = total_epochs * iters_per_epoch
            print(f'epoch takes {epoch_s / 60:.1f} min -> {total_epochs} epochs fit the budget')
        history.append({'epoch': epoch, 'loss': loss_sum / max(1, iters_per_epoch),
                        'val_mAP50': val['mAP50'], 'minutes': epoch_s / 60})
        print(history[-1])
        if val['mAP50'] > best_ap:
            best_ap, best_epoch = val['mAP50'], epoch
            torch.save(model.state_dict(), WORK / 'faster_rcnn_best.pt')
        if epoch >= total_epochs or epoch - best_epoch >= PATIENCE or SMOKE:
            break
    train_hours = (time.time() - t0) / 3600

    model.load_state_dict(torch.load(WORK / 'faster_rcnn_best.pt', map_location=dev))
    dets = predict('test', limit=4 if SMOKE else None)
    metrics = coco_evaluate(test_gt, dets)

    # latency: batch of one, after warm-up
    model.eval()
    model.roi_heads.score_thresh = 0.25
    names = sorted(read_split_labels(bench / 'test'))[: (6 if SMOKE else 120)]
    imgs = [TF.to_tensor(Image.open(bench / 'test' / 'images' / n).convert('RGB')) for n in names]
    with torch.no_grad():
        for im in imgs[:2 if SMOKE else 20]:
            model([im.to(dev)])
        if dev.type == 'cuda': torch.cuda.synchronize()
        t = time.time()
        timed = imgs[2 if SMOKE else 20:]
        for im in timed:
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                model([im.to(dev)])
        if dev.type == 'cuda': torch.cuda.synchronize()
    metrics['latency_ms_per_image'] = (time.time() - t) / len(timed) * 1000

    out = dict(model='faster-rcnn-r50-fpn-v2', params_M=sum(p.numel() for p in model.parameters()) / 1e6,
               epochs=len(history), best_epoch=best_epoch, train_hours=train_hours,
               budget_hours=BUDGET_HOURS, gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu',
               n_test_images=len(test_gt['images']), n_test_boxes=len(test_gt['annotations']),
               test=metrics, history=history)
    with open(WORK / 'results_faster-rcnn.json', 'w') as f:
        json.dump(out, f, indent=2)
    with open(WORK / 'test_dets_faster-rcnn.json', 'w') as f:
        json.dump(dets, f)
    print(json.dumps({k: v for k, v in out.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
