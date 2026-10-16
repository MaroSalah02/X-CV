"""
X-ray validation gate: a small binary classifier that tells plain radiographs (X-ray images)
apart from anything else a user might upload (photos, people, documents, book covers,
CT / MRI / ultrasound, ...). The Pipeline runs it before every other model and stops when
the upload is not an X-ray.

Training data (Kaggle datasets, attached to the kernel):
  X-rays:      FracAtlas, MURA, paediatric chest X-rays, knee X-rays
  not X-rays:  ImageNet-mini, book covers, LFW faces, scanned documents, brain MRI, chest CT,
               breast ultrasound, and generated featureless images (solid colours, gradients)
Held-out test sources (never used for training or threshold tuning) measure how the gate
behaves on image sources it has never seen:
  X-rays:      GRAZPEDWRI-DX (16-bit paediatric wrist), UNIFESP body parts (skull, spine,
               pelvis, abdomen, ...), tuberculosis chest X-rays, dental panoramics
  not X-rays:  Caltech-256 objects, UTKFace (people of all ages), Intel scenes, IAM
               handwriting, a second brain-MRI set
plus grayscale copies of the photo sets and simulated phone photos of X-ray films.

Runs on a Kaggle T4. For a local CPU smoke test set XCV_SMOKE=1, XCV_INPUT_ROOT and XCV_OUT_DIR.
"""
import io
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

SMOKE = os.environ.get('XCV_SMOKE') == '1'
INPUT = os.environ.get('XCV_INPUT_ROOT', '/kaggle/input')
OUT = os.environ.get('XCV_OUT_DIR', '/kaggle/working')
CACHE = os.environ.get('XCV_CACHE_DIR', '/tmp/xray_gate_cache')

SEED = 42
IMG_SIZE = 64 if SMOKE else 224
CACHE_MAX_SIDE = 96 if SMOKE else 288
BATCH = 8 if SMOKE else 64
EPOCHS = 1 if SMOKE else 6
SAMPLES_PER_EPOCH = 64 if SMOKE else 32000
LR = 3e-4
WEIGHT_DECAY = 1e-4
TARGET_XRAY_RECALL = 0.995   # on val: reject at most 0.5% of real X-rays
WORKERS = 0 if SMOKE else 4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# name: (kaggle slug, label 1 = X-ray, role, cap, path must contain)
SOURCES = {
    # training sources
    'fracatlas':        ('tommyngx/fracatlas', 1, 'train', 4100, 'images'),
    'mura':             ('cjinny/mura-v11', 1, 'train', 12000, ''),
    'chest_pneumonia':  ('paultimothymooney/chest-xray-pneumonia', 1, 'train', 5000, ''),
    'knee':             ('harpinders/osteoarthritis-initiative-knee-xray-224-and-299px', 1, 'train', 4000, 'kneeKL224'),
    'imagenet':         ('ifigotin/imagenetmini-1000', 0, 'train', 20000, ''),
    'book_covers':      ('lukaanicin/book-covers-dataset', 0, 'train', 5000, ''),
    'faces_lfw':        ('jessicali9530/lfw-dataset', 0, 'train', 4000, 'lfw-deepfunneled'),
    'documents':        ('shaz13/real-world-documents-collections', 0, 'train', 4000, ''),
    'mri_brain':        ('masoudnickparvar/brain-tumor-mri-dataset', 0, 'train', 4000, ''),
    'ct_chest':         ('plameneduardo/sarscov2-ctscan-dataset', 0, 'train', 3000, ''),
    'ultrasound':       ('aryashah2k/breast-ultrasound-images-dataset', 0, 'train', 2000, ''),
    # held-out test sources
    'graz_wrist':       ('jasonroggy/grazpedwri-dx', 1, 'test', 1500, ''),
    'unifesp_body':     ('ibombonato/xray-body-images-in-png-unifesp-competion', 1, 'test', 2000, 'images'),
    'chest_tb':         ('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', 1, 'test', 1500, ''),
    'dental_panoramic': ('orvile/panoramic-dental-xray-dataset', 1, 'test', 1000, ''),
    'caltech256':       ('jessicali9530/caltech256', 0, 'test', 3000, ''),
    'faces_utk':        ('jangedoo/utkface-new', 0, 'test', 1500, 'UTKFace'),
    'scenes_intel':     ('puneet6060/intel-image-classification', 0, 'test', 1500, ''),
    'handwriting':      ('tejasreddy/iam-handwriting-top50', 0, 'test', 1000, ''),
    'mri_brain_2':      ('navoneel/brain-mri-images-for-brain-tumor-detection', 0, 'test', 300, ''),
}
# featureless images (solid colours, gradients, flat shapes) generated on the fly; without them
# a blank upload scores ~0.5 because nothing in the real datasets looks like it
SYNTHETIC = {'synthetic_flat': ('train', 3000, 0), 'synthetic_flat_test': ('test', 500, 1_000_000)}
# extra held-out variants: (source, variant, cap)
TEST_VARIANTS = [('caltech256', 'gray', 1000), ('faces_utk', 'gray', 1000), ('scenes_intel', 'gray', 1000),
                 ('graz_wrist', 'photo', 500), ('unifesp_body', 'photo', 500), ('chest_tb', 'photo', 500)]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_root(slug):
    owner, name = slug.split('/')
    for p in (os.path.join(INPUT, 'datasets', owner, name), os.path.join(INPUT, name)):
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f'dataset {slug} is not attached (looked under {INPUT})')


def list_images(root, must_contain=''):
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != '__MACOSX']
        for f in files:
            if f.startswith('._') or 'mask' in f.lower():      # AppleDouble junk, segmentation masks
                continue
            p = os.path.join(dirpath, f)
            if os.path.splitext(f)[1].lower() in IMG_EXT and must_contain in os.path.relpath(p, root):
                out.append(p)
    return sorted(out)


def group_of(source, path):
    """Images of one patient / person stay in one split."""
    parts = path.replace('\\', '/').split('/')
    if source == 'mura':
        return next((p for p in parts if p.startswith('patient')), path)
    if source == 'faces_lfw':
        return parts[-2]
    return path


def build_manifest():
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (slug, label, role, cap, must) in SOURCES.items():
        paths = list_images(dataset_root(slug), must)
        n_all = len(paths)
        cap = min(cap, 12) if SMOKE else cap
        if len(paths) > cap:
            paths = sorted(rng.choice(paths, cap, replace=False))
        rows += [{'source': name, 'label': label, 'role': role, 'variant': 'none', 'path': p,
                  'group': group_of(name, p)} for p in paths]
        print(f'{name:17s} {"X-ray" if label else "other":5s} {role:5s} {n_all:6d} found, {len(paths):6d} used')
    for name, (role, n, seed0) in SYNTHETIC.items():
        n = 8 if SMOKE else n
        rows += [{'source': name, 'label': 0, 'role': role, 'variant': 'none', 'path': f'synthetic://{seed0 + i}',
                  'group': f'synthetic://{seed0 + i}'} for i in range(n)]
        print(f'{name:17s} other {role:5s} {n:6d} generated')
    df = pd.DataFrame(rows)

    # train / val split inside each training source (group-aware)
    df['split'] = 'test'
    for name in df.loc[df.role == 'train', 'source'].unique():
        idx = df.index[df.source == name]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr, va = next(gss.split(idx, groups=df.loc[idx, 'group']))
        df.loc[idx[tr], 'split'] = 'train'
        df.loc[idx[va], 'split'] = 'val'

    # held-out variants re-use held-out images under a different transform
    extra = []
    for name, variant, cap in TEST_VARIANTS:
        sub = df[(df.source == name) & (df.split == 'test')]
        sub = sub.sample(min(cap, len(sub)), random_state=SEED).copy()
        sub['variant'] = variant
        extra.append(sub)
    return pd.concat([df] + extra, ignore_index=True)


# ---------------------------------------------------------------- image handling
def to_rgb(img):
    """Any PIL image -> 8-bit RGB. 16-bit radiographs are windowed (0.5-99.5 percentile) first.
    Must match Pipeline/Inference_Modules/xray_validation.py."""
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return Image.fromarray(a.astype(np.uint8), 'L').convert('RGB')
    if img.mode in ('P', 'LA', 'RGBA', 'PA'):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert('RGB')


def synthetic_flat(seed):
    """A featureless image: solid colour, linear or radial gradient, a few flat rectangles or a
    two-tone split, in colour or gray, with optional sensor-like noise."""
    rng = np.random.default_rng(seed)
    w, h = int(rng.integers(160, 900)), int(rng.integers(160, 900))
    gray = rng.random() < 0.5

    def col():
        if gray:
            return np.full(3, rng.uniform(0, 255))
        return rng.uniform(0, 255, 3)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    kind = int(rng.integers(0, 5))
    if kind == 0:                                   # solid colour
        a = np.broadcast_to(col(), (h, w, 3)).astype(np.float32)
    elif kind == 1:                                 # linear gradient
        ang = rng.uniform(0, 2 * np.pi)
        t = np.cos(ang) * xx / w + np.sin(ang) * yy / h
        t = (t - t.min()) / (np.ptp(t) + 1e-6)
        a = (1 - t)[..., None] * col() + t[..., None] * col()
    elif kind == 2:                                 # radial gradient / vignette
        r = np.sqrt((xx - rng.uniform(0, w)) ** 2 + (yy - rng.uniform(0, h)) ** 2)
        t = r / (r.max() + 1e-6)
        a = (1 - t)[..., None] * col() + t[..., None] * col()
    elif kind == 3:                                 # a few flat rectangles (blank slide / UI)
        a = np.broadcast_to(col(), (h, w, 3)).astype(np.float32).copy()
        for _ in range(int(rng.integers(1, 5))):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            a[y0:y0 + int(rng.integers(10, h)), x0:x0 + int(rng.integers(10, w))] = col()
    else:                                           # two-tone split
        a = np.broadcast_to(col(), (h, w, 3)).astype(np.float32).copy()
        if rng.random() < 0.5:
            a[:, int(rng.uniform(0.2, 0.8) * w):] = col()
        else:
            a[int(rng.uniform(0.2, 0.8) * h):, :] = col()
    a = a + rng.normal(0, rng.uniform(0, 12), (h, w, 1 if gray else 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def cache_one(job):
    src, dst = job
    if os.path.exists(dst):
        return True
    try:
        if src.startswith('synthetic://'):
            img = synthetic_flat(int(src.split('//')[1]))
        else:
            img = Image.open(src)
            if img.format == 'JPEG':
                img.draft('RGB', (CACHE_MAX_SIDE, CACHE_MAX_SIDE))
        img = to_rgb(img)
        img.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.Resampling.LANCZOS)
        img.save(dst, quality=95)
        return True
    except Exception as e:  # unreadable / truncated files are dropped
        print('skip', src, type(e).__name__, e)
        return False


def cache_all(df):
    os.makedirs(CACHE, exist_ok=True)
    uniq = df.drop_duplicates('path')
    dst = {p: os.path.join(CACHE, f'{i:07d}.jpg') for i, p in enumerate(uniq.path)}
    jobs = list(dst.items())
    t0 = time.time()
    if WORKERS:
        with Pool(os.cpu_count()) as pool:
            ok = pool.map(cache_one, jobs, chunksize=64)
    else:
        ok = [cache_one(j) for j in jobs]
    good = {p for (p, _), k in zip(jobs, ok) if k}
    print(f'cached {len(good)}/{len(jobs)} images in {time.time() - t0:.0f}s')
    df = df[df.path.isin(good)].copy()
    df['cache'] = df.path.map(dst)
    return df


class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        left, top = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(left, top, s - w - left, s - h - top), fill=0)


def photo_of_film(img, rng):
    """Simulate a phone photo of an X-ray film or monitor: colour cast, uneven lighting,
    raised blacks, a background border, perspective, blur and JPEG compression."""
    w, h = img.size
    a = np.asarray(img, dtype=np.float32) * rng.uniform(0.85, 1.12, size=3)
    gx = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.3), w)[None, :, None]
    gy = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.3), h)[:, None, None]
    a = a * gx * gy + rng.uniform(0, 40)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    pad = int(max(w, h) * rng.uniform(0.03, 0.2))
    # neutral surroundings (lightbox, monitor bezel, desk, wall) with a slight tint
    level = rng.uniform(10, 245)
    bg = tuple(int(np.clip(level * t, 0, 255)) for t in rng.uniform(0.85, 1.15, 3))
    canvas = Image.new('RGB', (w + 2 * pad, h + 2 * pad), bg)
    canvas.paste(img, (pad, pad))
    cw, ch = canvas.size
    start = [[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]]
    j = 0.08
    end = [[int(x + rng.uniform(-j, j) * cw), int(y + rng.uniform(-j, j) * ch)] for x, y in start]
    canvas = TF.perspective(canvas, start, end, fill=list(bg))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.2)))
    buf = io.BytesIO()
    canvas.save(buf, 'JPEG', quality=int(rng.integers(40, 90)))
    return Image.open(io.BytesIO(buf.getvalue())).convert('RGB')


def grayscale_photo(img, rng, invert_p=0.15):
    img = ImageOps.grayscale(img)
    if rng.random() < invert_p:
        img = ImageOps.invert(img)
    return img.convert('RGB')


TRAIN_GEOM = transforms.Compose([
    PadToSquare(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
])
EVAL_TF = transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TO_TENSOR = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class GateDataset(Dataset):
    def __init__(self, df, train):
        self.paths = df.cache.tolist()
        self.labels = df.label.astype(np.float32).tolist()
        self.variants = df.variant.tolist()
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.train:
            rng = np.random.default_rng()
            if self.labels[i] == 1:
                if rng.random() < 0.10:
                    img = ImageOps.invert(img)
                if rng.random() < 0.25:
                    img = photo_of_film(img, rng)
            else:
                if rng.random() < 0.35:
                    img = grayscale_photo(img, rng)
            if rng.random() < 0.15:
                img = img.rotate(90 * int(rng.integers(1, 4)), expand=True)
            x = TO_TENSOR(TRAIN_GEOM(img))
        else:
            rng = np.random.default_rng(i)           # deterministic held-out variants
            if self.variants[i] == 'gray':
                img = grayscale_photo(img, rng, invert_p=0.0)
            elif self.variants[i] == 'photo':
                img = photo_of_film(img, rng)
            x = EVAL_TF(img)
        return x, self.labels[i]


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 1))
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
            out.append(torch.sigmoid(model(x.to(device)).float().squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def sampler_weights(df):
    """50/50 X-ray vs other; inside each class a source gets a share proportional to sqrt(size)."""
    w = np.zeros(len(df))
    for label in (0, 1):
        m = (df.label == label).values
        counts = df[m].source.value_counts()
        share = np.sqrt(counts) / np.sqrt(counts).sum()
        w[m] = 0.5 * df[m].source.map(share).values / df[m].source.map(counts).values
    return w


def save_grid(df, title, path, n=24):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(df) == 0:
        return
    df = df.head(n)
    cols = 6
    rows = (len(df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis('off')
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), df.iterrows()):
        img = Image.open(r.cache).convert('RGB')
        rng = np.random.default_rng(int(r.name))
        if r.variant == 'gray':
            img = grayscale_photo(img, rng, invert_p=0.0)
        elif r.variant == 'photo':
            img = photo_of_film(img, rng)
        ax.imshow(img)
        ax.set_title(f'{r.source}{"/" + r.variant if r.variant != "none" else ""}\np={r.prob:.3f}', fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    seed_everything(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('torch', torch.__version__, '| device:', torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu')
    os.makedirs(OUT, exist_ok=True)

    df = cache_all(build_manifest())
    print(df.groupby(['split', 'label', 'variant']).size().to_string())
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ('train', 'val', 'test'))

    sampler = WeightedRandomSampler(sampler_weights(tr), SAMPLES_PER_EPOCH, replacement=True)
    train_loader = DataLoader(GateDataset(tr, True), BATCH, sampler=sampler, num_workers=WORKERS,
                              pin_memory=device == 'cuda', drop_last=True, persistent_workers=WORKERS > 0)
    val_loader = DataLoader(GateDataset(va, False), BATCH * 2, num_workers=WORKERS)
    test_loader = DataLoader(GateDataset(te, False), BATCH * 2, num_workers=WORKERS)

    model = build_model(pretrained=not SMOKE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')
    lossf = nn.BCEWithLogitsLoss()

    best, history = None, []
    for epoch in range(EPOCHS):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                loss = lossf(model(x).squeeze(1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        p = predict(model, val_loader, device)
        y = va.label.values
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(p).clamp(1e-6, 1 - 1e-6),
                                                         torch.tensor(y, dtype=torch.float32)))
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else float('nan')
        acc = float(((p >= 0.5) == y).mean())
        history.append({'epoch': epoch + 1, 'train_loss': tot / n, 'val_loss': vloss, 'val_auc': auc, 'val_acc': acc})
        print(f'epoch {epoch + 1}/{EPOCHS} train loss {tot / n:.4f} | val loss {vloss:.4f} auc {auc:.5f} '
              f'acc {acc:.4f} | {time.time() - t0:.0f}s')
        if best is None or vloss < best[0]:
            best = (vloss, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    print(f'best epoch {best[1]} (val loss {best[0]:.4f})')
    model.load_state_dict(best[2])

    # threshold on val: default 0.5, lowered only if 0.5 would reject more than 0.5% of val X-rays
    va['prob'] = predict(model, val_loader, device)
    pos = va.prob[va.label == 1].values
    threshold = float(min(0.5, np.quantile(pos, 1 - TARGET_XRAY_RECALL)))
    print(f'threshold {threshold:.4f}')
    val_report = va.assign(correct=lambda d: (d.prob >= threshold) == (d.label == 1)) \
                   .groupby('source').correct.mean()
    print('val accuracy per source:\n' + val_report.round(4).to_string())

    te['prob'] = predict(model, test_loader, device)
    te['accepted'] = te.prob >= threshold
    rep = te.groupby(['label', 'source', 'variant']).agg(n=('prob', 'size'), accepted=('accepted', 'mean'))
    rep['correct'] = np.where(rep.index.get_level_values('label') == 1, rep.accepted, 1 - rep.accepted)
    print('\nheld-out sources (accepted = classified as X-ray):\n' + rep.round(4).to_string())
    pos_t, neg_t = te[te.label == 1], te[te.label == 0]
    summary = {'heldout_xray_accept_rate': float(pos_t.accepted.mean()),
               'heldout_nonxray_reject_rate': float(1 - neg_t.accepted.mean()),
               'heldout_auc': float(roc_auc_score(te.label, te.prob)) if te.label.nunique() == 2 else None}
    print(json.dumps(summary, indent=2))

    save_grid(neg_t.sort_values('prob', ascending=False), 'Held-out NON-X-rays with the highest X-ray score',
              os.path.join(OUT, 'false_accept_candidates.png'))
    save_grid(pos_t.sort_values('prob'), 'Held-out X-rays with the lowest X-ray score',
              os.path.join(OUT, 'false_reject_candidates.png'))

    torch.save({'model_state_dict': model.state_dict(), 'arch': 'efficientnet_b0', 'img_size': IMG_SIZE,
                'threshold': threshold, 'preprocessing': 'pad_to_square_rgb', 'mean': MEAN, 'std': STD,
                'output': 'sigmoid = P(X-ray)', 'best_epoch': best[1]},
               os.path.join(OUT, 'xray_gate.pt'))
    te.drop(columns=['cache']).to_csv(os.path.join(OUT, 'test_predictions.csv'), index=False)
    json.dump({'summary': summary, 'threshold': threshold, 'history': history,
               'val_accuracy_per_source': val_report.to_dict(),
               'heldout': rep.reset_index().to_dict(orient='records'),
               'config': {'img_size': IMG_SIZE, 'epochs': EPOCHS, 'samples_per_epoch': SAMPLES_PER_EPOCH,
                          'lr': LR, 'batch': BATCH, 'sources': {k: v[:4] for k, v in SOURCES.items()}}},
              open(os.path.join(OUT, 'results.json'), 'w'), indent=2, default=str)
    print('saved to', OUT)


if __name__ == '__main__':
    main()
