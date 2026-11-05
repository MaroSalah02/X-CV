"""Build the leakage-free detection benchmark from two public Kaggle datasets.

Sources
  bone       pkdarabi/bone-fracture-detection-computer-vision-project (Roboflow "Bone Fracture
             Detection" v4, YOLO format, 7 fracture classes -> 1)
  fracatlas  tommyngx/fracatlas (original FracAtlas: 717 fractured images with YOLO boxes; an
             equal-size random sample of non-fractured images is kept as backgrounds)

Why not reuse the sources' own splits: the Roboflow export ships several augmented copies of
the same radiograph (`image1_1015_png.rf.<hash>.jpg`), sometimes inside the same test split.
Images are grouped instead, and whole groups are assigned to one split:
  1. same Roboflow source stem (text before `.rf.`), and
  2. near-duplicate pixels: 64-bit pHash under all 8 flips/rotations (Hamming <= 10),
     confirmed by normalised cross-correlation >= 0.92 on 64x64 thumbnails.
Val/test keep one image per group so a radiograph is never scored twice; train keeps all.
Split 70/15/15 by group, stratified by source and by whether the image has boxes.

Output (/kaggle/working/det_bench or OUT_DIR): {train,val,test}/{images,labels}, data.yaml,
coco_val.json, coco_test.json, split_manifest.csv, prep_report.json.
"""
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# bench_common is pasted above this line on Kaggle
try:
    from bench_common import (SEED, IMG_EXTS, find_dir, parse_yolo_label, build_coco_gt)
except ImportError:
    pass

INPUT_ROOT = Path(os.environ.get('INPUT_ROOT', '/kaggle/input'))
OUT_DIR = Path(os.environ.get('OUT_DIR', '/kaggle/working/det_bench'))
SPLIT_FRACTIONS = {'test': 0.15, 'val': 0.15}          # rest is train
HAMMING_MAX = 10
NCC_MIN = 0.92


# ----------------------------------------------------------------------------- listing
def list_bone(root):
    recs = []
    for split in ('train', 'valid', 'test'):
        img_dir = root / split / 'images'
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                lbl = root / split / 'labels' / (p.stem + '.txt')
                recs.append(dict(source='bone', path=p, label_path=lbl,
                                 orig_split='val' if split == 'valid' else split,
                                 stem_key='bone:' + p.name.split('.rf.')[0]))
    return recs


def list_fracatlas(root, rng):
    import pandas as pd
    meta = pd.read_csv(root / 'dataset.csv')
    yolo_dir = find_dir(root, 'YOLO') / 'YOLO'
    img_root = find_dir(root, 'Fractured')
    recs = []
    frac = meta[meta.fractured == 1].image_id.tolist()
    normal = sorted(meta[meta.fractured == 0].image_id.tolist())
    normal = list(rng.choice(normal, size=min(len(frac), len(normal)), replace=False))
    for name, folder in [(n, 'Fractured') for n in frac] + [(n, 'Non_fractured') for n in normal]:
        p = img_root / folder / name
        if not p.exists():
            continue
        lbl = yolo_dir / (Path(name).stem + '.txt') if folder == 'Fractured' else Path('__none__')
        recs.append(dict(source='fracatlas', path=p, label_path=lbl, orig_split='none',
                         stem_key='fracatlas:' + name))
    return recs


# ----------------------------------------------------------------------------- hashing
def _dct_matrix(n):
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_D32 = _dct_matrix(32)
_BITS = (1 << np.arange(64, dtype=np.uint64)).astype(np.uint64)


def _phash(a32):
    low = (_D32 @ a32 @ _D32.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return np.uint64(np.sum(_BITS[bits], dtype=np.uint64))


def _variants(a):
    out = []
    for flip in (False, True):
        b = np.fliplr(a) if flip else a
        for k in range(4):
            out.append(np.rot90(b, k))
    return out


def fingerprint(path):
    with Image.open(path) as im:
        g = im.convert('L')
        a32 = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float64)
        a64 = np.asarray(g.resize((64, 64), Image.BILINEAR), dtype=np.float32)
        size = im.size
    hashes = np.array([_phash(v) for v in _variants(a32)], dtype=np.uint64)
    t = a64 - a64.mean()
    t /= (np.linalg.norm(t) + 1e-6)
    return hashes, t, size


def popcount64(x):
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    bone_root = find_dir(INPUT_ROOT, 'BoneFractureYolo8') / 'BoneFractureYolo8'
    fa_root = find_dir(INPUT_ROOT, 'dataset.csv', is_file=True)
    print('bone root:', bone_root, '| fracatlas root:', fa_root)

    recs = list_bone(bone_root) + list_fracatlas(fa_root, rng)
    n_poly = 0
    for r in recs:
        r['boxes'] = parse_yolo_label(r['label_path'])
        if r['label_path'].exists():
            with open(r['label_path']) as f:
                n_poly += sum(1 for line in f if len(line.split()) > 5)
    print(Counter(r['source'] for r in recs), '| polygon lines converted to boxes:', n_poly)

    # fingerprints
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x
    H, T = [], []
    for r in tqdm(recs, desc='fingerprinting'):
        h, t, r['size'] = fingerprint(r['path'])
        H.append(h); T.append(t)
    H = np.stack(H); T = np.stack(T)
    N = len(recs)

    dsu = DSU(N)
    by_stem = defaultdict(list)
    for i, r in enumerate(recs):
        by_stem[r['stem_key']].append(i)
    n_stem_links = 0
    for idx in by_stem.values():
        for j in idx[1:]:
            dsu.union(idx[0], j); n_stem_links += 1

    n_cand = n_pix = n_cross = 0
    for i in range(N - 1):
        d = popcount64(H[i + 1:] ^ H[i, 0])            # base hash of i vs all variants of j
        best_k = d.argmin(1); best = d[np.arange(len(d)), best_k]
        for off in np.nonzero(best <= HAMMING_MAX)[0]:
            j = i + 1 + off; n_cand += 1
            tj = _variants(T[j])[best_k[off]]
            if float((T[i] * tj).sum()) >= NCC_MIN:
                if dsu.find(i) != dsu.find(j):
                    n_pix += 1
                    n_cross += recs[i]['source'] != recs[j]['source']
                dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(N):
        groups[dsu.find(i)].append(i)
    group_list = sorted(groups.values(), key=lambda g: min(str(recs[i]['path']) for i in g))
    sizes = sorted((len(g) for g in group_list), reverse=True)
    print(f'{N} images -> {len(group_list)} groups (largest {sizes[:5]}); '
          f'pixel near-duplicate merges {n_pix} ({n_cross} across sources) from {n_cand} candidates')

    # stratified group split
    strata = defaultdict(list)
    for g in group_list:
        src = recs[g[0]]['source']
        has_box = any(recs[i]['boxes'] for i in g)
        strata[(src, has_box)].append(g)
    split_of = {}
    for key in sorted(strata):
        gs = strata[key]
        order = rng.permutation(len(gs))
        n = len(gs); n_test = round(n * SPLIT_FRACTIONS['test']); n_val = round(n * SPLIT_FRACTIONS['val'])
        for rank, gi in enumerate(order):
            s = 'test' if rank < n_test else 'val' if rank < n_test + n_val else 'train'
            for i in gs[gi]:
                split_of[i] = s

    # one representative per group in val/test
    pref = {'test': 0, 'val': 1, 'none': 2, 'train': 3}
    keep = set()
    for g in group_list:
        if split_of[g[0]] == 'train':
            keep.update(g)
        else:
            keep.add(min(g, key=lambda i: (pref[recs[i]['orig_split']],
                                           recs[i]['source'] != 'fracatlas', str(recs[i]['path']))))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for s in ('train', 'val', 'test'):
        (OUT_DIR / s / 'images').mkdir(parents=True)
        (OUT_DIR / s / 'labels').mkdir(parents=True)
    gid = {i: k for k, g in enumerate(group_list) for i in g}
    rows = []
    for i, r in enumerate(recs):
        s = split_of[i]
        out_name = f"{r['source']}__{r['path'].stem}{r['path'].suffix.lower()}"
        kept = i in keep
        if kept:
            shutil.copyfile(r['path'], OUT_DIR / s / 'images' / out_name)
            with open(OUT_DIR / s / 'labels' / (Path(out_name).stem + '.txt'), 'w') as f:
                for cx, cy, w, h in r['boxes']:
                    f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        rows.append(dict(out_name=out_name, source=r['source'], orig_path=str(r['path']),
                         orig_split=r['orig_split'], group=gid[i], split=s, kept=int(kept),
                         n_boxes=len(r['boxes'])))
    with open(OUT_DIR / 'split_manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    with open(OUT_DIR / 'data.yaml', 'w') as f:
        f.write(f'path: {OUT_DIR}\ntrain: train/images\nval: val/images\ntest: test/images\n'
                'nc: 1\nnames:\n- fracture\n')
    for s in ('val', 'test'):
        with open(OUT_DIR / f'coco_{s}.json', 'w') as f:
            json.dump(build_coco_gt(OUT_DIR / s), f)

    # leakage checks
    groups_per_split = defaultdict(set)
    for row in rows:
        groups_per_split[row['split']].add(row['group'])
    shared = {f'{a}&{b}': len(groups_per_split[a] & groups_per_split[b])
              for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert not any(shared.values()), shared

    summary = defaultdict(lambda: Counter())
    for row in rows:
        if row['kept']:
            summary[row['split']][f"{row['source']}_{'fx' if row['n_boxes'] else 'bg'}"] += 1
            summary[row['split']]['images'] += 1
            summary[row['split']]['boxes'] += row['n_boxes']
    report = dict(n_input_images=N, n_groups=len(group_list), largest_groups=sizes[:10],
                  stem_links=n_stem_links, pixel_merges=n_pix, cross_source_merges=n_cross,
                  polygon_lines_converted=n_poly, shared_groups_between_splits=shared,
                  dropped_duplicates_in_val_test=N - len(keep),
                  splits={k: dict(v) for k, v in summary.items()})
    with open(OUT_DIR / 'prep_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
