"""
Data pipeline for X-ray fracture classification.

Why this module exists (see DATA_AUDIT.md for the numbers):
  * The source datasets are full of exact and near-duplicate images (Dataset A is ~95%
    rotated / re-saved copies of a few hundred radiographs; FracAtlas re-uses some images
    under new IDs), so a random file-level split leaks test images into training.
  * Some radiographs appear with BOTH labels (Dataset B normals copied into its fracture
    folder, FracAtlas IMG0003034 == IMG0004314 == IMG0004341, ...).
  * Dataset B is 94% "fracture" and padded with copies (2000 fracture files from ~909
    distinct radiographs, 127 normal files from 34), and 13 of its "fracture" files are
    copies of its normals; it is excluded by default.
  * Dataset A's two classes come from visibly different sources (rotated wrist copies vs
    web figures, CT and MRI), so its labels are predictable from style; prefer
    FracAtlas + GRAZPEDWRI-DX, which are clinical radiographs.

Pipeline:
  1. build_manifest(): enumerate every source, fingerprint each file
     (md5, 16 rotation/flip/inversion-invariant perceptual hashes, 96x96 thumbnail).
  2. clean_and_split():
       - duplicate clusters := same md5, or 96px normalised cross-correlation >= 0.97
         under any of the 16 variants (calibrated by visual inspection);
       - clusters whose members carry different labels are dropped;
       - one canonical file per cluster is kept (FracAtlas original > copies);
       - split groups := duplicate clusters joined with Dataset A augmentation
         families ("12-rotated1-rotated3 (1).jpg" -> family "12") and with
         GRAZPEDWRI-DX patients, so every copy of a radiograph and every image of a
         patient lands in the same split;
       - optional fixed splits (e.g. reuse an earlier run's FracAtlas test set so results
         stay comparable), then a stratified (source x label) group split 70/15/15.
  3. cache_resized(): decode each kept image once and store a small 8-bit PNG, so
     training epochs do not re-decode full-resolution (often 16-bit) originals.
  4. get_dataloaders(): pad-to-square preprocessing (no centre crop that cuts off the
     ends of long-bone X-rays), mild geometric/intensity augmentation for training,
     and a sampler that balances classes *within each source* so the network cannot
     use "which dataset is this?" as a proxy for the label.
"""

import hashlib
import os
import re
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile, ImageOps
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Allow loading truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SOURCE_PRIORITY = {'fracatlas': 0, 'graz': 1, 'A': 2, 'B': 3}   # which copy of a duplicate to keep
THUMB = 96                                           # thumbnail side for NCC matching


# ============================================================================
# Source enumeration
# ============================================================================

def _list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(directory, f)))


def list_dataset_A(root: str) -> List[Dict]:
    """Bone_Fracture_Binary_Classification: {train,val,test}/{fractured,not fractured}/"""
    records = []
    for split in ['train', 'val', 'test']:
        for folder, label in [('not fractured', 0), ('fractured', 1)]:
            d = os.path.join(root, split, folder)
            for f in _list_images(d):
                records.append(dict(source='A', path=os.path.join(d, f), name=f, label=label, orig_split=split))
    if not records:
        raise FileNotFoundError(f"No images found under Dataset A root: {root}")
    return records


def list_dataset_B(root: str) -> List[Dict]:
    """Bone fracture dataset: {fracture,normal}/"""
    records = []
    for folder, label in [('normal', 0), ('fracture', 1)]:
        d = os.path.join(root, folder)
        for f in _list_images(d):
            records.append(dict(source='B', path=os.path.join(d, f), name=f, label=label, orig_split=''))
    if not records:
        raise FileNotFoundError(f"No images found under Dataset B root: {root}")
    return records


def list_fracatlas(root: str) -> List[Dict]:
    """FracAtlas: images/{Fractured,Non_fractured}/ (+ dataset.csv metadata if present)"""
    meta = {}
    csv_path = os.path.join(root, 'dataset.csv')
    if os.path.exists(csv_path):
        meta = pd.read_csv(csv_path).set_index('image_id').to_dict('index')
    records = []
    for folder, label in [('Non_fractured', 0), ('Fractured', 1)]:
        d = os.path.join(root, 'images', folder)
        for f in _list_images(d):
            m = meta.get(f, {})
            records.append(dict(source='fracatlas', path=os.path.join(d, f), name=f, label=label,
                                orig_split='', hardware=m.get('hardware', np.nan)))
    if not records:
        raise FileNotFoundError(f"No images found under FracAtlas root: {root}")
    return records


def list_grazpedwri(root: str, drop_cast: bool = True, drop_uncertain: bool = True,
                    drop_ambiguous_negatives: bool = True) -> List[Dict]:
    """
    GRAZPEDWRI-DX: pediatric wrist radiographs, dataset.csv has one row per image
    (filestem, patient_id, fracture_visible, cast, metal, ...). PNGs are located anywhere
    under root by filestem.

    Label = fracture visible on this image. Rows whose label is unreliable or trivially
    predictable are dropped:
      * cast: 96% of images with a plaster cast are fractured, so the cast is a shortcut
      * diagnosis_uncertain
      * ambiguous negatives: no visible fracture although the study has an AO fracture
        classification (the fracture exists, it just is not visible in this projection)
    Metal implants are kept and exposed as 'hardware', as for FracAtlas.
    """
    meta = pd.read_csv(os.path.join(root, 'dataset.csv'))
    fx = meta['fracture_visible'].eq(1)
    keep = pd.Series(True, index=meta.index)
    if drop_cast:
        keep &= meta['cast'].isna()
    if drop_uncertain:
        keep &= meta['diagnosis_uncertain'].isna()
    if drop_ambiguous_negatives:
        keep &= fx | meta['ao_classification'].isna()

    files = {}
    for dirpath, _, names in os.walk(root):
        for f in names:
            if f.lower().endswith('.png'):
                p = os.path.join(dirpath, f)
                stem = os.path.splitext(f)[0]
                # if a stem exists twice (e.g. original + converted copy) keep the larger file
                if stem not in files or os.path.getsize(p) > os.path.getsize(files[stem]):
                    files[stem] = p

    records, missing = [], 0
    for row, k, y in zip(meta.itertuples(index=False), keep, fx):
        if not k:
            continue
        p = files.get(row.filestem)
        if p is None:
            missing += 1
            continue
        records.append(dict(source='graz', path=p, name=os.path.basename(p), label=int(y), orig_split='',
                            hardware=float(row.metal == 1), patient_id=int(row.patient_id)))
    print(f"GRAZPEDWRI-DX: {len(meta)} rows, {int((~keep).sum())} filtered out, "
          f"{missing} images not found, {len(records)} kept")
    if not records:
        raise FileNotFoundError(f"No GRAZPEDWRI-DX images found under: {root}")
    return records


SOURCE_LISTERS = {'A': list_dataset_A, 'B': list_dataset_B, 'fracatlas': list_fracatlas,
                  'graz': list_grazpedwri}


# ============================================================================
# Fingerprinting
# ============================================================================

_POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Per-element bit count of a uint64 array (np.bitwise_count needs NumPy >= 2.0)."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    return _POP8[np.ascontiguousarray(x).view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1)


def _phash(a32: np.ndarray) -> int:
    d = dct(dct(a32, axis=0), axis=1)[:8, :8]
    return int(np.packbits((d > np.median(d)).flatten()).view('>u8')[0])


def _variants(a: np.ndarray) -> List[np.ndarray]:
    """4 rotations x mirror x intensity inversion = 16 variants."""
    out = []
    for base in (a, a.max() + a.min() - a):
        for k in range(4):
            r = np.rot90(base, k)
            out += [r, np.fliplr(r)]
    return out


def fingerprint(path: str) -> Dict:
    """md5, size, 16 perceptual hashes and a standardised 96x96 thumbnail of one image."""
    out = {'path': path}
    try:
        raw = open(path, 'rb').read()
        out['md5'] = hashlib.md5(raw).hexdigest()
        im = Image.open(path)
        out['w'], out['h'] = im.size
        if im.format == 'JPEG':
            im.draft('L', (2 * THUMB, 2 * THUMB))
        g = to_grayscale(im)
        a32 = np.asarray(g.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        out['phash'] = np.array([_phash(v) for v in _variants(a32)], dtype=np.uint64)
        t = np.asarray(g.resize((THUMB, THUMB), Image.Resampling.BILINEAR), dtype=np.float32)
        out['thumb'] = ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float16)
        out['ok'] = True
    except Exception as e:  # unreadable file
        out['ok'] = False
        out['error'] = repr(e)[:200]
    return out


def build_manifest(roots: Dict[str, str],
                   cache_path: Optional[str] = None,
                   num_workers: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Enumerate and fingerprint all images of the given sources.

    Args:
        roots: {'A': path, 'fracatlas': path, 'B': path}; only listed sources are used
        cache_path: optional .pkl path; fingerprints are reused if the file list is unchanged
        num_workers: processes for fingerprinting (default: all CPUs)

    Returns:
        (manifest DataFrame, thumbnails array [N, 96, 96] float16)
    """
    records = []
    for source, root in roots.items():
        records += SOURCE_LISTERS[source](root)
    df = pd.DataFrame(records)
    paths = df['path'].tolist()

    # The cache stores the full file listing, since unreadable files are dropped from its rows
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        if isinstance(cached, dict) and cached.get('listed') == paths:
            print(f"✓ Loaded fingerprints from cache: {cache_path}")
            c = cached['df']
            return c.drop(columns=['thumb']), np.stack(c['thumb'].values)

    workers = num_workers or os.cpu_count() or 1
    if workers > 1:
        with Pool(workers) as pool:
            fps = list(tqdm(pool.imap(fingerprint, paths, chunksize=32), total=len(paths), desc='Fingerprinting'))
    else:
        fps = [fingerprint(p) for p in tqdm(paths, desc='Fingerprinting')]

    fp = pd.DataFrame(fps)
    bad = ~fp['ok']
    if bad.any():
        print(f"⚠ {int(bad.sum())} unreadable images skipped")
    df = pd.concat([df, fp.drop(columns=['path'])], axis=1)[~bad.values].reset_index(drop=True)
    if cache_path:
        pd.to_pickle({'listed': paths, 'df': df}, cache_path)
    return df.drop(columns=['thumb']), np.stack(df['thumb'].values)


# ============================================================================
# Duplicate detection, cleaning and splitting
# ============================================================================

class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def find_duplicate_pairs(df: pd.DataFrame, thumbs: np.ndarray,
                         phash_max: int = 12, ncc_threshold: float = 0.97) -> List[Tuple[int, int, float]]:
    """
    Near-duplicate pairs (i, j, ncc). Candidates come from the rotation/flip/inversion
    invariant pHash (distance <= phash_max); a candidate is accepted when the best
    normalised cross-correlation of the 96x96 thumbnails over the 16 variants is
    >= ncc_threshold. pHash alone is not discriminative for X-rays (different patients
    in the same standard view collide), the NCC check removes those false matches.
    """
    H = np.stack(df['phash'].values)               # N x 16 (uint64)
    n = len(H)
    T = thumbs.astype(np.float32)
    # cheap 32x32 prefilter (3x3 block means of the 96px thumbnails); the 16 standardised
    # variants of every image are built once (N x 16 x 1024 float32, ~1.2 GB for 18k images)
    t32 = T.reshape(n, 32, 3, 32, 3).mean(axis=(2, 4))
    t32 = (t32 - t32.mean(axis=(1, 2), keepdims=True)) / (t32.std(axis=(1, 2), keepdims=True) + 1e-6)
    flat32 = t32.reshape(n, -1)
    V32 = np.stack([np.stack(_variants(t)).reshape(16, -1) for t in t32])
    V32 = (V32 - V32.mean(axis=2, keepdims=True)) / (V32.std(axis=2, keepdims=True) + 1e-6)

    # pHash candidates are prefiltered block by block, so the (possibly huge) candidate list
    # for look-alike views, e.g. thousands of standard wrist projections, is never held in memory
    pre = []
    for s in tqdm(range(0, n, 128), desc='Duplicate search'):
        d = _popcount(H[s:s + 128, None, 0:1] ^ H[None, :, :]).min(axis=2)
        ii, jj = np.nonzero(d <= phash_max)
        keep = jj > ii + s
        ii, jj = ii[keep] + s, jj[keep]
        for c in range(0, len(ii), 8192):
            a, b = ii[c:c + 8192], jj[c:c + 8192]
            ncc32 = np.einsum('evk,ek->ev', V32[a], flat32[b]).max(axis=1) / flat32.shape[1]
            ok32 = ncc32 > ncc_threshold - 0.04
            pre.append(np.stack([a[ok32], b[ok32]], 1))
    pre = np.concatenate(pre) if pre else np.zeros((0, 2), int)
    del V32

    def best_ncc(small: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        res = np.empty(len(pairs), dtype=np.float32)
        k = small.shape[1] * small.shape[2]
        flat = small.reshape(len(small), -1)
        # keep each [chunk, 16, k] float32 block near 256 MB (a fixed 4096 needed ~2.4 GB at 96 px)
        chunk = max(1, (256 << 20) // (16 * k * 4))
        for s in range(0, len(pairs), chunk):
            p = pairs[s:s + chunk]
            V = np.stack([np.stack(_variants(small[i])).reshape(16, -1) for i in p[:, 0]])
            V = (V - V.mean(axis=2, keepdims=True)) / (V.std(axis=2, keepdims=True) + 1e-6)
            res[s:s + chunk] = np.einsum('evk,ek->ev', V, flat[p[:, 1]]).max(axis=1) / k
        return res

    ncc = best_ncc(T, pre) if len(pre) else np.zeros(0)
    ok = ncc >= ncc_threshold
    return [(int(i), int(j), float(c)) for (i, j), c in zip(pre[ok], ncc[ok])]


def split_group_key(source: str, label: int, name: str, path: str, patient_id) -> str:
    """
    Key of images that must share a split besides duplicates: all images of one
    GRAZPEDWRI-DX patient, and Dataset A augmentation families
    ('12-rotated1-rotated3 - Copy (1).jpg' all derive from '12').
    """
    if source == 'graz':
        return f'graz|{int(patient_id)}'
    if source != 'A':
        return path
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'( - Copy)+|( \(\d+\))+', '', stem)
    stem = re.split(r'-rotated', stem)[0].strip()
    return f'A|{label}|{stem}'


def clean_and_split(df: pd.DataFrame, thumbs: np.ndarray,
                    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                    seed: int = 42, ncc_threshold: float = 0.97,
                    fixed_splits: Optional[Dict[str, str]] = None,
                    fixed_sources: Sequence[str] = (),
                    verbose: bool = True) -> pd.DataFrame:
    """
    Deduplicate, drop label conflicts and assign a leakage-free train/val/test split.

    Args:
        fixed_splits: optional {'<source>|<file name>': split} for images whose split is
            predetermined (e.g. an earlier run's assignment). A split group containing such
            an image takes its split (test beats val beats train if they disagree).
        fixed_sources: sources fully governed by fixed_splits; their images without a fixed
            split (e.g. files that were duplicates in the earlier run) go to train, so the
            earlier val/test sets stay exactly the same.

    Returns the cleaned manifest (one row per distinct radiograph) with columns
    'dup_cluster', 'split_group', 'n_copies' and 'split'.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    pairs = find_duplicate_pairs(df, thumbs, ncc_threshold=ncc_threshold)

    dup = _UnionFind(n)
    for _, idx in df.groupby('md5').indices.items():
        for k in idx[1:]:
            dup.union(idx[0], k)
    for i, j, _ in pairs:
        dup.union(i, j)
    df['dup_cluster'] = dup.labels()

    grp = _UnionFind(n)
    grp.parent = df['dup_cluster'].values.copy()
    pids = df['patient_id'] if 'patient_id' in df else pd.Series(np.nan, index=df.index)
    df['group_key'] = [split_group_key(s, l, nm, p, pid) for s, l, nm, p, pid in
                       zip(df.source, df.label, df.name, df.path, pids)]
    for _, idx in df.groupby('group_key').indices.items():
        for k in idx[1:]:
            grp.union(idx[0], k)
    df['split_group'] = grp.labels()

    n_labels = df.groupby('dup_cluster')['label'].transform('nunique')
    conflicted = df[n_labels > 1]
    df['n_copies'] = df.groupby('dup_cluster')['path'].transform('size')

    # canonical copy: preferred source, not a pre-augmented 'rotated' file, largest image
    df['_prio'] = df['source'].map(SOURCE_PRIORITY)
    df['_rot'] = df['name'].str.lower().str.contains('rotated').astype(int)
    df['_area'] = -(df['w'] * df['h'])
    clean = (df[n_labels == 1]
             .sort_values(['dup_cluster', '_prio', '_rot', '_area', 'path'])
             .drop_duplicates('dup_cluster')
             .drop(columns=['_prio', '_rot', '_area'])
             .reset_index(drop=True))

    # predetermined splits first
    clean['split'] = None
    n_fixed = n_orphan = 0
    if fixed_splits:
        rank = {'test': 0, 'val': 1, 'train': 2}
        fixed = (clean['source'].astype(str) + '|' + clean['name'].astype(str)).map(fixed_splits).map(rank)
        group_rank = fixed.groupby(clean['split_group']).transform('min')
        has = group_rank.notna()
        clean.loc[has, 'split'] = group_rank[has].map({v: k for k, v in rank.items()})
        orphan = ~has & clean['source'].isin(list(fixed_sources))
        clean.loc[orphan, 'split'] = 'train'
        n_fixed, n_orphan = int(fixed.notna().sum()), int(orphan.sum())

    # stratified group split of the rest: 20 folds -> test / val / train
    free = clean['split'].isna()
    if free.any():
        n_folds = 20
        n_test = int(round(ratios[2] * n_folds))
        n_val = int(round(ratios[1] * n_folds))
        rest = clean[free]
        strata = rest['source'].astype(str) + '_' + rest['label'].astype(str)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold = np.empty(len(rest), dtype=int)
        for f, (_, te) in enumerate(sgkf.split(rest, strata, groups=rest['split_group'])):
            fold[te] = f
        clean.loc[free, 'split'] = np.where(fold < n_test, 'test', np.where(fold < n_test + n_val, 'val', 'train'))

    if verbose:
        print("=" * 70)
        print("Data cleaning summary")
        print("=" * 70)
        for s, sub in df.groupby('source'):
            print(f"  {s:10s}: {len(sub):6d} files -> {sub['dup_cluster'].nunique():5d} distinct radiographs")
        print(f"  near-duplicate pairs found: {len(pairs)} | exact-duplicate files: {int(df.duplicated('md5').sum())}")
        print(f"  label-conflicting radiographs dropped: {conflicted['dup_cluster'].nunique()} "
              f"({len(conflicted)} files)")
        cross = df.groupby('dup_cluster')['source'].nunique().gt(1).sum()
        print(f"  duplicate clusters spanning more than one source: {cross}")
        if fixed_splits:
            print(f"  fixed splits matched: {n_fixed} | unmatched images of fixed sources sent to train: {n_orphan}")
        print(f"  kept: {len(clean)} distinct radiographs in {clean['split_group'].nunique()} split groups")
        tab = clean.groupby(['split', 'source', 'label']).size().unstack(fill_value=0)
        tab.columns = ['not fractured', 'fractured']
        print(tab.to_string())
        leaks = set(clean.loc[clean.split == 'train', 'split_group']) & \
            set(clean.loc[clean.split != 'train', 'split_group'])
        print(f"  split groups shared between train and val/test: {len(leaks)} (must be 0)")
        if 'patient_id' in clean and clean['patient_id'].notna().any():
            has_pid = clean['patient_id'].notna()
            p_leaks = set(clean.loc[has_pid & (clean.split == 'train'), 'patient_id']) & \
                set(clean.loc[has_pid & (clean.split != 'train'), 'patient_id'])
            print(f"  patients shared between train and val/test: {len(p_leaks)} (must be 0)")
        print("=" * 70)
    return clean


# ============================================================================
# Image loading and transforms
# ============================================================================

def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert any PIL image (16-bit, palette, RGBA, RGB) to 8-bit grayscale 'L'."""
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return Image.fromarray(a.astype(np.uint8), 'L')
    if img.mode in ('P', 'LA', 'RGBA', 'PA'):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert('L')


def load_xray(img: Image.Image) -> Image.Image:
    """Normalise an input image to the 3-channel grayscale the network expects."""
    return to_grayscale(img).convert('RGB')


def _cache_one(job: Tuple[str, str, int]) -> None:
    src, dst, max_side = job
    if os.path.exists(dst):
        return
    img = Image.open(src)
    if img.format == 'JPEG':
        img.draft('L', (max_side, max_side))
    g = to_grayscale(img)
    g.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)  # keeps aspect ratio, only shrinks
    g.save(dst)


def cache_resized(manifest: pd.DataFrame, cache_dir: str, max_side: int = 640,
                  num_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Decode every image once (including 16-bit windowing) and store an 8-bit grayscale PNG
    whose longest side is at most max_side. Returns a copy of the manifest whose 'path'
    points at the cached files ('orig_path' keeps the original). Keep max_side somewhat
    above the training size so random crops are not upsampled.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = manifest.copy()
    out['orig_path'] = out['path']
    out['path'] = [os.path.join(cache_dir, hashlib.md5(p.encode()).hexdigest()[:20] + '.png')
                   for p in out['orig_path']]
    jobs = list(zip(out['orig_path'], out['path'], [max_side] * len(out)))
    workers = num_workers or os.cpu_count() or 1
    if workers > 1:
        with Pool(workers) as pool:
            list(tqdm(pool.imap(_cache_one, jobs, chunksize=16), total=len(jobs), desc='Caching'))
    else:
        for j in tqdm(jobs, desc='Caching'):
            _cache_one(j)
    return out


class PadToSquare:
    """Pad the shorter side with black so the whole radiograph survives resizing."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if w == h:
            return img
        left, top = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(left, top, s - w - left, s - h - top), fill=self.fill)

    def __repr__(self):
        return f'{self.__class__.__name__}(fill={self.fill})'


def get_transforms(phase: str, img_size: int = 384) -> transforms.Compose:
    """
    Image transforms for 'train', 'val' or 'test'.

    Every phase pads to a square and resizes the full image (no centre crop).
    Training adds mild geometric and intensity augmentation that keeps the
    radiograph realistic (no vertical flips, no random erasing that could hide
    the fracture itself).
    """
    if phase not in ['train', 'val', 'test']:
        raise ValueError(f"phase must be 'train', 'val', or 'test', got {phase}")

    if phase == 'train':
        return transforms.Compose([
            PadToSquare(),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class XrayDataset(Dataset):
    """Dataset over manifest rows (path, label)."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform=None, draft_size: int = 1024):
        self.paths = list(paths)
        self.labels = [int(l) for l in labels]
        self.transform = transform
        self.draft_size = draft_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx])
        if img.format == 'JPEG':
            img.draft('L', (self.draft_size, self.draft_size))  # fast reduced-size JPEG decode
        img = load_xray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def source_balanced_weights(sources: Sequence[str], labels: Sequence[int],
                            source_power: float = 1.0, class_balance: float = 1.0) -> np.ndarray:
    """
    Per-sample sampling weights.

    Sources have very different fracture rates (FracAtlas ~17%, GRAZPEDWRI-DX ~58%), so
    classes are balanced *within each source*: balancing globally would reward recognising
    the dataset instead of the fracture.

    Args:
        source_power: a source's share of draws is proportional to n_source ** source_power
            (1 = natural shares, 0 = equal shares)
        class_balance: within a source, class shares are proportional to
            n_class ** (1 - class_balance) (1 = balanced classes, 0 = natural ratio).
            Values below 1 repeat the rare class less often, which overfits less.
    """
    df = pd.DataFrame({'s': list(sources), 'y': list(labels)})
    src_n = df['s'].value_counts().astype(float)
    src_share = src_n ** source_power
    src_share /= src_share.sum()
    cell_n = df.groupby(['s', 'y']).size().astype(float)
    cls_share = cell_n ** (1.0 - class_balance)
    cls_share /= cls_share.groupby(level=0).transform('sum')
    w = np.array([src_share[s] * cls_share[(s, y)] / cell_n[(s, y)] for s, y in zip(df['s'], df['y'])])
    return w / w.sum()


def get_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Inverse-frequency class weights (shape [2])."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    return torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)


def get_dataloaders(manifest: pd.DataFrame,
                    img_size: int = 384,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    balance: str = 'source_class',
                    source_power: float = 1.0,
                    class_balance: float = 1.0) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from a cleaned manifest (output of clean_and_split).

    Args:
        balance: 'source_class' (per-source class balance, default), 'class', or 'none'
        source_power, class_balance: see source_balanced_weights
    """
    loaders = {}
    for phase in ['train', 'val', 'test']:
        sub = manifest[manifest['split'] == phase]
        ds = XrayDataset(sub['path'].values, sub['label'].values, transform=get_transforms(phase, img_size))
        if phase == 'train' and balance != 'none':
            if balance == 'source_class':
                w = source_balanced_weights(sub['source'].values, sub['label'].values,
                                            source_power=source_power, class_balance=class_balance)
            else:
                w = source_balanced_weights(['all'] * len(sub), sub['label'].values,
                                            class_balance=class_balance)
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(sub),
                                            replacement=True)
            loaders[phase] = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
        else:
            loaders[phase] = DataLoader(ds, batch_size=batch_size, shuffle=(phase == 'train'),
                                        num_workers=num_workers, pin_memory=True,
                                        persistent_workers=num_workers > 0)
    return loaders


def show_batch(dataloader: DataLoader, n: int = 8) -> None:
    """
    Visualize a batch of images with class labels.

    Args:
        dataloader (DataLoader): DataLoader to visualize from
        n (int): Number of images to display
    """
    images, labels = next(iter(dataloader))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    images = torch.clamp(images * std + mean, 0, 1)

    n = min(n, len(images))
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), squeeze=False)
    class_labels = ['Not Fractured', 'Fractured']
    for ax, idx in zip(axes.flat, range(n)):
        ax.imshow(images[idx].permute(1, 2, 0).numpy())
        ax.set_title(class_labels[int(labels[idx])], fontsize=12, fontweight='bold')
        ax.axis('off')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
