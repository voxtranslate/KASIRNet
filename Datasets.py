# ==========================================================================
# KASIRNet - data: sources, pairing, leakage audit, training pool, test sets
# --------------------------------------------------------------------------
# * Pairs are matched by an explicit key (file stem or a regex), never by the
#   position in two independently sorted lists (sorted-order zipping mis-paired
#   all 1800 RainTrainH pairs, 'norain-10.png' <-> 'norain-1001x2.png').
# * A training pair whose clean image is a near-duplicate of a benchmark clean
#   image is removed (LeakageAuditor); the audit is written to data_manifest/.
# * The training pool (seeded subset, held-out validation split) is computed once
#   and frozen in data_manifest/train_manifest.json so that every run of the
#   revision sees exactly the same images.
# ==========================================================================
import os
import io
import re
import glob
import json
import zlib
import random
import hashlib
import tarfile
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ==========================================
# 1. Sources (kagglehub / Google Drive archives)
# ==========================================

def _extract(archive, dest):
    if archive.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def resolve_sources(cfg, keys):
    """Download (once) and return {source key: local root directory}."""
    roots = {}
    for key in sorted(set(keys)):
        if key in cfg.KAGGLE_SOURCES:
            import kagglehub
            roots[key] = kagglehub.dataset_download(cfg.KAGGLE_SOURCES[key])
        elif key in cfg.GDOWN_SOURCES:
            src = cfg.GDOWN_SOURCES[key]
            dest = os.path.join(cfg.DATA_ROOT, key)
            done = os.path.join(dest, '.extracted')
            if not os.path.exists(done):
                import gdown
                os.makedirs(dest, exist_ok=True)
                archive = os.path.join(cfg.DATA_ROOT, src['file'])
                if not os.path.exists(archive):
                    gdown.download(id=src['id'], output=archive, quiet=False)
                print(f'[data] extracting {archive} ...')
                _extract(archive, dest)
                open(done, 'w').write('ok')
                os.remove(archive)
            roots[key] = dest
        else:
            raise KeyError(f'Unknown data source {key}')
    return roots


def spec_sources(spec, benchmarks=None):
    keys = []
    for f in ('input', 'gt', 'root', 'clean', 'lmdb', 'lmdb_gt'):
        if f in spec:
            keys.append(spec[f][0])
    if 'base' in spec and benchmarks is not None:
        keys += spec_sources(benchmarks[spec['base']], benchmarks)
    return keys


def resolve_path(roots, ref):
    """ref = (source key, relative path); a relative path containing '*' is searched
    recursively and the first (sorted) matching directory is returned."""
    key, rel = ref
    if key not in roots:
        raise FileNotFoundError(f'data source {key!r} is not available (download failed or not requested)')
    root = roots[key]
    if '*' in rel:
        hits = sorted(h for h in glob.glob(os.path.join(root, rel), recursive=True) if os.path.isdir(h))
        hits = [h for h in hits if '__MACOSX' not in h]
        if not hits:
            raise FileNotFoundError(f'No directory matches {rel} under {root}')
        return hits[0]
    return os.path.normpath(os.path.join(root, rel))


def list_images(d, recursive=False):
    if recursive:
        files = glob.glob(os.path.join(d, '**', '*'), recursive=True)
    else:
        files = glob.glob(os.path.join(d, '*'))
    return sorted(f for f in files if f.lower().endswith(IMG_EXT) and os.path.isfile(f))


# ==========================================
# 2. Pairing
# ==========================================

def _key(path, rx):
    stem = os.path.splitext(os.path.basename(path))[0]
    if rx is None:
        return stem.lower()
    m = re.search(rx, stem)
    return None if m is None else '_'.join(m.groups()).lower()


def pair_files(input_dir, gt_dir, match='stem', include=None):
    """Return (pairs, report).  pairs = [(input_path, gt_path, key)]."""
    rx_in = rx_gt = None
    if isinstance(match, (tuple, list)) and match[0] == 'regex':
        rx_in, rx_gt = match[1], match[2]
    ins, gts = list_images(input_dir), list_images(gt_dir)
    gt_map = {}
    for g in gts:
        k = _key(g, rx_gt)
        if k is not None:
            gt_map.setdefault(k, g)
    pairs, unmatched = [], []
    for f in ins:
        k = _key(f, rx_in)
        if k is None:
            continue
        if include and not re.search(include, os.path.basename(f)):
            continue
        if k in gt_map and os.path.abspath(gt_map[k]) != os.path.abspath(f):
            pairs.append((f, gt_map[k], k))
        else:
            unmatched.append(os.path.basename(f))
    report = dict(input_dir=input_dir, gt_dir=gt_dir, n_input_files=len(ins), n_gt_files=len(gts),
                  n_pairs=len(pairs), n_unmatched=len(unmatched), unmatched_examples=unmatched[:5])
    return pairs, report


def pair_by_replace(root, input_glob, gt_replace):
    old, new = gt_replace
    ins = sorted(glob.glob(os.path.join(root, input_glob), recursive=True))
    pairs, missing = [], 0
    for f in ins:
        g = os.path.join(os.path.dirname(f), os.path.basename(f).replace(old, new))
        if os.path.exists(g):
            pairs.append((f, g, os.path.splitext(os.path.basename(f))[0]))
        else:
            missing += 1
    return pairs, dict(root=root, n_input_files=len(ins), n_pairs=len(pairs), n_unmatched=missing)


def spec_pairs(spec, roots):
    if 'root' in spec:
        return pair_by_replace(resolve_path(roots, spec['root']), spec['input_glob'], spec['gt_replace'])
    return pair_files(resolve_path(roots, spec['input']), resolve_path(roots, spec['gt']),
                      spec.get('match', 'stem'), spec.get('include'))


# ==========================================
# 3. Leakage audit (near-duplicate clean images between train and test)
# ==========================================

class LeakageAuditor:
    """24x24 grey thumbnails, zero-mean / unit-norm; a training clean image whose
    correlation with any benchmark clean image exceeds `threshold` is a duplicate.
    (On RainTrainH vs Rain100H this finds exactly the 546 duplicated backgrounds
    reported by the PReNet authors.)"""

    def __init__(self, cache_dir, threshold=0.97, size=24, workers=8):
        self.cache_dir, self.threshold, self.size, self.workers = cache_dir, threshold, size, workers
        os.makedirs(cache_dir, exist_ok=True)

    def _thumb(self, f):
        im = Image.open(f).convert('L').resize((self.size, self.size), Image.BILINEAR)
        v = np.asarray(im, np.float32).ravel()
        v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-6)

    def thumbs(self, files, tag):
        h = hashlib.md5('\n'.join(files).encode()).hexdigest()[:12]
        cache = os.path.join(self.cache_dir, f'thumbs_{tag}_{h}.npy')
        if os.path.exists(cache):
            return np.load(cache)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, self.workers)) as ex:
            X = np.stack(list(ex.map(self._thumb, files)))
        np.save(cache, X)
        return X

    def duplicates(self, train_gts, test_gts, tag_train, tag_test):
        A, B = self.thumbs(train_gts, tag_train), self.thumbs(test_gts, tag_test)
        best = np.zeros(len(A), np.float32)
        for i in range(0, len(A), 4096):
            best[i:i + 4096] = (A[i:i + 4096] @ B.T).max(1)
        return best > self.threshold, best


# ==========================================
# 4. Training pool manifest
# ==========================================

def benchmark_clean_files(cfg, roots, name):
    spec = cfg.benchmarks()[name]
    if 'clean' in spec:
        return list_images(resolve_path(roots, spec['clean']))
    pairs, _ = spec_pairs(spec, roots)
    return [g for _, g, _ in pairs]


def build_training_manifest(cfg, roots, force=False):
    """Pairs of every training category -> leakage filtering -> seeded cap ->
    held-out validation split.  Written once to data_manifest/train_manifest.json."""
    os.makedirs(cfg.manifest_dir, exist_ok=True)
    path = os.path.join(cfg.manifest_dir, 'train_manifest.json')
    cats = cfg.train_categories()
    sig = hashlib.md5(json.dumps([cats, cfg.MAX_PER_CATEGORY, cfg.VAL_PER_CATEGORY, cfg.DATA_SUBSET_SEED,
                                  cfg.EXCLUDE_TEST_OVERLAP, cfg.OVERLAP_THRESHOLD, cfg.KOHLER_PROTOCOL],
                                 sort_keys=True, default=str).encode()).hexdigest()
    if os.path.exists(path) and not force:
        man = json.load(open(path))
        if man.get('signature') == sig:
            return man
        print('[data] configuration changed -> rebuilding the training manifest')
    auditor = LeakageAuditor(os.path.join(cfg.manifest_dir, 'thumb_cache'), cfg.OVERLAP_THRESHOLD, workers=cfg.NUM_WORKERS)
    rng = random.Random(cfg.DATA_SUBSET_SEED)
    man = dict(signature=sig, categories={}, train=[], val=[], audit={})
    bench_cache = {}
    for ci, (name, spec) in enumerate(cats.items()):
        try:
            pairs, rep = spec_pairs(spec, roots)
        except FileNotFoundError as e:
            print(f'[data] !! category {name} unavailable: {e}')
            man['categories'][name] = dict(available=False, error=str(e))
            continue
        removed = []
        if cfg.EXCLUDE_TEST_OVERLAP and spec.get('exclude_overlap_with') and pairs:
            dup = np.zeros(len(pairs), bool)
            for b in spec['exclude_overlap_with']:
                if b not in bench_cache:
                    try:
                        bench_cache[b] = benchmark_clean_files(cfg, roots, b)
                    except (FileNotFoundError, KeyError) as e:
                        print(f'[data] overlap check {name} vs {b} skipped: {e}')
                        bench_cache[b] = []
                if bench_cache[b]:
                    d, score = auditor.duplicates([p[1] for p in pairs], bench_cache[b], name, b)
                    man['audit'][f'{name}_vs_{b}'] = dict(n_train=len(pairs), n_test=len(bench_cache[b]), n_duplicates=int(d.sum()))
                    dup |= d
            removed = [os.path.basename(p[1]) for p, d in zip(pairs, dup) if d]
            pairs = [p for p, d in zip(pairs, dup) if not d]
        pairs = sorted(pairs)
        rng.shuffle(pairs)
        n_val = min(cfg.VAL_PER_CATEGORY, max(1, len(pairs) // 10)) if len(pairs) >= 5 else 0
        val, train = pairs[:n_val], pairs[n_val:]
        train = sorted(train)[:] if len(train) <= cfg.MAX_PER_CATEGORY else sorted(rng.sample(train, cfg.MAX_PER_CATEGORY))
        rep = dict(rep, available=True, family=spec['family'], index=ci, n_removed_overlap=len(removed),
                   removed_examples=removed[:10], n_train=len(train), n_val=len(val), repeat=spec.get('repeat', 1))
        man['categories'][name] = rep
        man['train'] += [dict(input=a, gt=b, category=name, family=spec['family'], repeat=spec.get('repeat', 1)) for a, b, _ in train]
        man['val'] += [dict(input=a, gt=b, category=name, family=spec['family']) for a, b, _ in val]
        print(f'[data] {name:11s} pairs={rep["n_pairs"]:6d}  overlap-removed={len(removed):5d}  train={len(train):5d}  val={len(val):3d}  repeat={spec.get("repeat", 1)}')
    with open(path + '.tmp', 'w') as f:   # atomic: several sessions may build it at the same time
        json.dump(man, f, indent=1)
    os.replace(path + '.tmp', path)
    return man


# ==========================================
# 5. Synthetic degradations (test-time only, deterministic per image)
# ==========================================

def image_seed(base, name):
    return (base + zlib.crc32(os.path.basename(name).encode())) % (2 ** 31)


def degrade(img_t, ops, seed, haze_depth=(0.5, 1.0)):
    """img_t: (3,H,W) float in [0,1].  ops: [('gaussian', sigma_255) | ('jpeg', q) | ('haze', beta, A)]."""
    x = img_t.clone()
    g = torch.Generator().manual_seed(seed)
    for op in ops:
        if op[0] == 'gaussian':
            x = (x + torch.randn(x.shape, generator=g) * (op[1] / 255.0)).clamp(0, 1)
        elif op[0] == 'jpeg':
            buf = io.BytesIO()
            TF.to_pil_image(x.clamp(0, 1)).save(buf, format='JPEG', quality=int(op[1]))
            x = TF.to_tensor(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
        elif op[0] == 'haze':  # atmospheric scattering model I = J t + A (1 - t), t = exp(-beta d)
            beta, A = op[1], op[2]
            H = x.shape[1]
            far, near = haze_depth[1], haze_depth[0]
            d = torch.linspace(far, near, H).view(1, H, 1)
            t = torch.exp(-beta * d)
            x = (x * t + A * (1 - t)).clamp(0, 1)
        else:
            raise ValueError(op)
    return x


# ==========================================
# 6. Datasets
# ==========================================

def load_rgb(path):
    return Image.open(path).convert('RGB')


class UniversalRestorationDataset(Dataset):
    """Training / validation patches.  Train: random patch_size crop + horizontal and
    vertical flips (images smaller than the patch are first resized up, as in the
    submitted code).  Validation: centre crop."""

    def __init__(self, items, patch_size=256, is_train=True, families=None):
        self.patch_size, self.is_train = patch_size, is_train
        self.samples = []
        for it in items:
            self.samples += [it] * (int(it.get('repeat', 1)) if is_train else 1)
        cats = sorted({it['category'] for it in items})
        self.task_to_id = {c: i for i, c in enumerate(cats)}
        self.families = families or sorted({it['family'] for it in items})
        self.family_to_id = {f: i for i, f in enumerate(self.families)}

    def __len__(self):
        return len(self.samples)

    def transform(self, image, gt):
        if gt.size != image.size:
            gt = gt.resize(image.size, Image.BICUBIC)
        w, h = image.size
        p = self.patch_size
        if w < p or h < p:
            new_w, new_h = max(w, p), max(h, p)
            image = TF.resize(image, [new_h, new_w])
            gt = TF.resize(gt, [new_h, new_w])
            w, h = image.size
        if self.is_train:
            left, top = random.randint(0, w - p), random.randint(0, h - p)
            image = image.crop((left, top, left + p, top + p))
            gt = gt.crop((left, top, left + p, top + p))
            if random.random() > 0.5:
                image, gt = TF.hflip(image), TF.hflip(gt)
            if random.random() > 0.5:
                image, gt = TF.vflip(image), TF.vflip(gt)
        else:
            left, top = (w - p) // 2, (h - p) // 2
            image = image.crop((left, top, left + p, top + p))
            gt = gt.crop((left, top, left + p, top + p))
        return TF.to_tensor(image), TF.to_tensor(gt)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inp, gt = self.transform(load_rgb(item['input']), load_rgb(item['gt']))
        return {'input': inp, 'gt': gt, 'task_id': self.task_to_id[item['category']],
                'family_id': self.family_to_id[item['family']]}


class LMDBPairReader:
    """BasicSR-style lmdb (SIDD validation crops): keys from meta_info.txt, PNG bytes."""

    def __init__(self, lq_path, gt_path):
        self.lq_path, self.gt_path = lq_path, gt_path
        with open(os.path.join(lq_path, 'meta_info.txt')) as f:
            self.keys = [ln.split(' ')[0].rsplit('.', 1)[0] for ln in f if ln.strip()]
        self._env = None

    def _open(self):
        import lmdb
        self._env = (lmdb.open(self.lq_path, readonly=True, lock=False, readahead=False),
                     lmdb.open(self.gt_path, readonly=True, lock=False, readahead=False))

    def get(self, key):
        if self._env is None:
            self._open()
        out = []
        for env in self._env:
            with env.begin(write=False) as txn:
                out.append(Image.open(io.BytesIO(txn.get(key.encode('ascii')))).convert('RGB'))
        return out


class BenchmarkDataset(Dataset):
    """Full-resolution test set.  Items: name, input, gt, and for mixed sets the
    singly-degraded input ('first') used by the prompt-override control."""

    def __init__(self, cfg, roots, name, max_images=None):
        self.cfg, self.name = cfg, name
        bench = cfg.benchmarks()
        self.spec = spec = bench[name]
        self.family = spec['family']
        self.lmdb = None
        self.base = None
        self.report = {}
        if 'lmdb' in spec:
            self.lmdb = LMDBPairReader(resolve_path(roots, spec['lmdb']), resolve_path(roots, spec['lmdb_gt']))
            self.items = [(k, None, k) for k in self.lmdb.keys]
            self.report = dict(n_pairs=len(self.items), source='lmdb')
        elif 'clean' in spec:
            files = list_images(resolve_path(roots, spec['clean']))
            self.items = [(f, f, os.path.splitext(os.path.basename(f))[0]) for f in files]
            self.report = dict(n_pairs=len(self.items), source='synthetic', degrade=spec['degrade'])
        elif 'base' in spec:
            self.base = BenchmarkDataset(cfg, roots, spec['base'])
            self.items = list(range(len(self.base)))
            self.report = dict(n_pairs=len(self.items), source='mixed', base=spec['base'], degrade=spec['degrade'])
        else:
            pairs, self.report = spec_pairs(spec, roots)
            self.items = [(a, b, k) for a, b, k in pairs]
        limit = max_images or spec.get('max_images')
        if limit and len(self.items) > limit:
            idx = sorted(random.Random(cfg.DATA_SUBSET_SEED).sample(range(len(self.items)), limit))
            self.items = [self.items[i] for i in idx]
            self.report['subset'] = limit

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        spec = self.spec
        if self.base is not None:
            b = self.base[self.items[idx]]
            x = degrade(b['input'], spec['degrade'], image_seed(self.cfg.SYNTHETIC_SEED, b['name']), self.cfg.HAZE_DEPTH)
            return {'name': b['name'], 'key': b['key'], 'gt_name': b['gt_name'], 'input': x, 'gt': b['gt'],
                    'first': b['input'], 'gt_resized': b['gt_resized']}
        a, g, k = self.items[idx]
        if self.lmdb is not None:
            inp, gt = self.lmdb.get(a)
            inp, gt = TF.to_tensor(inp), TF.to_tensor(gt)
        elif 'clean' in spec:
            gt = TF.to_tensor(load_rgb(g))
            inp = degrade(gt, spec['degrade'], image_seed(self.cfg.SYNTHETIC_SEED, g), self.cfg.HAZE_DEPTH)
        else:
            inp, gt = TF.to_tensor(load_rgb(a)), TF.to_tensor(load_rgb(g))
        resized = False
        if gt.shape[1:] != inp.shape[1:]:  # same rule as the submitted inference code; counted in the report
            gt = TF.resize(gt, list(inp.shape[1:]), antialias=True).clamp(0, 1)
            resized = True
        is_file = isinstance(a, str) and os.sep in a
        return {'name': os.path.basename(a) if is_file else str(k), 'key': str(k),
                'gt_name': os.path.basename(g) if isinstance(g, str) and os.sep in g else str(k),
                'input': inp, 'gt': gt, 'gt_resized': resized}


def collate_single(batch):
    return batch[0]
