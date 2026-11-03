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
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ==========================================
# 1. Sources: kagglehub / Hugging Face / Google Drive, and the Drive cache
# ==========================================

def _extract(archive, dest, stream=False):
    if archive.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return
    with tarfile.open(archive, 'r|*' if stream else 'r:*') as t:
        try:
            t.extractall(dest, filter='data')
        except TypeError:          # Python < 3.12
            t.extractall(dest)


def _done(dest):
    return os.path.join(dest, '.extracted')


def _download_gdown(cfg, key):
    src = cfg.GDOWN_SOURCES[key] if key in cfg.GDOWN_SOURCES else key
    dest = os.path.join(cfg.DATA_ROOT, src['file'].split('.')[0] if isinstance(key, dict) else key)
    if os.path.exists(_done(dest)):
        return dest
    os.makedirs(dest, exist_ok=True)
    manual = os.path.join(cfg.DATA_CACHE_ROOT, 'archives', src['file'])
    archive = manual if os.path.exists(manual) else os.path.join(cfg.DATA_ROOT, src['file'])
    if not os.path.exists(archive):
        import gdown
        out = gdown.download(id=src['id'], output=archive, quiet=False)
        if out is None or not os.path.exists(archive):
            raise RuntimeError(f'Google-Drive download of {src["file"]} failed (often "quota exceeded", resets within 24 h). '
                               f'Either run again later, or open https://drive.google.com/file/d/{src["id"]} in the browser, '
                               f'download it and upload it to {manual}')
    print(f'[data] extracting {archive} -> {dest}')
    _extract(archive, dest, stream=not archive.endswith('.zip'))
    open(_done(dest), 'w').write('ok')
    if archive != manual:
        os.remove(archive)
    return dest


def _download_hf_parquet(cfg, key):
    """Hugging Face parquet mirror -> image files <dir>/<stem of key column>.<ext> (same stem in every column)."""
    spec = cfg.HF_PARQUET_SOURCES[key]
    dest = os.path.join(cfg.DATA_ROOT, key)
    if os.path.exists(_done(dest)):
        return dest
    import urllib.request
    import pyarrow.parquet as pq
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in spec['files']:
        local = os.path.join(cfg.DATA_ROOT, key + '_' + os.path.basename(f))
        if not os.path.exists(local):
            url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/main/{f}'
            print(f'[data] downloading {url}')
            urllib.request.urlretrieve(url, local + '.part')
            os.replace(local + '.part', local)
        table = pq.read_table(local)
        for row in table.to_pylist():
            stem = os.path.splitext(os.path.basename(row[spec['key_column']]['path']))[0]
            for col, sub in spec['columns'].items():
                ext = os.path.splitext(row[col]['path'])[1] or '.png'
                out = os.path.join(dest, sub, stem + ext)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, 'wb') as fh:
                    fh.write(row[col]['bytes'])
            n += 1
        os.remove(local)
    print(f'[data] {key}: {n} pairs written to {dest}')
    open(_done(dest), 'w').write('ok')
    return dest


def _download_hf_zip(cfg, key):
    spec = cfg.HF_ZIP_SOURCES[key]
    dest = os.path.join(cfg.DATA_ROOT, key)
    if os.path.exists(_done(dest)):
        return dest
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    local = os.path.join(cfg.DATA_ROOT, key + '_' + spec['file'])
    url = f'https://huggingface.co/datasets/{spec["repo"]}/resolve/main/{spec["file"]}'
    print(f'[data] downloading {url}')
    urllib.request.urlretrieve(url, local + '.part')
    os.replace(local + '.part', local)
    _extract(local, dest)
    os.remove(local)
    open(_done(dest), 'w').write('ok')
    return dest


def download_source(cfg, key):
    if key in cfg.HF_ZIP_SOURCES:
        return _download_hf_zip(cfg, key)
    if key in cfg.KAGGLE_SOURCES:
        import kagglehub
        return kagglehub.dataset_download(cfg.KAGGLE_SOURCES[key])
    if key in cfg.GDOWN_SOURCES:
        return _download_gdown(cfg, key)
    if key in cfg.HF_PARQUET_SOURCES:
        try:
            return _download_hf_parquet(cfg, key)
        except Exception as e:
            if key == 'dpdd_test':   # official archive (zip with test/inputC, test/target)
                print(f'[data] Hugging Face mirror failed ({e}); trying the official Google-Drive archive')
                return _download_gdown(cfg, dict(id=cfg.DPDD_GDOWN_ID, file='dpdd_test.zip'))
            raise
    raise KeyError(f'Unknown data source {key}')


class DatasetCache:
    """One uncompressed tar per source on Google Drive (DATA_CACHE_ROOT/<key>.tar + <key>.json).
    'used' tars hold only the files read by the experiments; they are never used to rebuild the
    training manifest (need_full=True), which requires the complete source."""

    def __init__(self, cfg):
        self.cfg, self.mode, self.root = cfg, cfg.DATA_CACHE_MODE, cfg.DATA_CACHE_ROOT
        self.local = os.path.join(cfg.DATA_ROOT, 'from_drive_cache')

    def _tar(self, key):
        return os.path.join(self.root, f'{key}.tar')

    def info(self, key):
        p = os.path.join(self.root, f'{key}.json')
        if os.path.exists(p) and os.path.exists(self._tar(key)):
            return json.load(open(p))
        return None

    def restore(self, key, need_full=False):
        if self.mode == 'off' or self.cfg.DATA_CACHE_REFRESH:
            return None
        m = self.info(key)
        if m is None or (need_full and m['mode'] != 'full'):
            return None
        if os.path.getsize(self._tar(key)) < m['bytes']:
            print(f'[cache] !! {key}: the tar on Drive is incomplete (upload interrupted or Drive full) -> downloaded again')
            return None
        dest = os.path.join(self.local, key)
        marker = os.path.join(dest, '.cache_meta.json')
        if os.path.exists(marker) and json.load(open(marker)).get('created') == m['created']:
            return dest
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        print(f'[cache] restoring {key} ({m["n_files"]} files, {m["bytes"] / 1e9:.2f} GB, mode {m["mode"]}) from Google Drive')
        _extract(self._tar(key), dest, stream=True)
        json.dump(m, open(marker, 'w'))
        return dest

    def wanted(self, key):
        if self.mode == 'off':
            return False
        if getattr(self.cfg, 'DATA_CACHE_SOURCES', 'fragile') == 'all':
            return True
        return key in self.cfg.GDOWN_SOURCES or key in self.cfg.HF_PARQUET_SOURCES or key in self.cfg.HF_ZIP_SOURCES

    def inventory(self, keys):
        """Print what the Drive cache holds (helps to see a cache lost to a full Drive)."""
        import shutil
        try:
            free = shutil.disk_usage(self.root if os.path.isdir(self.root) else os.path.dirname(self.root)).free / 1e9
        except OSError:
            free = float('nan')
        print(f'[cache] {self.root}  (free space reported for this Drive: {free:.1f} GB)')
        for k in sorted(set(keys)):
            m, t = self.info(k), self._tar(k)
            state = (f'cached, {m["mode"]}, {m["n_files"]} files, {m["bytes"] / 1e9:.2f} GB' if m else
                     ('tar without metadata (incomplete upload?)' if os.path.exists(t) else
                      ('not cached' if self.wanted(k) else 'not cached (Kaggle source, downloaded each session)')))
            if m and os.path.getsize(t) < m['bytes']:
                state += f'  !! tar is only {os.path.getsize(t) / 1e9:.2f} GB: incomplete, will be rebuilt'
            print(f'          {k:16s} {state}')

    def store(self, key, root, relfiles=None):
        """relfiles=None or mode 'full' -> the whole source.  Never overwrites an existing cache
        unless DATA_CACHE_REFRESH (a 'full' cache is never replaced by a 'used' one)."""
        if not self.wanted(key) or root.startswith(self.local):
            return
        m = self.info(key)
        if m is not None and not self.cfg.DATA_CACHE_REFRESH:
            return
        if self.mode == 'full' or relfiles is None:
            files, mode = [], 'full'
            for dp, _, fs in os.walk(root):
                files += [os.path.relpath(os.path.join(dp, f), root) for f in fs if not f.startswith('.cache_meta')]
        else:
            files, mode = list(relfiles), 'used'
        files = sorted(set(files))
        os.makedirs(self.root, exist_ok=True)
        need = sum(os.path.getsize(os.path.join(root, f)) for f in files)
        try:
            import shutil
            free = shutil.disk_usage(self.root).free
            if free < need * 1.05:
                print(f'[cache] !! {key}: needs {need / 1e9:.2f} GB but the Drive has {free / 1e9:.2f} GB free -> not cached')
                return
        except OSError:
            pass
        tmp = self._tar(key) + '.tmp'
        total = 0
        try:
            with tarfile.open(tmp, 'w') as t:
                for rel in files:
                    p = os.path.join(root, rel)
                    t.add(p, arcname=rel, recursive=False)
                    total += os.path.getsize(p)
            os.replace(tmp, self._tar(key))
        except OSError as e:
            print(f'[cache] !! could not cache {key} on Drive ({e}); it will be downloaded again next session')
            if os.path.exists(tmp):
                os.remove(tmp)
            return
        meta = dict(key=key, mode=mode, n_files=len(files), bytes=total, created=time.time(), source_root=root)
        with open(os.path.join(self.root, f'{key}.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        print(f'[cache] {key}: {len(files)} files ({total / 1e9:.2f} GB, {mode}) cached in {self.root}')


def resolve_sources(cfg, keys, cache=None, need_full=False):
    """{source key: local root}.  The Drive cache is used first, then the original download."""
    roots = {}
    for key in sorted(set(keys)):
        r = cache.restore(key, need_full) if cache is not None else None
        roots[key] = r or download_source(cfg, key)
    return roots


def spec_sources(spec, benchmarks=None):
    keys = []
    for f in ('input', 'gt', 'root', 'clean', 'lmdb', 'lmdb_gt'):
        if f in spec:
            keys.append(spec[f][0])
    if 'base' in spec and benchmarks is not None:
        keys += spec_sources(benchmarks[spec['base']], benchmarks)
    return keys


def resolve_path(roots, ref, must_exist=True):
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
    p = os.path.normpath(os.path.join(root, rel))
    if must_exist and not os.path.exists(p):
        raise FileNotFoundError(f'{p} does not exist (source {key!r})')
    return p


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
    """Return (pairs, report).  pairs = [(input_path, gt_path, key)].  A missing directory or an
    empty one is an error (the submitted pool silently skipped empty folders)."""
    for d in (input_dir, gt_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f'directory {d} does not exist')
    rx_in = rx_gt = None
    if isinstance(match, (tuple, list)) and match[0] == 'regex':
        rx_in, rx_gt = match[1], match[2]
    ins, gts = list_images(input_dir), list_images(gt_dir)
    gt_map, n_gt_keys = {}, 0
    for g in gts:
        k = _key(g, rx_gt)
        if k is not None:
            n_gt_keys += 1
            gt_map.setdefault(k, g)
    pairs, unmatched, n_in_keys = [], [], 0
    for f in ins:
        k = _key(f, rx_in)
        if k is None:
            continue
        if include and not re.search(include, os.path.basename(f)):
            continue
        n_in_keys += 1
        if k in gt_map and os.path.abspath(gt_map[k]) != os.path.abspath(f):
            pairs.append((f, gt_map[k], k))
        else:
            unmatched.append(os.path.basename(f))
    report = dict(input_dir=input_dir, gt_dir=gt_dir, n_input_files=len(ins), n_gt_files=len(gts),
                  n_input_keys=n_in_keys, n_gt_keys=n_gt_keys, n_pairs=len(pairs), n_unmatched=len(unmatched),
                  unmatched_examples=unmatched[:5])
    if not pairs:
        raise FileNotFoundError(f'no pairs found between {input_dir} and {gt_dir} (match={match})')
    return pairs, report


def pair_by_replace(root, input_glob, gt_replace):
    if not os.path.isdir(root):
        raise FileNotFoundError(f'directory {root} does not exist')
    old, new = gt_replace
    ins = sorted(glob.glob(os.path.join(root, input_glob), recursive=True))
    pairs, missing = [], 0
    for f in ins:
        g = os.path.join(os.path.dirname(f), os.path.basename(f).replace(old, new))
        if os.path.exists(g):
            pairs.append((f, g, os.path.splitext(os.path.basename(f))[0]))
        else:
            missing += 1
    if not pairs:
        raise FileNotFoundError(f'no {input_glob} pairs under {root}')
    return pairs, dict(root=root, n_input_files=len(ins), n_input_keys=len(ins), n_gt_keys=len(pairs),
                       n_pairs=len(pairs), n_unmatched=missing)


def spec_pairs(spec, roots):
    if 'root' in spec:
        return pair_by_replace(resolve_path(roots, spec['root']), spec['input_glob'], spec['gt_replace'])
    return pair_files(resolve_path(roots, spec['input']), resolve_path(roots, spec['gt']),
                      spec.get('match', 'stem'), spec.get('include'))


def spec_keys(spec):
    """(source of the inputs, source of the ground truths)."""
    if 'root' in spec:
        return spec['root'][0], spec['root'][0]
    return spec['input'][0], spec['gt'][0]


def benchmark_files(cfg, roots, name):
    """[(source key, relative path)] of every file a benchmark reads (for the Drive cache)."""
    spec = cfg.benchmarks()[name]
    out = []
    if 'base' in spec:
        return out
    if 'lmdb' in spec:
        for ref in (spec['lmdb'], spec['lmdb_gt']):
            d = resolve_path(roots, ref)
            out += [(ref[0], os.path.relpath(os.path.join(d, f), roots[ref[0]])) for f in os.listdir(d)]
        return out
    if 'clean' in spec:
        k = spec['clean'][0]
        return [(k, os.path.relpath(f, roots[k])) for f in list_images(resolve_path(roots, spec['clean']))]
    pairs, _ = spec_pairs(spec, roots)
    ki, kg = spec_keys(spec)
    for a, g, _ in pairs:
        out += [(ki, os.path.relpath(a, roots[ki])), (kg, os.path.relpath(g, roots[kg]))]
    return out


# ==========================================
# 3. Leakage audit (near-duplicate clean images between train and test)
# ==========================================

class LeakageAuditor:
    """Two-stage near-duplicate test on the clean images.  Screening: zero-mean / unit-norm
    24x24 grey thumbnails, correlation > `threshold`.  Confirmation of every screened pair:
    64x64 thumbnails, correlation > `confirm`, aspect ratios within 5 %.  (On RainTrainH vs
    Rain100H this finds exactly the 546 duplicated backgrounds reported by the PReNet authors.)"""

    def __init__(self, cache_dir, threshold=0.97, confirm=0.95, workers=8):
        self.cache_dir, self.threshold, self.confirm, self.workers = cache_dir, threshold, confirm, max(1, workers)
        os.makedirs(cache_dir, exist_ok=True)
        self._fine = {}

    @staticmethod
    def _vec(im, s):
        v = np.asarray(im.resize((s, s), Image.BILINEAR), np.float32).ravel()
        v = v - v.mean()
        return v / (np.linalg.norm(v) + 1e-6)

    def _thumb(self, f):
        im = Image.open(f).convert('L')
        return self._vec(im, 24), im.size[0] / im.size[1]

    def _fine_thumb(self, f):
        if f not in self._fine:
            self._fine[f] = self._vec(Image.open(f).convert('L'), 64)
        return self._fine[f]

    def thumbs(self, files, tag, ids=None):
        ids = ids or files
        h = hashlib.md5('\n'.join(ids).encode()).hexdigest()[:12]
        cache = os.path.join(self.cache_dir, f'thumbs_{tag}_{h}.npz')
        if os.path.exists(cache):
            z = np.load(cache)
            return z['X'], z['A']
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(self._thumb, files))
        X, A = np.stack([r[0] for r in res]), np.array([r[1] for r in res], np.float32)
        np.savez(cache, X=X, A=A)
        return X, A

    def duplicates(self, train_gts, test_gts, tag_train, tag_test, train_ids=None, test_ids=None):
        (Xa, Aa), (Xb, Ab) = self.thumbs(train_gts, tag_train, train_ids), self.thumbs(test_gts, tag_test, test_ids)
        dup = np.zeros(len(Xa), bool)
        match = [None] * len(Xa)
        for i in range(0, len(Xa), 2048):
            S = Xa[i:i + 2048] @ Xb.T
            for r, c in zip(*np.nonzero(S > self.threshold)):
                ia = i + r
                if dup[ia] or abs(Aa[ia] / Ab[c] - 1) > 0.05:
                    continue
                if float(self._fine_thumb(train_gts[ia]) @ self._fine_thumb(test_gts[c])) > self.confirm:
                    dup[ia], match[ia] = True, os.path.basename(test_gts[c])
        return dup, match


# ==========================================
# 4. Training pool manifest (paths stored relative to their source)
# ==========================================

def manifest_signature(cfg):
    return hashlib.md5(json.dumps([cfg.train_categories(), cfg.MAX_PER_CATEGORY, cfg.VAL_PER_CATEGORY, cfg.DATA_SUBSET_SEED,
                                   cfg.EXCLUDE_TEST_OVERLAP, cfg.OVERLAP_THRESHOLD, cfg.OVERLAP_CONFIRM, cfg.KOHLER_PROTOCOL,
                                   sorted(overlap_benchmarks(cfg).items())],
                                  sort_keys=True, default=str).encode()).hexdigest()


def manifest_path(cfg):
    return os.path.join(cfg.manifest_dir, 'train_manifest.json')


def manifest_is_current(cfg):
    p = manifest_path(cfg)
    return os.path.exists(p) and json.load(open(p)).get('signature') == manifest_signature(cfg)


def overlap_benchmarks(cfg):
    """family -> benchmarks whose clean images training data must not duplicate."""
    out = {}
    for name, s in cfg.benchmarks().items():
        if 'base' in s or 'lmdb' in s:
            continue
        out.setdefault(s['family'], []).append(name)
    return {k: sorted(v) for k, v in out.items()}


def resolve_manifest(man, roots):
    """Copy of the manifest with absolute paths for this session."""
    missing = sorted({it[k] for s in ('train', 'val') for it in man[s] for k in ('src_in', 'src_gt')} - set(roots))
    if missing:
        raise FileNotFoundError(f'data sources needed by the training manifest are not available: {missing} '
                                f'(see the download messages above)')
    out = dict(man)
    for split in ('train', 'val'):
        out[split] = [dict(it, input=os.path.join(roots[it['src_in']], it['input']),
                           gt=os.path.join(roots[it['src_gt']], it['gt'])) for it in man[split]]
    return out


def build_training_manifest(cfg, roots, force=False):
    """Pairs of every training category -> seeded shuffle -> leakage filtering of the candidates ->
    held-out validation split + capped training subset.  Written once (relative paths) to
    data_manifest/train_manifest.json; returned with absolute paths for this session."""
    os.makedirs(cfg.manifest_dir, exist_ok=True)
    path = manifest_path(cfg)
    sig = manifest_signature(cfg)
    if os.path.exists(path) and not force:
        man = json.load(open(path))
        if man.get('signature') == sig:
            return resolve_manifest(man, roots)
        print('[data] configuration changed -> rebuilding the training manifest')
    cats = cfg.train_categories()
    fam_bench = overlap_benchmarks(cfg)
    auditor = LeakageAuditor(os.path.join(cfg.manifest_dir, 'thumb_cache'), cfg.OVERLAP_THRESHOLD, cfg.OVERLAP_CONFIRM,
                             workers=cfg.NUM_WORKERS)
    rng = random.Random(cfg.DATA_SUBSET_SEED)
    man = dict(signature=sig, categories={}, train=[], val=[], audit={})
    bench_gts = {}
    errors = []
    for ci, (name, spec) in enumerate(cats.items()):
        try:
            pairs, rep = spec_pairs(spec, roots)
        except FileNotFoundError as e:
            errors.append(f'{name}: {e}')
            man['categories'][name] = dict(available=False, error=str(e))
            continue
        ki, kg = spec_keys(spec)
        pairs = sorted(pairs)
        rng.shuffle(pairs)
        n_val = min(cfg.VAL_PER_CATEGORY, max(1, len(pairs) // 10)) if len(pairs) >= 5 else 0
        need = n_val + cfg.MAX_PER_CATEGORY
        checks = sorted(set(spec.get('exclude_overlap_with', [])) | set(fam_bench.get(spec['family'], []))) \
            if cfg.EXCLUDE_TEST_OVERLAP else []
        for b in checks:
            if b not in bench_gts:
                try:
                    bs = cfg.benchmarks()[b]
                    kb = bs['clean'][0] if 'clean' in bs else spec_keys(bs)[1]
                    files = benchmark_gt_paths(cfg, roots, b)
                    bench_gts[b] = (files, [os.path.relpath(f, roots[kb]) for f in files])
                except (FileNotFoundError, KeyError) as e:
                    print(f'[data] overlap check vs {b} skipped: {e}')
                    bench_gts[b] = ([], [])
        kept, removed, pos = [], [], 0
        while len(kept) < need and pos < len(pairs):
            chunk = pairs[pos:pos + int(1.2 * (need - len(kept))) + 32]
            pos += len(chunk)
            dup = np.zeros(len(chunk), bool)
            ids = [f'{ki}/{os.path.relpath(g, roots[kg])}' for _, g, _ in chunk]
            for b in checks:
                files, rel = bench_gts[b]
                if not files:
                    continue
                d, match = auditor.duplicates([g for _, g, _ in chunk], files, name, b, ids, rel)
                a = man['audit'].setdefault(f'{name}_vs_{b}', dict(n_checked=0, n_duplicates=0, examples=[]))
                a['n_checked'] += len(chunk)
                a['n_duplicates'] += int(d.sum())
                a['examples'] = (a['examples'] + [f'{os.path.basename(chunk[i][1])} = {b}/{match[i]}' for i in np.nonzero(d)[0]])[:10]
                dup |= d
            removed += [chunk[i] for i in np.nonzero(dup)[0]]
            kept += [chunk[i] for i in np.nonzero(~dup)[0]]
        kept = kept[:need]
        val, train = kept[:n_val], sorted(kept[n_val:])
        rep = dict(rep, available=True, family=spec['family'], index=ci, n_removed_overlap=len(removed),
                   n_candidates_checked=pos, overlap_checked_against=checks,
                   n_train=len(train), n_val=len(val), repeat=spec.get('repeat', 1))
        man['categories'][name] = rep

        def item(a, g):
            return dict(src_in=ki, input=os.path.relpath(a, roots[ki]), src_gt=kg, gt=os.path.relpath(g, roots[kg]),
                        category=name, family=spec['family'])
        man['train'] += [dict(item(a, g), repeat=spec.get('repeat', 1)) for a, g, _ in train]
        man['val'] += [item(a, g) for a, g, _ in val]
        print(f'[data] {name:11s} pairs={rep["n_pairs"]:6d}  checked={pos:6d}  overlap-removed={len(removed):4d}  '
              f'train={len(train):5d}  val={len(val):3d}  repeat={spec.get("repeat", 1)}')
    if errors and not cfg.ALLOW_MISSING_CATEGORIES:
        raise RuntimeError('training categories without data:\n  ' + '\n  '.join(errors))
    with open(path + '.tmp', 'w') as f:   # atomic: several sessions may build it at the same time
        json.dump(man, f, indent=1)
    os.replace(path + '.tmp', path)
    return resolve_manifest(man, roots)


def benchmark_gt_paths(cfg, roots, name):
    spec = cfg.benchmarks()[name]
    if 'clean' in spec:
        return list_images(resolve_path(roots, spec['clean']))
    pairs, _ = spec_pairs(spec, roots)
    return [g for _, g, _ in pairs]


def manifest_used_files(cfg, roots, man):
    """{source key: set(relative paths)} read by the experiments (Drive cache in 'used' mode)."""
    used = {}
    for split in ('train', 'val'):
        for it in man[split]:
            used.setdefault(it['src_in'], set()).add(os.path.relpath(it['input'], roots[it['src_in']]))
            used.setdefault(it['src_gt'], set()).add(os.path.relpath(it['gt'], roots[it['src_gt']]))
    for name in cfg.benchmarks():
        try:
            for k, r in benchmark_files(cfg, roots, name):
                used.setdefault(k, set()).add(r)
        except (FileNotFoundError, KeyError):
            pass
    return used


# ==========================================
# 4b. Pair validation (alignment, direction of the degradation, sizes, readability)
# ==========================================

DIRECTION = {'blur': 'sharper_gt', 'defocus': 'sharper_gt', 'rain': 'cleaner_gt', 'snow': 'cleaner_gt',
             'noise': 'cleaner_gt', 'haze': 'contrast_gt', 'lowlight': 'brighter_gt'}


def _pair_stats(a_path, g_path, a_img=None, g_img=None):
    a = a_img or Image.open(a_path).convert('RGB')
    g = g_img or Image.open(g_path).convert('RGB')
    same = a.size == g.size
    if not same:
        g = g.resize(a.size, Image.BICUBIC)
    s = 512 / max(a.size)
    if s < 1:
        a, g = a.resize((max(1, int(a.size[0] * s)), max(1, int(a.size[1] * s))), Image.BILINEAR), \
               g.resize((max(1, int(a.size[0] * s)), max(1, int(a.size[1] * s))), Image.BILINEAR)
    A = np.asarray(a.convert('L'), np.float32) / 255.
    G = np.asarray(g.convert('L'), np.float32) / 255.
    hf = lambda x: float(np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean())
    va, vg = LeakageAuditor._vec(Image.fromarray((A * 255).astype(np.uint8)), 64), LeakageAuditor._vec(Image.fromarray((G * 255).astype(np.uint8)), 64)
    mse = float(((np.asarray(a, np.float32) - np.asarray(g, np.float32)) ** 2).mean()) / 255. ** 2
    return dict(same_size=same, corr=float(va @ vg), psnr=10 * np.log10(1 / (mse + 1e-10)), hf_in=hf(A), hf_gt=hf(G),
                std_in=float(A.std()), std_gt=float(G.std()), mean_in=float(A.mean()), mean_gt=float(G.mean()))


def _direction_ok(st, rule):
    return {'sharper_gt': st['hf_in'] < st['hf_gt'], 'cleaner_gt': st['hf_in'] > st['hf_gt'],
            'contrast_gt': st['std_in'] < st['std_gt'], 'brighter_gt': st['mean_in'] < st['mean_gt']}[rule]


def _summarise(name, family, stats, rule, cfg, extra=None):
    corr = [s['corr'] for s in stats]
    r = dict(name=name, family=family, n_checked=len(stats), median_corr=float(np.median(corr)) if corr else None,
             min_corr=float(np.min(corr)) if corr else None, median_input_psnr=float(np.median([s['psnr'] for s in stats])) if stats else None,
             n_low_corr=int(sum(c < cfg.MIN_PAIR_CORR for c in corr)), n_size_mismatch=int(sum(not s['same_size'] for s in stats)),
             direction_rule=rule, **(extra or {}))
    r['direction_rate'] = float(np.mean([_direction_ok(s, rule) for s in stats])) if (rule and stats) else None
    flags = []
    if r['median_corr'] is not None and r['median_corr'] < cfg.MIN_PAIR_CORR:
        flags.append('MISPAIR')
    if r['direction_rate'] is not None and r['direction_rate'] < cfg.MIN_DIRECTION_RATE:
        flags.append('SWAP?')
    if r['n_size_mismatch']:
        flags.append('SIZE')
    if r.get('n_unreadable'):
        flags.append('UNREADABLE')
    if r.get('pair_rate') is not None and r['pair_rate'] < 0.9:
        flags.append('UNMATCHED')
    r['flags'] = flags
    return r


def validate_data(cfg, roots, man):
    """Checks a sample of every training category and every benchmark; writes
    data_manifest/data_validation.json.  Training categories flagged MISPAIR or SWAP? stop the run."""
    rng = random.Random(cfg.DATA_SUBSET_SEED + 1)
    rows = []

    def sample_stats(pairs):
        stats, bad = [], 0
        for a, g in rng.sample(pairs, min(cfg.VALIDATE_SAMPLES, len(pairs))):
            try:
                stats.append(_pair_stats(a, g))
            except Exception:
                bad += 1
        return stats, bad

    for name, spec in cfg.train_categories().items():
        items = [(it['input'], it['gt']) for it in man['train'] if it['category'] == name]
        if not items:
            continue
        stats, bad = sample_stats(items)
        cat = man['categories'][name]
        pr = cat['n_pairs'] / max(1, min(cat.get('n_input_keys', cat['n_pairs']), cat.get('n_gt_keys', cat['n_pairs'])))
        rows.append(dict(_summarise(name, spec['family'], stats, spec.get('direction', DIRECTION.get(spec['family'])), cfg,
                                    dict(n_unreadable=bad, pair_rate=pr)), kind='train'))
    for name, spec in cfg.benchmarks().items():
        if 'base' in spec or 'clean' in spec:
            continue
        try:
            if 'lmdb' in spec:
                rd = LMDBPairReader(resolve_path(roots, spec['lmdb']), resolve_path(roots, spec['lmdb_gt']))
                keys = rng.sample(rd.keys, min(cfg.VALIDATE_SAMPLES, len(rd.keys)))
                stats = [_pair_stats(None, None, *rd.get(k)) for k in keys]
                bad, extra = 0, dict(n_pairs=len(rd.keys))
            else:
                pairs, rep = spec_pairs(spec, roots)
                stats, bad = sample_stats([(a, g) for a, g, _ in pairs])
                extra = dict(n_pairs=rep['n_pairs'], pair_rate=rep['n_pairs'] / max(1, min(rep['n_input_keys'], rep['n_gt_keys'])))
        except (FileNotFoundError, KeyError) as e:
            rows.append(dict(name=name, kind='benchmark', family=spec['family'], flags=['MISSING'], error=str(e)))
            continue
        rows.append(dict(_summarise(name, spec['family'], stats, spec.get('direction', DIRECTION.get(spec['family'])), cfg,
                                    dict(extra, n_unreadable=bad)), kind='benchmark'))
    os.makedirs(cfg.manifest_dir, exist_ok=True)
    with open(os.path.join(cfg.manifest_dir, 'data_validation.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print(f'{"set":22s} {"kind":9s} {"n":>4s} {"corr":>6s} {"dir":>5s} {"psnr_in":>7s}  flags')
    for r in rows:
        print(f'{r["name"]:22s} {r["kind"]:9s} {r.get("n_checked", 0):4d} {r.get("median_corr") or 0:6.3f} '
              f'{(r.get("direction_rate") if r.get("direction_rate") is not None else float("nan")):5.2f} '
              f'{r.get("median_input_psnr") or 0:7.2f}  {",".join(r["flags"])}')
    fatal = [r['name'] for r in rows if r['kind'] == 'train' and set(r['flags']) & {'MISPAIR', 'SWAP?', 'UNREADABLE'}]
    if fatal and cfg.VALIDATION_STRICT:
        raise RuntimeError(f'training data failed validation: {fatal} (see data_manifest/data_validation.json)')
    return rows


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
    vertical flips.  Images whose shorter side is below the patch size are first up-scaled
    (bicubic, aspect ratio kept; the submitted code stretched them to a square).
    Validation: centre crop."""

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
        if w < p or h < p:   # e.g. CelebA 178x218: bicubic up-scaling that keeps the aspect ratio (short side = p)
            sc = p / min(w, h)
            new_w, new_h = max(p, round(w * sc)), max(p, round(h * sc))
            image = image.resize((new_w, new_h), Image.BICUBIC)
            gt = gt.resize((new_w, new_h), Image.BICUBIC)
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
