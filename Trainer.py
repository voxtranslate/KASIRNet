# ==========================================================================
# KASIRNet - Trainer (resumable, iteration-based, FP16, curves, run.json)
# --------------------------------------------------------------------------
# * Budget = EPOCHS x floor(N_train / BATCH_SIZE) iterations; Adam, per-iteration
#   cosine annealing, gradient clipping 1.0, FP16 autocast + GradScaler.
# * Automatic resume: last.pth (model, optimiser, scheduler, scaler, data-order
#   position, RNG states, history, accumulated wall-clock) is written atomically
#   every CKPT_EVERY_ITERS iterations, so an epoch may span several sessions and
#   a resumed run follows exactly the same data order.
# * Model selection: best mean PSNR on the held-out validation split of the
#   training pool (never a test set), or the last iterate.  Stopping criterion:
#   the fixed iteration budget (no early stopping).
# * Writes runs/<run_id>/run.json, history.csv and curves.png (train vs val).
# ==========================================================================
import os
import csv
import json
import hashlib
import math
import time
import uuid
import random
import platform
import warnings
import numpy as np

# With FP16 the GradScaler skips the optimiser step of the first iterations (inf gradients while it
# calibrates its scale); PyTorch then warns that the scheduler stepped first.  Harmless and expected.
warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from Datasets import UniversalRestorationDataset, manifest_signature  # nb:skip
from Model import KASIRNet, component_parameter_table  # nb:skip
from Loss import OmniRestorationLoss  # nb:skip
from metrics import MetricSuite, psnr_torch  # nb:skip

FAMILIES = ['blur', 'rain', 'haze', 'snow', 'noise', 'mixed', 'lowlight', 'defocus', 'jpeg']


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def environment_info():
    info = dict(python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                cpu_count=os.cpu_count())
    try:
        import torchvision
        info['torchvision'] = torchvision.__version__
    except Exception:
        pass
    try:
        import pyiqa
        info['pyiqa'] = pyiqa.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_count=torch.cuda.device_count(), gpu_memory_gb=round(p.total_memory / 2 ** 30, 1),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        try:
            import subprocess
            info['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                                     text=True).strip().splitlines()[0]
        except Exception:
            pass
    return info


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


class ResumableSampler(Sampler):
    """Epoch-seeded permutation; starts at `start` so that a resumed epoch continues
    with the very same sample order."""

    def __init__(self, n, seed, epoch, start=0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


SESSION_ID = uuid.uuid4().hex[:12]


def lock_path(cfg, run_id):
    return os.path.join(cfg.run_dir(run_id), 'LOCK.json')


def try_lock(cfg, run_id):
    """True if this session may train `run_id` (no other live session holds it)."""
    lk = read_json(lock_path(cfg, run_id))
    if lk and lk.get('session') != SESSION_ID and time.time() - lk.get('heartbeat', 0) < cfg.LOCK_STALE_MINUTES * 60:
        return False
    write_json(dict(session=SESSION_ID, host=platform.node(), heartbeat=time.time()), lock_path(cfg, run_id))
    return True


def release_lock(cfg, run_id):
    lk = read_json(lock_path(cfg, run_id))
    if lk and lk.get('session') == SESSION_ID:
        os.remove(lock_path(cfg, run_id))


def build_model(cfg, run):
    return KASIRNet(**cfg.model_kwargs(run))


def load_weights(model, path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=True)
    return ck


LOSS_CALIBRATION_VERSION = 1
TRAIN_CODE_VERSION = 3   # bumped when the training computation changes (3: calibrated loss, ortho FFT, eval-mode loss nets, no train clamp)


def recipe_core(cfg, run):
    """What a run learns apart from its length: training pool, optimiser, warm-up, initialisation, seed, model, loss."""
    blob = [manifest_signature(cfg), float(run.get('lr', cfg.LR)), int(run.get('warmup', cfg.WARMUP_ITERS)),
            run.get('init_from'), run['seed'], cfg.model_kwargs(run), cfg.loss_weights(run), cfg.LOSS_CALIBRATION,
            LOSS_CALIBRATION_VERSION, cfg.BATCH_SIZE, cfg.PATCH_SIZE, list(cfg.BETAS), cfg.WEIGHT_DECAY, cfg.ETA_MIN,
            cfg.GRAD_CLIP, cfg.AMP, cfg.AMP_DTYPE, TRAIN_CODE_VERSION]
    return hashlib.md5(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def recipe_signature(cfg, run):
    """Core recipe + budget: a COMPLETE run counts only with exactly this signature."""
    return hashlib.md5(json.dumps([recipe_core(cfg, run), run['epochs']]).encode()).hexdigest()


def _legacy_signature(cfg, run, rec):
    """Signature of `run` with the budget the stored record was trained for (records written before recipe_core)."""
    ep = (rec.get('budget') or {}).get('epochs')
    return recipe_signature(cfg, dict(run, epochs=ep)) if ep is not None else None


def resumable_checkpoint(cfg, run, rdir, cdir):
    """Iteration of a paused checkpoint in (rdir, cdir) that has the same core recipe as `run` (any budget), else None.
    Records of earlier notebook versions (no recipe_core) are checked against the checkpoint's own metadata."""
    rec = read_json(os.path.join(rdir, 'run.json'), {})
    path = os.path.join(cdir, 'last.pth')
    if not os.path.exists(path) or rec.get('status') == 'complete':
        return None
    core = recipe_core(cfg, run)
    ok = rec.get('recipe_core') == core or (rec.get('recipe_signature') and rec.get('recipe_signature') == _legacy_signature(cfg, run, rec))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not ok:   # legacy record: compare what the checkpoint and the record state explicitly
        opt = rec.get('optimisation') or {}
        ok = (ck.get('manifest_signature') == manifest_signature(cfg) and ck.get('model_flags') == cfg.model_kwargs(run)
              and ck.get('loss_weights') == cfg.loss_weights(run) and rec.get('seed') == run['seed']
              and 'loss_scales' in rec and opt.get('warmup_iters') == int(run.get('warmup', cfg.WARMUP_ITERS))
              and abs(float(opt.get('lr', -1)) - float(run.get('lr', cfg.LR))) < 1e-12
              and (opt.get('initialised_from') or None) == (run.get('init_from') or None))
    return int(ck.get('iteration', 0)) if ok else None


def supersede_if_stale(cfg, run):
    """Moves runs/<id> and checkpoints/<id> aside (<id>__superseded_N, nothing deleted) when they were produced
    with another recipe.  Kept: a complete run with exactly this recipe and budget, and a PAUSED run with the same
    core recipe (any budget: it resumes and its cosine schedule follows the new budget)."""
    rid = run['run_id']
    rdir, cdir = cfg.run_dir(rid), cfg.ckpt_dir(rid)
    rec = read_json(os.path.join(rdir, 'run.json'), {})
    has_ck = os.path.isdir(cdir) and any(f.endswith('.pth') for f in os.listdir(cdir))
    if not rec and not has_ck:
        return False
    if rec.get('recipe_signature') == recipe_signature(cfg, run) and rec.get('status') == 'complete':
        return False
    if rec.get('status') != 'complete' and resumable_checkpoint(cfg, run, rdir, cdir) is not None:
        return False
    _move_aside(rdir, cdir)
    print(f'[{rid}] !! stored run was trained with another recipe (training pool, budget, learning rate, loss or code '
          f'version): moved aside (*__superseded_N); it is trained again with the current recipe')
    return True


def _move_aside(rdir, cdir):
    k = 1
    while os.path.exists(f'{rdir}__superseded_{k}') or os.path.exists(f'{cdir}__superseded_{k}'):
        k += 1
    for d in (rdir, cdir):
        if os.path.exists(d):
            os.replace(d, f'{d}__superseded_{k}')


def adopt_checkpoint(cfg, run, also=()):
    """Do not lose training time: if `run` has no resumable checkpoint of its own, take over the most advanced
    PAUSED checkpoint with the same core recipe among <id>__superseded_*, and the runs in `also` (e.g. the final-model
    run, which has the same recipe as the reference when both are trained from scratch with the same seed)."""
    rid = run['run_id']
    rdir, cdir = cfg.run_dir(rid), cfg.ckpt_dir(rid)
    own = resumable_checkpoint(cfg, run, rdir, cdir) if os.path.isdir(cdir) else None
    root_r, root_c = os.path.dirname(rdir), os.path.dirname(cdir)
    names = set()
    for base in (os.path.basename(rdir),) + tuple(also):
        if os.path.isdir(root_c):
            names |= {n for n in os.listdir(root_c) if n == base or n.startswith(base + '__superseded_')}
    best = (own if own is not None else -1, None)
    for n in sorted(names - {os.path.basename(cdir)}):
        try:
            it = resumable_checkpoint(cfg, run, os.path.join(root_r, n), os.path.join(root_c, n))
        except Exception as e:
            print(f'[{rid}] checkpoint {n} not readable: {e}')
            continue
        if it is not None and it > best[0]:
            best = (it, n)
    if best[1] is None:
        return False
    if os.path.exists(rdir) or os.path.exists(cdir):
        _move_aside(rdir, cdir)
    os.replace(os.path.join(root_c, best[1]), cdir)
    if os.path.isdir(os.path.join(root_r, best[1])):
        os.replace(os.path.join(root_r, best[1]), rdir)
    print(f'[{rid}] continues from the checkpoint of {best[1]} (iteration {best[0]}): same training pool, model, loss and '
          f'learning rate; only the budget may differ')
    return True


def loss_scales(cfg, manifest, device):
    """Fixed per-term scales of the loss (Loss.py, 'Calibrated weights'): s_k = g_pix / g_k with g_k the mean
    norm of dL_k/d(pred) at pred = degraded input over the held-out validation patches.  Model-independent,
    computed once per training pool and shared by every run (data_manifest/loss_calibration.json)."""
    terms = OmniRestorationLoss.TERMS
    if not cfg.LOSS_CALIBRATION:
        return {k: 1.0 for k in terms}
    path = os.path.join(cfg.manifest_dir, 'loss_calibration.json')
    key = [manifest['signature'], LOSS_CALIBRATION_VERSION, cfg.PATCH_SIZE]
    rec = read_json(path, {})
    if rec.get('key') == key:
        return rec['scales']
    crit = OmniRestorationLoss({k: 1.0 for k in terms}).to(device)
    ds = UniversalRestorationDataset(manifest['val'], cfg.PATCH_SIZE, False, FAMILIES)
    norms = {k: [] for k in terms}
    for b in DataLoader(ds, batch_size=16, shuffle=False, num_workers=min(4, cfg.NUM_WORKERS)):
        x, y = b['input'].to(device), b['gt'].to(device)
        for k in terms:
            p = x.clone().requires_grad_(True)
            val = crit.terms(p, y, only={k})[k]
            g, = torch.autograd.grad(val, p)
            norms[k].append(float(g.norm()))
    g = {k: float(np.mean(v)) for k, v in norms.items()}
    scales = {k: (g['pix'] / g[k] if g[k] > 0 else 1.0) for k in terms}
    w = cfg.LOSS_WEIGHTS
    share = {k: w[k] / sum(w.values()) for k in terms}
    raw = {k: w[k] * g[k] / sum(w[j] * g[j] for j in terms) for k in terms}
    write_json(dict(key=key, grad_norm_at_input=g, scales=scales, gradient_share_uncalibrated=raw,
                    gradient_share_calibrated=share, n_patches=len(ds)), path)
    print('[loss] calibrated scales ' + ', '.join(f'{k} {v:.3g}' for k, v in scales.items()) +
          ' | gradient share without calibration ' + ', '.join(f'{k} {100 * v:.0f}%' for k, v in raw.items()))
    del crit
    return scales


class Trainer:
    def __init__(self, cfg, run, manifest, device=None):
        self.cfg, self.run = cfg, run
        self.run_id = run['run_id']
        supersede_if_stale(cfg, run)
        self.recipe, self.recipe_core = recipe_signature(cfg, run), recipe_core(cfg, run)
        self.device = torch.device(device or (cfg.DEVICE if torch.cuda.is_available() else 'cpu'))
        self.ckdir, self.rundir = cfg.ckpt_dir(self.run_id), cfg.run_dir(self.run_id)
        os.makedirs(self.ckdir, exist_ok=True)
        os.makedirs(self.rundir, exist_ok=True)
        self.json_path = os.path.join(self.rundir, 'run.json')
        self.record = read_json(self.json_path, {})
        if self.record.get('status') == 'complete':
            return
        seed_everything(run['seed'], cfg.DETERMINISTIC)
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK and not cfg.DETERMINISTIC

        self.manifest = manifest
        self.train_ds = UniversalRestorationDataset(manifest['train'], cfg.PATCH_SIZE, True, FAMILIES)
        self.val_ds = UniversalRestorationDataset(manifest['val'], cfg.PATCH_SIZE, False, FAMILIES)
        self.ipe = len(self.train_ds) // cfg.BATCH_SIZE
        self.total_iters = int(run['epochs'] * self.ipe)

        self.model = build_model(cfg, run).to(self.device)
        self.amp_dtype = torch.bfloat16 if cfg.AMP_DTYPE == 'bfloat16' else torch.float16
        self.use_amp = bool(cfg.AMP and self.device.type == 'cuda')
        self.loss_scales = loss_scales(cfg, manifest, self.device)
        self.criterion = OmniRestorationLoss(cfg.loss_weights(run), amp=self.use_amp, amp_dtype=self.amp_dtype,
                                             scales=self.loss_scales).to(self.device).eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.lr = float(run.get('lr', cfg.LR))
        self.warmup = int(run.get('warmup', cfg.WARMUP_ITERS))
        self.optimizer = torch.optim.Adam(params, lr=self.lr, betas=tuple(cfg.BETAS), weight_decay=cfg.WEIGHT_DECAY)
        floor = cfg.ETA_MIN / self.lr
        T, W = max(1, self.total_iters), min(self.warmup, max(0, self.total_iters - 1))

        def lr_factor(it):   # linear warm-up over W iterations, then cosine from lr to ETA_MIN over the rest
            if it < W:
                return (it + 1) / W
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(it - W, T - W) / max(1, T - W)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.metrics = MetricSuite(self.device, ('psnr', 'ssim'))

        self.iteration, self.best_val, self.best_iter, self.train_seconds = 0, -1.0, 0, 0.0
        self.history = dict(train=[], val=[])
        self.skipped_steps = 0
        self.init_from = None
        self._resume()
        if self.iteration == 0 and run.get('init_from'):
            self._initialise_from(run['init_from'])

    # ------------------------------------------------------------------ io
    def _ckpt(self, name):
        return os.path.join(self.ckdir, name)

    def _state(self, full=True):
        s = dict(run_id=self.run_id, iteration=self.iteration, total_iters=self.total_iters,
                 manifest_signature=self.manifest['signature'],
                 model_state_dict=self.model.state_dict(), model_flags=self.cfg.model_kwargs(self.run),
                 loss_weights=self.cfg.loss_weights(self.run), best_val_psnr=self.best_val, best_iter=self.best_iter,
                 init_from=self.init_from)
        if full:
            s.update(optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(),
                     scaler_state_dict=self.scaler.state_dict(), history=self.history, train_seconds=self.train_seconds,
                     skipped_steps=self.skipped_steps,
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        return s

    def _resume(self):
        path = self._ckpt('last.pth')
        if not os.path.exists(path):
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        prev = ck.get('manifest_signature') or self.record.get('data', {}).get('manifest_signature')
        if prev and prev != self.manifest['signature']:
            # the training pool changed: resuming would mix two pools inside one run -> restart this run
            for n in ('last.pth', 'best.pth'):
                if os.path.exists(self._ckpt(n)):
                    os.replace(self._ckpt(n), self._ckpt(n.replace('.pth', '_previous_pool.pth')))
            print(f'[{self.run_id}] !! the training pool changed since this checkpoint (iteration {ck["iteration"]}): '
                  f'the run restarts from scratch so that it is trained on one pool only (old checkpoint kept as last_previous_pool.pth)')
            self.record = {}
            return
        if ck.get('total_iters') != self.total_iters:
            print(f'[{self.run_id}] !! the budget changed since the checkpoint ({ck.get("total_iters")} -> {self.total_iters} '
                  f'iterations, e.g. EPOCHS or the training pool changed); the cosine schedule follows the new budget')
        self.model.load_state_dict(ck['model_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.scheduler.load_state_dict(ck['scheduler_state_dict'])
        self.scaler.load_state_dict(ck['scaler_state_dict'])
        self.iteration, self.best_val, self.best_iter = ck['iteration'], ck['best_val_psnr'], ck['best_iter']
        self.history, self.train_seconds = ck['history'], ck.get('train_seconds', 0.0)
        self.skipped_steps = ck.get('skipped_steps', 0)
        self.init_from = ck.get('init_from')
        r = ck.get('rng')
        if r:
            random.setstate(r['python'])
            np.random.set_state(r['numpy'])
            torch.set_rng_state(r['torch'].cpu())
            if r.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in r['cuda']])
        print(f'[{self.run_id}] resumed at iteration {self.iteration}/{self.total_iters}')

    def _initialise_from(self, src_id):
        """Warm restart: start from the selected weights of another (complete) run with the same model."""
        path = os.path.join(self.cfg.ckpt_dir(src_id), 'final.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{self.run_id} starts from {src_id}, whose final.pth does not exist yet')
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get('manifest_signature') not in (None, self.manifest['signature']):
            raise RuntimeError(f'{src_id} was trained on another training pool than the current manifest')
        self.model.load_state_dict(ck['model_state_dict'], strict=True)
        self.init_from = dict(run_id=src_id, selected=ck.get('selected'))
        print(f'[{self.run_id}] initialised from {src_id} ({ck.get("selected")}); warm restart at lr {self.lr:g}, '
              f'warm-up {self.warmup} iterations')

    # ---------------------------------------------------------- validation
    @torch.no_grad()
    def validate(self):
        if len(self.val_ds) == 0:
            return dict(psnr=float('nan'), ssim=float('nan'), per_category={})
        self.model.eval()
        loader = DataLoader(self.val_ds, batch_size=8, shuffle=False, num_workers=min(4, self.cfg.NUM_WORKERS))
        per_cat, ps, ss = {}, [], []
        inv = {v: k for k, v in self.val_ds.task_to_id.items()}
        for b in loader:
            out, _ = self.model(b['input'].to(self.device))
            m = self.metrics(out, b['gt'])
            ps += m['psnr']
            ss += m['ssim']
            for t, p in zip(b['task_id'].tolist(), m['psnr']):
                per_cat.setdefault(inv[t], []).append(p)
        self.model.train()
        return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), per_category={k: float(np.mean(v)) for k, v in per_cat.items()})

    # ---------------------------------------------------------------- curves
    def _write_curves(self):
        rows = self.history['train']
        if rows:
            with open(os.path.join(self.rundir, 'history_train.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
                w.writeheader()
                w.writerows(rows)
        vrows = [{k: v for k, v in r.items() if k != 'per_category'} for r in self.history['val']]
        if vrows:
            with open(os.path.join(self.rundir, 'history_val.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(vrows[-1].keys()))
                w.writeheader()
                w.writerows(vrows)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
            it = [r['iteration'] for r in rows]
            for k in ('total', 'pix', 'fft', 'perc', 'ssim', 'id', 'edge'):
                if rows and k in rows[0]:
                    ax[0].plot(it, [r[k] for r in rows], label=k, lw=1)
            ax[0].set_yscale('log'); ax[0].set_title('training loss terms (unweighted) and total'); ax[0].legend(fontsize=7)
            ax[1].plot(it, [r['train_psnr'] for r in rows], label='train (random patches)', lw=1)
            ax[1].plot([r['iteration'] for r in vrows], [r['psnr'] for r in vrows], 'o-', label='validation (held-out)', ms=3)
            ax[1].set_title('PSNR (dB)'); ax[1].legend()
            ax[2].plot(it, [r['lr'] for r in rows]); ax[2].set_title('learning rate')
            for a in ax:
                a.set_xlabel('iteration'); a.grid(alpha=.3)
            fig.suptitle(self.run_id)
            fig.tight_layout()
            fig.savefig(os.path.join(self.rundir, 'curves.png'), dpi=120)
            plt.close(fig)
        except Exception as e:
            print('[curves] skipped:', e)

    def _write_record(self, status):
        rec = dict(self.record)
        rec.update(run_id=self.run_id, label=self.run.get('label'), group=self.run['group'], seed=self.run['seed'], status=status,
                   recipe_signature=self.recipe, recipe_core=self.recipe_core,
                   budget=dict(epochs=self.run['epochs'], iters_per_epoch=self.ipe, total_iters=self.total_iters,
                               iterations_done=self.iteration, batch_size=self.cfg.BATCH_SIZE, patch=self.cfg.PATCH_SIZE,
                               samples_seen=self.iteration * self.cfg.BATCH_SIZE),
                   optimisation=dict(optimizer='Adam', lr=self.lr, betas=list(self.cfg.BETAS), weight_decay=self.cfg.WEIGHT_DECAY,
                                     warmup_iters=self.warmup, initialised_from=self.init_from,
                                     schedule=(f'linear warm-up {self.warmup} it, then ' if self.warmup else '') +
                                              f'cosine per iteration, {self.lr} -> {self.cfg.ETA_MIN}', grad_clip=self.cfg.GRAD_CLIP,
                                     amp=self.use_amp, amp_dtype=str(self.amp_dtype), skipped_nonfinite_steps=self.skipped_steps),
                   model_flags=self.cfg.model_kwargs(self.run), loss_weights=self.cfg.loss_weights(self.run),
                   loss_scales=self.loss_scales,
                   parameters_M=component_parameter_table(self.model),
                   data=dict(n_train_items=len(self.manifest['train']), n_train_samples_per_epoch=len(self.train_ds),
                             n_val=len(self.val_ds), manifest_signature=self.manifest['signature'],
                             categories={k: v.get('n_train') for k, v in self.manifest['categories'].items()}),
                   selection=dict(rule=self.cfg.MODEL_SELECTION, best_val_psnr=self.best_val, best_iter=self.best_iter,
                                  stopping='fixed iteration budget'),
                   wall_clock_hours=round(self.train_seconds / 3600, 3),
                   checkpoint=self._ckpt('final.pth') if status == 'complete' else self._ckpt('last.pth'),
                   environment=environment_info())
        if self.history['val']:
            rec['last_val'] = self.history['val'][-1]
        self.record = rec
        write_json(rec, self.json_path)

    # ------------------------------------------------------------------ fit
    def fit(self):
        if self.record.get('status') == 'complete':
            print(f'[{self.run_id}] already complete -> skipped')
            return self.record
        cfg = self.cfg
        print(f'[{self.run_id}] {len(self.train_ds)} samples/epoch, {self.ipe} it/epoch, budget {self.total_iters} it '
              f'({self.run["epochs"]} epochs), batch {cfg.BATCH_SIZE}, AMP={self.use_amp}')
        session_start = time.time()
        self._t_base, self._t_start = self.train_seconds, session_start
        self.model.train()
        acc, n_acc, t_last = {}, 0, time.time()
        while self.iteration < self.total_iters:
            epoch, pos = divmod(self.iteration, self.ipe)
            g = torch.Generator().manual_seed(self.run['seed'] * 7919 + epoch * 131 + pos)
            loader = DataLoader(self.train_ds, batch_size=cfg.BATCH_SIZE,
                                sampler=ResumableSampler(len(self.train_ds), self.run['seed'], epoch, pos * cfg.BATCH_SIZE),
                                num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY and self.device.type == 'cuda',
                                drop_last=True, generator=g, persistent_workers=False)
            for batch in loader:
                inputs = batch['input'].to(self.device, non_blocking=True)
                targets = batch['gt'].to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    restored, _ = self.model(inputs)
                loss, logs = self.criterion(restored, targets)
                if not torch.isfinite(loss):
                    self.skipped_steps += 1
                    if self.skipped_steps > 1000:
                        raise RuntimeError('too many non-finite losses')
                    self.iteration += 1
                    self.scheduler.step()
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.iteration += 1
                with torch.no_grad():
                    logs['train_psnr'] = float(psnr_torch(restored.detach(), targets).mean())
                for k, v in logs.items():
                    acc[k] = acc.get(k, 0.0) + v
                n_acc += 1
                self.train_seconds = self._t_base + (time.time() - self._t_start)  # wall-clock incl. data loading, validation, checkpoints

                if self.iteration % cfg.LOG_EVERY_ITERS == 0:
                    row = {k: v / n_acc for k, v in acc.items()}
                    row.update(iteration=self.iteration, epoch=self.iteration / self.ipe, lr=self.scheduler.get_last_lr()[0],
                               sec_per_iter=(time.time() - t_last) / n_acc)
                    self.history['train'].append(row)
                    eta_h = row['sec_per_iter'] * (self.total_iters - self.iteration) / 3600
                    print(f'[{self.run_id}] it {self.iteration}/{self.total_iters} ep {row["epoch"]:.2f} loss {row["total"]:.4f} '
                          f'psnr {row["train_psnr"]:.2f} lr {row["lr"]:.2e} {row["sec_per_iter"]:.3f}s/it ETA {eta_h:.1f}h')
                    acc, n_acc, t_last = {}, 0, time.time()
                if self.iteration % cfg.VAL_EVERY_ITERS == 0 or self.iteration == self.total_iters:
                    self._validate_and_select()
                if self.iteration % cfg.CKPT_EVERY_ITERS == 0:
                    atomic_save(self._state(), self._ckpt('last.pth'))
                    write_json(dict(session=SESSION_ID, host=platform.node(), heartbeat=time.time()), lock_path(cfg, self.run_id))
                    self._write_record('running')
                    self._write_curves()
                if self.iteration >= self.total_iters:
                    break
                if (time.time() - session_start) / 3600 > cfg.MAX_SESSION_HOURS:
                    atomic_save(self._state(), self._ckpt('last.pth'))
                    self._write_record('paused')
                    self._write_curves()
                    print(f'[{self.run_id}] session time limit reached -> paused at {self.iteration}; re-run to resume')
                    return self.record
        if not self.history['val'] or self.history['val'][-1]['iteration'] != self.iteration:
            self._validate_and_select()
        self._finalise()
        return self.record

    def _validate_and_select(self):
        v = self.validate()
        v['iteration'] = self.iteration
        self.history['val'].append(v)
        print(f'[{self.run_id}] validation @ {self.iteration}: PSNR {v["psnr"]:.3f} SSIM {v["ssim"]:.4f}')
        if v['psnr'] == v['psnr'] and v['psnr'] > self.best_val:
            self.best_val, self.best_iter = v['psnr'], self.iteration
            atomic_save(self._state(full=False), self._ckpt('best.pth'))

    def _finalise(self):
        """Keep only the selected weights as final.pth (weights only) to save Drive space."""
        if self.cfg.MODEL_SELECTION == 'best_val' and os.path.exists(self._ckpt('best.pth')):
            sel = torch.load(self._ckpt('best.pth'), map_location='cpu', weights_only=False)
            sel['selected'] = f'best_val @ iteration {self.best_iter}'
        else:
            sel = self._state(full=False)
            sel['selected'] = f'last @ iteration {self.iteration}'
        sel['training_complete'] = True
        atomic_save(sel, self._ckpt('final.pth'))
        self._write_curves()
        self._write_record('complete')
        self.record['selection']['selected'] = sel['selected']
        write_json(self.record, self.json_path)
        for n in ('last.pth', 'best.pth'):
            if os.path.exists(self._ckpt(n)):
                os.remove(self._ckpt(n))
        print(f'[{self.run_id}] complete: {sel["selected"]}, {self.train_seconds / 3600:.2f} h of training')
