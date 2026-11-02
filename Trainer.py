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
import math
import time
import uuid
import random
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from Datasets import UniversalRestorationDataset  # nb:skip
from Model import KASIRNet, component_parameter_table  # nb:skip
from Loss import OmniRestorationLoss  # nb:skip
from metrics import MetricSuite, psnr_torch  # nb:skip

FAMILIES = ['blur', 'rain', 'haze', 'snow', 'noise']


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


class Trainer:
    def __init__(self, cfg, run, manifest, device=None):
        self.cfg, self.run = cfg, run
        self.run_id = run['run_id']
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
        self.criterion = OmniRestorationLoss(cfg.loss_weights(run), amp=self.use_amp, amp_dtype=self.amp_dtype).to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=cfg.LR, betas=tuple(cfg.BETAS), weight_decay=cfg.WEIGHT_DECAY)
        floor = cfg.ETA_MIN / cfg.LR
        T = max(1, self.total_iters)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda it: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(it, T) / T)))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.metrics = MetricSuite(self.device, ('psnr', 'ssim'))

        self.iteration, self.best_val, self.best_iter, self.train_seconds = 0, -1.0, 0, 0.0
        self.history = dict(train=[], val=[])
        self.skipped_steps = 0
        self._resume()

    # ------------------------------------------------------------------ io
    def _ckpt(self, name):
        return os.path.join(self.ckdir, name)

    def _state(self, full=True):
        s = dict(run_id=self.run_id, iteration=self.iteration, total_iters=self.total_iters,
                 model_state_dict=self.model.state_dict(), model_flags=self.cfg.model_kwargs(self.run),
                 loss_weights=self.cfg.loss_weights(self.run), best_val_psnr=self.best_val, best_iter=self.best_iter)
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
        self.model.load_state_dict(ck['model_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.scheduler.load_state_dict(ck['scheduler_state_dict'])
        self.scaler.load_state_dict(ck['scaler_state_dict'])
        self.iteration, self.best_val, self.best_iter = ck['iteration'], ck['best_val_psnr'], ck['best_iter']
        self.history, self.train_seconds = ck['history'], ck.get('train_seconds', 0.0)
        self.skipped_steps = ck.get('skipped_steps', 0)
        r = ck.get('rng')
        if r:
            random.setstate(r['python'])
            np.random.set_state(r['numpy'])
            torch.set_rng_state(r['torch'].cpu())
            if r.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in r['cuda']])
        print(f'[{self.run_id}] resumed at iteration {self.iteration}/{self.total_iters}')

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
        rec.update(run_id=self.run_id, group=self.run['group'], seed=self.run['seed'], status=status,
                   budget=dict(epochs=self.run['epochs'], iters_per_epoch=self.ipe, total_iters=self.total_iters,
                               iterations_done=self.iteration, batch_size=self.cfg.BATCH_SIZE, patch=self.cfg.PATCH_SIZE,
                               samples_seen=self.iteration * self.cfg.BATCH_SIZE),
                   optimisation=dict(optimizer='Adam', lr=self.cfg.LR, betas=list(self.cfg.BETAS), weight_decay=self.cfg.WEIGHT_DECAY,
                                     schedule=f'cosine per iteration, {self.cfg.LR} -> {self.cfg.ETA_MIN}', grad_clip=self.cfg.GRAD_CLIP,
                                     amp=self.use_amp, amp_dtype=str(self.amp_dtype), skipped_nonfinite_steps=self.skipped_steps),
                   model_flags=self.cfg.model_kwargs(self.run), loss_weights=self.cfg.loss_weights(self.run),
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
