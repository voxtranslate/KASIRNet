# ==========================================================================
# KASIRNet - Inferencer: benchmark evaluation + explainability suite
# --------------------------------------------------------------------------
# Evaluator.evaluate() restores every image of a benchmark at full resolution
# (batch 1, FP32), scores it with metrics.MetricSuite (the single metric
# function) and writes runs/<run_id>/eval/<benchmark>/per_image.csv with
#   image, key, PSNR, SSIM, LPIPS, DISTS, input PSNR/SSIM, H, W, latency
# plus summary.json; the run.json of the run receives the benchmark means.
# Competitor outputs held by the authors are scored with the same function.
# KASIRNetInferencer (explainability figures of the paper) is kept from the
# submitted notebook and runs on a few images per benchmark.
# ==========================================================================
import os
import csv
import glob
import time
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Datasets import BenchmarkDataset, collate_single, IMG_EXT  # nb:skip
from Model import SpectralResonanceBlock, SemanticInjection, SFTLayer  # nb:skip
from metrics import MetricSuite  # nb:skip
from Trainer import build_model, load_weights, read_json, write_json  # nb:skip


def save_png(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def write_rows(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


TEXT_FIELDS = {'image', 'key', 'gt_name', 'benchmark', 'method', 'run_id', 'competitor', 'metric', 'variant', 'level', 'family'}


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in TEXT_FIELDS:
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


class Evaluator:
    def __init__(self, cfg, roots, device=None, metric_names=('psnr', 'ssim', 'lpips', 'dists', 'psnr_y', 'ssim_y')):
        self.cfg, self.roots = cfg, roots
        self.device = torch.device(device or (cfg.DEVICE if torch.cuda.is_available() else 'cpu'))
        self.metric_names = tuple(metric_names)
        self.metrics = MetricSuite(self.device, self.metric_names)
        self._light = None

    @property
    def light(self):  # PSNR/SSIM only, for inputs
        if self._light is None:
            self._light = MetricSuite(self.device, ('psnr', 'ssim'))
        return self._light

    # ------------------------------------------------------------- models
    def load_run_model(self, run):
        model = build_model(self.cfg, run)
        path = os.path.join(self.cfg.ckpt_dir(run['run_id']), 'final.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{run["run_id"]}: no final.pth (training not complete)')
        load_weights(model, path)
        return model.to(self.device).eval()

    def bench_dir(self, run_id, bench, tag=''):
        return os.path.join(self.cfg.run_dir(run_id), 'eval', bench + tag)

    def is_done(self, run_id, bench, tag=''):
        return os.path.exists(os.path.join(self.bench_dir(run_id, bench, tag), 'summary.json'))

    # ----------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, model, run_id, bench, prompt_mode=None, save_images=False, collect_prompts=False,
                 max_images=None, metric_names=None, force=False):
        """prompt_mode=None: normal inference.  prompt_mode='first': for mixed benchmarks the
        descriptor p is replaced by the descriptor of the singly-degraded image (control)."""
        tag = '' if prompt_mode is None else f'__prompt_{prompt_mode}'
        out_dir = self.bench_dir(run_id, bench, tag)
        if self.is_done(run_id, bench, tag) and not force and not collect_prompts:
            return read_json(os.path.join(out_dir, 'summary.json'))
        names = tuple(metric_names or self.metric_names)
        suite = self.metrics if set(names) <= set(self.metric_names) else MetricSuite(self.device, names)
        ds = BenchmarkDataset(self.cfg, self.roots, bench, max_images=max_images)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=min(6, self.cfg.NUM_WORKERS), collate_fn=collate_single)
        rows, prompts, n_resized = [], [], 0
        amp = bool(self.cfg.EVAL_AMP and self.device.type == 'cuda')
        torch.backends.cudnn.benchmark = False   # test images have many different sizes
        for item in tqdm(loader, desc=f'{run_id} | {bench}{tag}', leave=False):
            x = item['input'].unsqueeze(0).to(self.device)
            gt = item['gt'].unsqueeze(0)
            override = None
            if prompt_mode == 'first' and 'first' in item and getattr(model, 'use_prompt', False):
                xf = item['first'].unsqueeze(0).to(self.device)
                override = model.prompt_learner(model.check_image_size(xf)[0])
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type=self.device.type, enabled=amp):
                out, p = model(x, prompt_override=override)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            m = suite(out, gt)
            mi = self.light(x, gt)
            n_resized += int(bool(item.get('gt_resized')))
            row = dict(image=item['name'], key=item['key'])
            for k in names:
                row[k] = m[k][0]
            row.update(input_psnr=mi['psnr'][0], input_ssim=mi['ssim'][0], H=x.shape[2], W=x.shape[3], ms=round(ms, 2))
            rows.append(row)
            if collect_prompts and p is not None:
                prompts.append(p.float().cpu().numpy()[0])
            if save_images:
                save_png(out[0], os.path.join(out_dir, 'restored', os.path.splitext(item['name'])[0] + '.png'))
        write_rows(rows, os.path.join(out_dir, 'per_image.csv'))
        summ = dict(run_id=run_id, benchmark=bench, prompt_mode=prompt_mode, n_images=len(rows), n_gt_resized=n_resized,
                    dataset_report=ds.report)
        for k in list(names) + ['input_psnr', 'input_ssim']:
            v = [r[k] for r in rows]
            summ[k] = float(np.mean(v)) if v else None
            summ[k + '_std'] = float(np.std(v)) if v else None
        summ['mean_ms'] = float(np.mean([r['ms'] for r in rows[1:]])) if len(rows) > 1 else None
        write_json(summ, os.path.join(out_dir, 'summary.json'))
        if collect_prompts and prompts:
            np.save(os.path.join(out_dir, 'prompts.npy'), np.stack(prompts))
        rj = os.path.join(self.cfg.run_dir(run_id), 'run.json')
        rec = read_json(rj, {})
        rec.setdefault('eval', {})[bench + tag] = {k: summ[k] for k in summ if k not in ('dataset_report',)}
        write_json(rec, rj)
        print(f'[eval] {run_id:22s} {bench + tag:28s} n={len(rows):5d}  PSNR {summ["psnr"]:.2f}  SSIM {summ["ssim"]:.3f}'
              f'  (input {summ["input_psnr"]:.2f}/{summ["input_ssim"]:.3f})')
        return summ

    # --------------------------------------------------------- competitors
    @torch.no_grad()
    def score_folder(self, bench, method, folder, out_csv):
        """Score restored images of another method with the same metric function.
        Files are matched by input stem, ground-truth stem or pairing key."""
        if os.path.exists(out_csv):
            return read_rows(out_csv)
        if not os.path.isdir(folder):
            print(f'[competitor] {method} on {bench}: folder {folder} missing -> skipped')
            return []
        files = {}
        for f in glob.glob(os.path.join(folder, '**', '*'), recursive=True):
            if f.lower().endswith(IMG_EXT):
                files.setdefault(os.path.splitext(os.path.basename(f))[0].lower(), f)
        ds = BenchmarkDataset(self.cfg, self.roots, bench)
        rows, missing, shape_mismatch = [], 0, 0
        for i in tqdm(range(len(ds)), desc=f'{method} | {bench}', leave=False):
            it = ds[i]
            cands = [os.path.splitext(it['name'])[0].lower(), os.path.splitext(it['gt_name'])[0].lower(), it['key'].lower()]
            f = next((files[c] for c in cands if c in files), None)
            if f is None:
                missing += 1
                continue
            r = TF.to_tensor(Image.open(f).convert('RGB'))
            if r.shape != it['gt'].shape:
                shape_mismatch += 1
                continue
            m = self.metrics(r.unsqueeze(0), it['gt'].unsqueeze(0))
            rows.append(dict(image=it['name'], key=it['key'], **{k: m[k][0] for k in self.metric_names}))
        write_rows(rows, out_csv)
        write_json(dict(method=method, benchmark=bench, folder=folder, n_scored=len(rows), n_missing=missing,
                        n_shape_mismatch=shape_mismatch), out_csv.replace('.csv', '.json'))
        print(f'[competitor] {method:14s} {bench:10s} scored {len(rows)} (missing {missing}, shape mismatch {shape_mismatch})')
        return rows


# ==========================================================================
# Explainability & interpretability suite (figures of the paper), from the
# submitted notebook; now takes an already loaded model, skips modules that an
# ablated model does not have, and runs on a handful of images per benchmark.
# ==========================================================================

class KASIRNetInferencer:
    def __init__(self, model, output_root, metrics, explain_max_size=512, enable_occlusion=False):
        self.device = next(model.parameters()).device
        self.model = model.eval()
        self.metrics_calc = metrics
        self.output_root = output_root
        self.explain_max_size = explain_max_size
        self.enable_occlusion = enable_occlusion

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)

        self._acts = {}
        self._grads = {}
        self._hook_handles = []

        self.global_prompts = []
        self.global_metrics = {}

        os.makedirs(self.output_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Hook Configuration
    # ------------------------------------------------------------------
    def _get_stage_modules(self):
        names = [('head', 'head_attn'), ('encoder1', 'encoder1'), ('encoder2', 'encoder2'), ('encoder3', 'encoder3'),
                 ('encoder4', 'encoder4'), ('bottleneck', 'bottleneck'), ('decoder4', 'decoder4'), ('decoder3', 'decoder3'),
                 ('decoder2', 'decoder2'), ('decoder1', 'decoder1')]
        return {k: getattr(self.model, a) for k, a in names if hasattr(self.model, a)}

    def _get_srb_modules(self):
        return {name: mod for name, mod in self.model.named_modules() if isinstance(mod, SpectralResonanceBlock)}

    def _get_sim_modules(self):
        return {name: mod for name, mod in self.model.named_modules() if isinstance(mod, SemanticInjection)}

    def _get_sft_modules(self):
        return {name: mod for name, mod in self.model.named_modules() if isinstance(mod, SFTLayer)}

    def _register_hooks(self, capture_grads=False):
        self._remove_hooks()
        self._acts, self._grads = {}, {}

        def make_fwd_hook(key):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(t):
                    self._acts[key] = t
                    if capture_grads and t.requires_grad:
                        def grad_hook(grad, k=key):
                            self._grads[k] = grad.detach()
                        t.register_hook(grad_hook)
            return hook

        for name, module in self._get_stage_modules().items():
            self._hook_handles.append(module.register_forward_hook(make_fwd_hook(f'stage/{name}')))
        for name, module in self._get_srb_modules().items():
            self._hook_handles.append(module.resonance_gate.register_forward_hook(make_fwd_hook(f'gate/{name}')))
        for name, module in self._get_sim_modules().items():
            self._hook_handles.append(module.spatial_attn.register_forward_hook(make_fwd_hook(f'sim/{name}')))
            self._hook_handles.append(module.channel_attn.register_forward_hook(make_fwd_hook(f'simch/{name}')))
        for name, module in self._get_sft_modules().items():
            self._hook_handles.append(module.sft_scale.register_forward_hook(make_fwd_hook(f'sft_scale/{name}')))
            self._hook_handles.append(module.sft_shift.register_forward_hook(make_fwd_hook(f'sft_shift/{name}')))

    def _remove_hooks(self):
        for h in self._hook_handles: h.remove()
        self._hook_handles = []

    @torch.no_grad()
    def restore(self, input_tensor):
        input_tensor = input_tensor.to(self.device).unsqueeze(0)
        restored, prompt_emb = self.model(input_tensor)
        if prompt_emb is None:
            prompt_emb = torch.zeros(1, 256, device=restored.device)
        return torch.clamp(restored, 0, 1).squeeze(0).cpu(), prompt_emb.float().squeeze(0).cpu()

    def _resize_for_explain(self, tensor):
        _, H, W = tensor.shape
        m = max(H, W)
        if m <= self.explain_max_size: return tensor
        scale = self.explain_max_size / m
        new_h, new_w = max(16, int(H * scale) // 16 * 16), max(16, int(W * scale) // 16 * 16)
        return torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

    # ------------------------------------------------------------------
    # Utility Savers for Individual Image Extractions
    # ------------------------------------------------------------------
    @staticmethod
    def _to_np(t): return t.detach().permute(1, 2, 0).cpu().numpy().clip(0, 1)

    @staticmethod
    def _normalize_map(m): return (m - m.min()) / (m.max() - m.min() + 1e-8)

    def _overlay_heatmap(self, image_np, heat_np, alpha=0.5, cmap='jet'):
        heat = self._normalize_map(heat_np)
        # Fixed DeprecationWarning: Using plt.get_cmap instead of cm.get_cmap
        heat_rgb = plt.get_cmap(cmap)(heat)[..., :3]
        return (1 - alpha) * image_np + alpha * heat_rgb

    def _save_single_img(self, img_np, save_path, cmap=None):
        if img_np.ndim == 2:
            plt.imsave(save_path, img_np, cmap=cmap)
        else:
            plt.imsave(save_path, np.clip(img_np, 0, 1))

    # ------------------------------------------------------------------
    # V1. Comparison
    # ------------------------------------------------------------------
    def vis_comparison(self, input_t, restored_t, gt_t, metrics, save_path, task_name):
        panels = [self._to_np(input_t), self._to_np(restored_t)]
        titles = [f"Input ({task_name})", f"Restored (PSNR: {metrics.get('psnr','N/A')})"]
        if gt_t is not None:
            panels.append(self._to_np(gt_t))
            titles.append("Ground Truth")
            err = np.abs(self._to_np(restored_t) - self._to_np(gt_t)).mean(axis=2)
            panels.append(err)
            titles.append(f"|Error|")

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, img, title in zip(np.atleast_1d(axes), panels, titles):
            if img.ndim == 2:
                im = ax.imshow(img, cmap='inferno')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        self._save_single_img(panels[0], os.path.join(base_dir, f"{stem}_input.png"))
        self._save_single_img(panels[1], os.path.join(base_dir, f"{stem}_restored.png"))
        if gt_t is not None:
            self._save_single_img(panels[2], os.path.join(base_dir, f"{stem}_gt.png"))
            self._save_single_img(panels[3], os.path.join(base_dir, f"{stem}_errormap.png"), cmap='inferno')

    # ------------------------------------------------------------------
    # V2. Error Maps
    # ------------------------------------------------------------------
    def vis_error_maps(self, restored_t, gt_t, save_path):
        if gt_t is None: return
        r, g = self._to_np(restored_t), self._to_np(gt_t)
        err = np.abs(r - g)

        with torch.no_grad():
            rt, gt_tnsr = restored_t.unsqueeze(0).to(self.device), gt_t.unsqueeze(0).to(self.device)
            rl, gl = rt.mean(dim=1, keepdim=True), gt_tnsr.mean(dim=1, keepdim=True)
            k, C1, C2 = 11, 0.01 ** 2, 0.03 ** 2
            mu_r = torch.nn.functional.avg_pool2d(rl, k, 1, k // 2)
            mu_g = torch.nn.functional.avg_pool2d(gl, k, 1, k // 2)
            s_r = torch.nn.functional.avg_pool2d(rl * rl, k, 1, k // 2) - mu_r ** 2
            s_g = torch.nn.functional.avg_pool2d(gl * gl, k, 1, k // 2) - mu_g ** 2
            s_rg = torch.nn.functional.avg_pool2d(rl * gl, k, 1, k // 2) - mu_r * mu_g
            ssim_map = ((2 * mu_r * mu_g + C1) * (2 * s_rg + C2)) / ((mu_r ** 2 + mu_g ** 2 + C1) * (s_r + s_g + C2))
            ssim_map = ssim_map.squeeze().cpu().numpy()

        maps = [err.mean(axis=2), err[..., 0], err[..., 1], err[..., 2], 1 - ssim_map]
        titles = ["Mean |Error|", "Error R", "Error G", "Error B", "1 - SSIM map"]
        cmaps = ['inferno', 'Reds', 'Greens', 'Blues', 'magma']

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for ax, m, t, cmp_ in zip(axes, maps, titles, cmaps):
            im = ax.imshow(m, cmap=cmp_)
            ax.set_title(t, fontsize=10); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        for i, suffix in enumerate(['mean_err', 'R_err', 'G_err', 'B_err', 'inv_ssim']):
            self._save_single_img(maps[i], os.path.join(base_dir, f"{stem}_{suffix}.png"), cmap=cmaps[i])

    # ------------------------------------------------------------------
    # V3. Grad-CAM
    # ------------------------------------------------------------------
    def vis_gradcam(self, input_t, gt_t, save_path):
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        gt_small = None
        if gt_t is not None:
            gt_small = torch.nn.functional.interpolate(gt_t.unsqueeze(0).to(self.device), size=small.shape[2:], mode='bilinear', align_corners=False)

        self._register_hooks(capture_grads=True)
        self.model.zero_grad()

        with torch.enable_grad():
            restored, _ = self.model(small)
            if gt_small is not None:
                scalar = torch.nn.functional.l1_loss(restored, gt_small)
            else:
                scalar = restored.mean()
            scalar.backward()

        targets = ['stage/bottleneck', 'stage/decoder4', 'stage/decoder3', 'stage/decoder2', 'stage/decoder1']
        input_np = self._to_np(small.squeeze(0))
        cams, names = [], []

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem

        for key in targets:
            if key not in self._acts or key not in self._grads: continue
            act, grad = self._acts[key], self._grads[key]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1)).squeeze(0)
            cam = torch.nn.functional.interpolate(cam[None, None], size=small.shape[2:], mode='bilinear', align_corners=False).squeeze().detach().cpu().numpy()
            cams.append(cam)

            name = key.split('/')[-1]
            names.append(name)

            self._save_single_img(self._normalize_map(cam), os.path.join(base_dir, f"{stem}_{name}_gradcam.png"), cmap='jet')
            self._save_single_img(self._overlay_heatmap(input_np, cam), os.path.join(base_dir, f"{stem}_{name}_overlay.png"))

        self._remove_hooks()
        self.model.zero_grad()

        if not cams: return
        n = len(cams) + 1
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        axes[0, 0].imshow(input_np); axes[0, 0].set_title("Input"); axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        for i, (cam, name) in enumerate(zip(cams, names)):
            axes[0, i + 1].imshow(self._normalize_map(cam), cmap='jet'); axes[0, i + 1].set_title(f"Grad-CAM: {name}", fontsize=10); axes[0, i + 1].axis('off')
            axes[1, i + 1].imshow(self._overlay_heatmap(input_np, cam)); axes[1, i + 1].set_title(f"Overlay: {name}", fontsize=10); axes[1, i + 1].axis('off')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

    # ------------------------------------------------------------------
    # V4. Feature Maps
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_feature_maps(self, input_t, save_path, topk=8):
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        self._register_hooks(capture_grads=False)
        _ = self.model(small)
        stage_keys = sorted([k for k in self._acts if k.startswith('stage/')])
        self._remove_hooks()

        n_stages = len(stage_keys)
        fig, axes = plt.subplots(n_stages, topk + 1, figsize=(2.2 * (topk + 1), 2.2 * n_stages))
        if n_stages == 1: axes = axes[None, :]
        input_np = self._to_np(small.squeeze(0))

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem

        for r, key in enumerate(stage_keys):
            feat = self._acts[key].squeeze(0)
            energy = feat.abs().mean(dim=(1, 2))
            idx = torch.argsort(energy, descending=True)[:topk]

            stage_name = key.split('/')[-1]
            axes[r, 0].imshow(input_np); axes[r, 0].set_ylabel(stage_name, fontsize=10, fontweight='bold'); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

            for c, ci in enumerate(idx):
                fmap = feat[ci].cpu().numpy()
                axes[r, c + 1].imshow(self._normalize_map(fmap), cmap='viridis')
                axes[r, c + 1].set_title(f"ch{ci.item()}", fontsize=7); axes[r, c + 1].axis('off')

                self._save_single_img(self._normalize_map(fmap), os.path.join(base_dir, f"{stem}_{stage_name}_ch{ci.item()}.png"), cmap='viridis')

        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=110); plt.close(fig)

    # ------------------------------------------------------------------
    # V5. Spectral Analysis
    # ------------------------------------------------------------------
    def vis_spectral_analysis(self, input_t, restored_t, gt_t, save_path):
        def log_spectrum(t):
            gray = t.mean(dim=0)
            f = torch.fft.fftshift(torch.fft.fft2(gray))
            return torch.log1p(torch.abs(f)).cpu().numpy()

        def radial_profile(spec):
            h, w = spec.shape
            cy, cx = h // 2, w // 2
            y, x = np.indices((h, w))
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
            tbin = np.bincount(r.ravel(), spec.ravel())
            nr = np.bincount(r.ravel())
            return tbin / np.maximum(nr, 1)

        tensors = {'Input': input_t, 'Restored': restored_t}
        if gt_t is not None: tensors['Ground Truth'] = gt_t

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        profiles = {}

        n = len(tensors)
        fig = plt.figure(figsize=(5 * n + 6, 5))
        gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.4])

        for i, (name, t) in enumerate(tensors.items()):
            spec = log_spectrum(t)
            profiles[name] = radial_profile(spec)
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(spec, cmap='magma')
            ax.set_title(f"log|FFT| — {name}", fontsize=11); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            self._save_single_img(self._normalize_map(spec), os.path.join(base_dir, f"{stem}_fft_{name.replace(' ', '_')}.png"), cmap='magma')

        ax_p = fig.add_subplot(gs[0, n])
        colors = {'Input': 'tab:red', 'Restored': 'tab:blue', 'Ground Truth': 'tab:green'}
        for name, prof in profiles.items():
            ax_p.plot(prof[:min(len(prof), 256)], label=name, color=colors.get(name), lw=1.5)
        ax_p.set_yscale('log'); ax_p.legend(); ax_p.grid(alpha=0.3)
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        fig_prof, ax_prof = plt.subplots(figsize=(6, 4))
        for name, prof in profiles.items():
            ax_prof.plot(prof[:min(len(prof), 256)], label=name, color=colors.get(name), lw=1.5)
        ax_prof.set_yscale('log'); ax_prof.legend(); ax_prof.grid(alpha=0.3)
        fig_prof.tight_layout()
        fig_prof.savefig(os.path.join(base_dir, f"{stem}_radial_profile.png"), dpi=130); plt.close(fig_prof)

    # ------------------------------------------------------------------
    # V6. Resonance Gates
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_resonance_gates(self, input_t, save_path):
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        self._register_hooks(capture_grads=False)
        _ = self.model(small)
        gate_keys = sorted([k for k in self._acts if k.startswith('gate/')])
        gates = {k.replace('gate/', ''): self._acts[k].squeeze().cpu().numpy() for k in gate_keys}
        self._remove_hooks()
        if not gates: return

        n = len(gates)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.2 * rows))
        axes = np.atleast_2d(axes)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem

        for i, (name, g) in enumerate(gates.items()):
            ax = axes[i // cols, i % cols]
            ax.bar(np.arange(len(g)), np.sort(g)[::-1], color='steelblue', width=1.0)
            ax.axhline(g.mean(), color='red', ls='--', lw=1)
            ax.set_title(name.replace('.', ' → '), fontsize=8); ax.set_ylim(0, 1)

            fig_bar, ax_bar = plt.subplots(figsize=(5,3))
            ax_bar.bar(np.arange(len(g)), np.sort(g)[::-1], color='steelblue', width=1.0)
            ax_bar.axhline(g.mean(), color='red', ls='--', lw=1)
            ax_bar.set_ylim(0, 1); ax_bar.set_title(name)
            fig_bar.tight_layout()
            fig_bar.savefig(os.path.join(base_dir, f"{stem}_gate_{name.replace('.', '_')}.png"), dpi=110); plt.close(fig_bar)

        for j in range(n, rows * cols): axes[j // cols, j % cols].axis('off')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=110); plt.close(fig)

    # ------------------------------------------------------------------
    # V7. Semantic Attention
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_semantic_attention(self, input_t, save_path):
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        self._register_hooks(capture_grads=False)
        _ = self.model(small)
        sim_keys = sorted([k for k in self._acts if k.startswith('sim/')])
        maps = {}
        for k in sim_keys:
            m = self._acts[k]
            m = torch.nn.functional.interpolate(m, size=small.shape[2:], mode='bilinear', align_corners=False)
            maps[k.replace('sim/', '')] = m.squeeze().cpu().numpy()
        self._remove_hooks()
        if not maps: return

        input_np = self._to_np(small.squeeze(0))
        n = len(maps)
        cols = 4
        rows = int(np.ceil((n + 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
        axes = np.atleast_2d(axes)
        axes[0, 0].imshow(input_np); axes[0, 0].set_title("Input"); axes[0, 0].axis('off')

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem

        for i, (name, m) in enumerate(maps.items(), start=1):
            ax = axes[i // cols, i % cols]
            overlay = self._overlay_heatmap(input_np, m)
            ax.imshow(overlay); ax.set_title(f"{name}", fontsize=8); ax.axis('off')

            self._save_single_img(overlay, os.path.join(base_dir, f"{stem}_sim_{name}.png"))

        for j in range(n + 1, rows * cols): axes[j // cols, j % cols].axis('off')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=110); plt.close(fig)

    # ------------------------------------------------------------------
    # V8. Edges
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_edges(self, input_t, restored_t, gt_t, save_path):
        def edges(t):
            t = t.unsqueeze(0).to(self.device)
            ex = torch.nn.functional.conv2d(t, self.sobel_x, padding=1, groups=3)
            ey = torch.nn.functional.conv2d(t, self.sobel_y, padding=1, groups=3)
            return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6).mean(dim=1).squeeze(0).cpu().numpy()

        e_in, e_re = edges(input_t), edges(restored_t)
        panels = [e_in, e_re]
        titles = [f"Input edges", f"Restored edges"]
        if gt_t is not None:
            e_gt = edges(gt_t)
            panels += [e_gt, np.abs(e_re - e_gt)]
            titles += [f"GT edges", f"|Restored - GT| edge error"]

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, p, t in zip(np.atleast_1d(axes), panels, titles):
            im = ax.imshow(p, cmap='gray' if 'error' not in t else 'inferno')
            ax.set_title(t, fontsize=10); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        keys = ['input', 'restored', 'gt', 'error']
        for i, p in enumerate(panels):
            cmap_arg = 'inferno' if keys[i] == 'error' else 'gray'
            self._save_single_img(p, os.path.join(base_dir, f"{stem}_edge_{keys[i]}.png"), cmap=cmap_arg)

    # ------------------------------------------------------------------
    # V9. Histograms
    # ------------------------------------------------------------------
    def vis_histograms(self, input_t, restored_t, gt_t, save_path):
        tensors = {'Input': input_t, 'Restored': restored_t}
        if gt_t is not None: tensors['Ground Truth'] = gt_t
        colors3 = ['red', 'green', 'blue']

        fig, axes = plt.subplots(1, len(tensors) + 1, figsize=(5.5 * (len(tensors) + 1), 4.5))

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem

        for i, (name, t) in enumerate(tensors.items()):
            arr = self._to_np(t)

            fig_h, ax_h = plt.subplots()
            for c in range(3):
                axes[i].hist(arr[..., c].ravel(), bins=64, color=colors3[c], alpha=0.45, density=True)
                ax_h.hist(arr[..., c].ravel(), bins=64, color=colors3[c], alpha=0.45, density=True)
            axes[i].set_title(f"RGB histogram — {name}", fontsize=11); axes[i].grid(alpha=0.3)

            fig_h.tight_layout()
            fig_h.savefig(os.path.join(base_dir, f"{stem}_hist_{name.replace(' ', '_')}.png"), dpi=130)
            plt.close(fig_h)

        ls = {'Input': 'tab:red', 'Restored': 'tab:blue', 'Ground Truth': 'tab:green'}
        for name, t in tensors.items():
            lum = self._to_np(t).mean(axis=2).ravel()
            xs = np.sort(lum)
            axes[-1].plot(xs, np.linspace(0, 1, len(xs)), label=name, color=ls.get(name), lw=1.5)
        axes[-1].legend(); axes[-1].grid(alpha=0.3)
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

    # ------------------------------------------------------------------
    # V10. Prompts
    # ------------------------------------------------------------------
    def vis_prompt(self, prompt_emb, save_path, task_name):
        p = prompt_emb.numpy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        axes[0].bar(np.arange(len(p)), p, color='skyblue', width=1.0)
        axes[0].set_title(f"Degradation Prompt ({len(p)}D)", fontsize=11); axes[0].grid(alpha=0.3, axis='y')

        top = np.argsort(np.abs(p))[::-1][:20]
        axes[1].barh(np.arange(len(top)), p[top], color=['tab:blue' if v >= 0 else 'tab:red' for v in p[top]])
        axes[1].set_yticks(np.arange(len(top))); axes[1].set_yticklabels([f"dim {d}" for d in top], fontsize=7)
        axes[1].invert_yaxis(); axes[1].grid(alpha=0.3, axis='x')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        fig_b, ax_b = plt.subplots(figsize=(8,3))
        ax_b.bar(np.arange(len(p)), p, color='skyblue', width=1.0)
        fig_b.tight_layout(); fig_b.savefig(os.path.join(base_dir, f"{stem}_prompt_bar.png"), dpi=130); plt.close(fig_b)

    # ------------------------------------------------------------------
    # V11. Zoom Crops
    # ------------------------------------------------------------------
    def vis_zoom_crops(self, input_t, restored_t, gt_t, save_path, crop=96):
        if gt_t is None: return
        err = np.abs(self._to_np(restored_t) - self._to_np(gt_t)).mean(axis=2)
        H, W = err.shape
        crop = max(2, min(crop, H, W))
        eh = torch.nn.functional.avg_pool2d(torch.from_numpy(err)[None, None].float(), crop, stride=crop // 2)
        idx = torch.argmax(eh.flatten()).item()
        gw = eh.shape[-1]
        by, bx = idx // gw, idx % gw
        top = min(by * (crop // 2), max(0, H - crop))
        left = min(bx * (crop // 2), max(0, W - crop))

        def cut(t): return self._to_np(t)[top:top + crop, left:left + crop]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, img, title in zip(axes, [cut(input_t), cut(restored_t), cut(gt_t), err[top:top + crop, left:left + crop]], ["Input crop", "Restored crop", "GT crop", "Error crop"]):
            if img.ndim == 2: im = ax.imshow(img, cmap='inferno'); plt.colorbar(im, ax=ax, fraction=0.046)
            else: ax.imshow(img)
            ax.set_title(title, fontsize=11); ax.axis('off')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        self._save_single_img(cut(input_t), os.path.join(base_dir, f"{stem}_zoom_input.png"))
        self._save_single_img(cut(restored_t), os.path.join(base_dir, f"{stem}_zoom_restored.png"))
        self._save_single_img(cut(gt_t), os.path.join(base_dir, f"{stem}_zoom_gt.png"))
        self._save_single_img(err[top:top + crop, left:left + crop], os.path.join(base_dir, f"{stem}_zoom_error.png"), cmap='inferno')

    # ------------------------------------------------------------------
    # V12. Occlusion Maps
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_occlusion(self, input_t, gt_t, save_path, patch=64, stride=64):
        if gt_t is None: return
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        gt_small = torch.nn.functional.interpolate(gt_t.unsqueeze(0).to(self.device), size=small.shape[2:], mode='bilinear', align_corners=False)
        base, _ = self.model(small)
        base_err = torch.nn.functional.l1_loss(base, gt_small).item()

        _, _, H, W = small.shape
        heat = np.zeros(((H - 1) // stride + 1, (W - 1) // stride + 1))
        mean_val = small.mean()
        for iy, y in enumerate(range(0, H, stride)):
            for ix, x in enumerate(range(0, W, stride)):
                occluded = small.clone()
                occluded[:, :, y:y + patch, x:x + patch] = mean_val
                out, _ = self.model(occluded)
                heat[iy, ix] = torch.nn.functional.l1_loss(out, gt_small).item() - base_err

        heat_full = torch.nn.functional.interpolate(torch.from_numpy(heat)[None, None].float(), size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()
        input_np = self._to_np(small.squeeze(0))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(input_np); axes[0].set_title("Input"); axes[0].axis('off')
        im = axes[1].imshow(heat_full, cmap='jet')
        axes[1].set_title(f"Occlusion ΔL1"); axes[1].axis('off'); plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[2].imshow(self._overlay_heatmap(input_np, heat_full)); axes[2].set_title("Overlay"); axes[2].axis('off')
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        self._save_single_img(heat_full, os.path.join(base_dir, f"{stem}_occlusion_heatmap.png"), cmap='jet')
        self._save_single_img(self._overlay_heatmap(input_np, heat_full), os.path.join(base_dir, f"{stem}_occlusion_overlay.png"))

    # ------------------------------------------------------------------
    # V13. SFT Layer Modulation Output
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vis_sft_modulation(self, input_t, save_path):
        small = self._resize_for_explain(input_t).to(self.device).unsqueeze(0)
        self._register_hooks(capture_grads=False)
        _ = self.model(small)

        sft_scale_keys = sorted([k for k in self._acts if k.startswith('sft_scale/')])
        sft_shift_keys = sorted([k for k in self._acts if k.startswith('sft_shift/')])

        scales, shifts = {}, {}
        for k_sc, k_sh in zip(sft_scale_keys, sft_shift_keys):
            name = k_sc.replace('sft_scale/', '')
            scales[name] = self._acts[k_sc].mean(dim=1).squeeze().cpu().numpy()
            shifts[name] = self._acts[k_sh].mean(dim=1).squeeze().cpu().numpy()

        self._remove_hooks()
        if not scales: return

        n = len(scales)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1: axes = axes[:, None]
        for i, name in enumerate(scales.keys()):
            sc_map, sh_map = scales[name], shifts[name]
            im1 = axes[0, i].imshow(sc_map, cmap='coolwarm'); axes[0, i].set_title(f"SFT Scale: {name}", fontsize=10); axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
            im2 = axes[1, i].imshow(sh_map, cmap='viridis'); axes[1, i].set_title(f"SFT Shift: {name}", fontsize=10); axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

        base_dir, stem = os.path.dirname(save_path), Path(save_path).stem
        for name in scales.keys():
            self._save_single_img(scales[name], os.path.join(base_dir, f"{stem}_{name}_sft_scale.png"), cmap='coolwarm')
            self._save_single_img(shifts[name], os.path.join(base_dir, f"{stem}_{name}_sft_shift.png"), cmap='viridis')

    # ------------------------------------------------------------------
    # Global Reports
    # ------------------------------------------------------------------
    def vis_prompt_projection(self, prompts, labels, save_path, title):
        if len(prompts) < 3: return
        X = np.stack(prompts)
        method = "PCA"
        try:
            if len(prompts) >= 8:
                from sklearn.manifold import TSNE
                perp = max(2, min(30, len(prompts) - 1))
                X2 = TSNE(n_components=2, perplexity=perp, init='pca', learning_rate='auto', random_state=42).fit_transform(X)
                method = f"t-SNE (perplexity={perp})"
            else:
                raise ValueError("too few samples for t-SNE")
        except Exception:
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2, random_state=42).fit_transform(X)
            method = "PCA"

        uniq = sorted(set(labels))
        # Fixed DeprecationWarning: Using plt.get_cmap instead of cm.get_cmap
        cmap = plt.get_cmap('tab10', max(len(uniq), 1))
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, u in enumerate(uniq):
            mask = np.array([l == u for l in labels])
            ax.scatter(X2[mask, 0], X2[mask, 1], s=40, alpha=0.8, color=cmap(i), label=u, edgecolors='k', linewidths=0.3)
        ax.set_title(f"{title}\n{method} projection of Degradation Prompts", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

    def vis_prompt_heatmap(self, prompts, names, save_path, title):
        if not prompts: return
        X = np.stack(prompts)
        fig, ax = plt.subplots(figsize=(14, max(3, 0.28 * len(prompts))))
        im = ax.imshow(X, aspect='auto', cmap='coolwarm', vmin=-np.abs(X).max(), vmax=np.abs(X).max())
        ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=6); ax.set_xlabel("prompt dimension")
        ax.set_title(f"{title} — Prompt heatmap (rows=images)", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout(); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

    def vis_metrics_summary(self, metric_rows, save_path, title):
        if not metric_rows: return
        keys = ['psnr', 'ssim', 'lpips']
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        for ax, k in zip(axes, keys):
            vals = [float(m[k]) for m in metric_rows if m.get(k, 'N/A') != 'N/A']
            if not vals: ax.axis('off'); continue
            ax.plot(vals, 'o-', ms=4, lw=1, color='steelblue')
            ax.axhline(np.mean(vals), color='red', ls='--', label=f"mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")
            ax.set_title(k.upper(), fontsize=11); ax.set_xlabel("image index"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f"{title} — Per-image metrics", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig(save_path, bbox_inches='tight', dpi=130); plt.close(fig)

    # ------------------------------------------------------------------
    # Core Image Explainer Routine
    # ------------------------------------------------------------------
    def explain_image(self, input_tensor, gt_tensor, stem, dataset_dirs, task_name):

        restored_tensor, prompt_emb = self.restore(input_tensor)

        metrics = {'psnr': 'N/A', 'ssim': 'N/A', 'lpips': 'N/A'}
        if gt_tensor is not None:
            with torch.no_grad():
                r, g = restored_tensor.unsqueeze(0).to(self.device), gt_tensor.unsqueeze(0).to(self.device)
                m = self.metrics_calc(r, g)
                for k in ['psnr', 'ssim', 'lpips']:
                    if k in m:
                        metrics[k] = f"{m[k][0]:.4f}"

        try:
            self.vis_comparison(input_tensor, restored_tensor, gt_tensor, metrics, os.path.join(dataset_dirs['01_comparison'], f"{stem}.png"), task_name)
            self.vis_error_maps(restored_tensor, gt_tensor, os.path.join(dataset_dirs['02_error_maps'], f"{stem}.png"))
            self.vis_gradcam(input_tensor, gt_tensor, os.path.join(dataset_dirs['03_gradcam'], f"{stem}.png"))
            self.vis_feature_maps(input_tensor, os.path.join(dataset_dirs['04_feature_maps'], f"{stem}.png"))
            self.vis_spectral_analysis(input_tensor, restored_tensor, gt_tensor, os.path.join(dataset_dirs['05_spectral_analysis'], f"{stem}.png"))
            self.vis_resonance_gates(input_tensor, os.path.join(dataset_dirs['06_resonance_gates'], f"{stem}.png"))
            self.vis_semantic_attention(input_tensor, os.path.join(dataset_dirs['07_semantic_attention'], f"{stem}.png"))
            self.vis_edges(input_tensor, restored_tensor, gt_tensor, os.path.join(dataset_dirs['08_edge_analysis'], f"{stem}.png"))
            self.vis_histograms(input_tensor, restored_tensor, gt_tensor, os.path.join(dataset_dirs['09_histograms'], f"{stem}.png"))
            self.vis_prompt(prompt_emb, os.path.join(dataset_dirs['10_prompt_analysis'], f"{stem}.png"), task_name)
            self.vis_zoom_crops(input_tensor, restored_tensor, gt_tensor, os.path.join(dataset_dirs['11_zoom_crops'], f"{stem}.png"))
            self.vis_sft_modulation(input_tensor, os.path.join(dataset_dirs['13_sft_modulation'], f"{stem}.png"))
            if self.enable_occlusion:
                self.vis_occlusion(input_tensor, gt_tensor, os.path.join(dataset_dirs['12_occlusion'], f"{stem}.png"))
        except Exception as e:
            print(f"  [warn] visualization failed for {stem}: {e}")
        finally:
            self._remove_hooks()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return metrics, prompt_emb.numpy()

    def run_dataset_inference(self, dataset_name, items):
        """items: list of dicts with 'input', 'gt' (tensors) and 'name'."""
        dataset_root = os.path.join(self.output_root, dataset_name)
        sub_names = ['01_comparison', '02_error_maps', '03_gradcam', '04_feature_maps',
                     '05_spectral_analysis', '06_resonance_gates', '07_semantic_attention',
                     '08_edge_analysis', '09_histograms', '10_prompt_analysis',
                     '11_zoom_crops', '13_sft_modulation']
        if self.enable_occlusion: sub_names.append('12_occlusion')
        dataset_dirs = {}
        for s in sub_names:
            d = os.path.join(dataset_root, s)
            os.makedirs(d, exist_ok=True)
            dataset_dirs[s] = d
        metric_rows, prompts, names = [], [], []
        for it in tqdm(items, desc=f"Explaining [{dataset_name}]", leave=False):
            stem = Path(it['name']).stem
            try:
                metrics, prompt = self.explain_image(it['input'], it['gt'], stem, dataset_dirs, dataset_name)
                metric_rows.append({'image': it['name'], **metrics})
                prompts.append(prompt)
                names.append(stem)
                self.global_prompts.append((dataset_name, prompt))
            except Exception as e:
                print(f"  [warn] explain failed for {stem}: {e}")

        if metric_rows:
            csv_path = os.path.join(dataset_root, 'metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image', 'psnr', 'ssim', 'lpips'])
                writer.writeheader(); writer.writerows(metric_rows)

        self.vis_metrics_summary(metric_rows, os.path.join(dataset_root, 'metrics_summary.png'), dataset_name)
        self.vis_prompt_projection(prompts, [dataset_name] * len(prompts), os.path.join(dataset_root, 'prompt_projection.png'), dataset_name)
        self.vis_prompt_heatmap(prompts, names, os.path.join(dataset_root, 'prompt_heatmap.png'), dataset_name)

        self.global_metrics[dataset_name] = metric_rows
        return metric_rows

    def finalize_global_reports(self):
        if len(self.global_prompts) >= 3:
            labels = [d for d, _ in self.global_prompts]
            prompts = [p for _, p in self.global_prompts]
            self.vis_prompt_projection(prompts, labels, os.path.join(self.output_root, 'all_datasets_prompt_projection.png'), "All datasets")

        if self.global_metrics:
            keys = ['psnr', 'ssim', 'lpips']
            names = list(self.global_metrics.keys())
            fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5))
            for ax, k in zip(axes, keys):
                means, stds = [], []
                for n in names:
                    vals = [float(m[k]) for m in self.global_metrics[n] if m.get(k, 'N/A') != 'N/A']
                    means.append(np.mean(vals) if vals else 0)
                    stds.append(np.std(vals) if vals else 0)
                ax.bar(names, means, yerr=stds, capsize=4, color='steelblue', alpha=0.85)
                ax.set_title(k.upper(), fontsize=12); ax.tick_params(axis='x', rotation=30); ax.grid(alpha=0.3, axis='y')
                for x, m in zip(range(len(names)), means):
                    ax.text(x, m, f"{m:.3f}", ha='center', va='bottom', fontsize=8)
            fig.suptitle("Cross-Dataset Performance Comparison", fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            path = os.path.join(self.output_root, 'all_datasets_metrics_comparison.png')
            plt.savefig(path, bbox_inches='tight', dpi=130); plt.close(fig)
            print(f"Global comparison saved: {path}")
