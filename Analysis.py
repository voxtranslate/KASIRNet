# ==========================================================================
# KASIRNet - analyses requested by the reviewers (TODO items of the revision)
# --------------------------------------------------------------------------
#   step0_audit            TODO-01/03/04/05/13/15 facts (metric, SSIM-loss gradient, gate
#                          initialisation, submitted checkpoint, data pairing / leakage)
#   cost_report            TODO-13  parameters, MACs, peak memory, latency (FP32/FP16)
#   significance_report    TODO-12C bootstrap 95 % CI, paired Wilcoxon (+ Holm) vs competitors
#                          TODO-14  LPIPS / DISTS and "Divergent" percentage
#   prompt_probe_report    TODO-16  logistic-regression probe on p (5-fold CV) + mixed sets
#   severity_report        TODO-09A SIM vs severity  (+ frozen-feature stability per level)
#   spectral_report        R3-2/R5-2 amplitude-vs-phase attribution, SRB phase deviation, gates
#   qualitative_figures    TODO-06  supplementary Figs. S1 (Rain100L) and S2 (Snow100K-L)
#   ablation_tables        TODO-09B/12A/12B/15 + Tables 6/7 + SCWAM design
#   main_tables            Tables 4/5 (8 benchmarks), Rain100H (TODO-02), zero-shot (TODO-17)
#   section41_report       Section 4.1 reproducibility facts collected from run.json files
# Every function writes CSV (machine-readable) and a LaTeX-ready .tex fragment
# (PSNR two decimals, SSIM three decimals) under OUTPUT_ROOT/tables or /analysis.
# ==========================================================================
import os
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Datasets import BenchmarkDataset, resolve_path, list_images, pair_files, LeakageAuditor  # nb:skip
from Model import (KASIRNet, SpectralResonanceBlock, PlainConvBlock, WindowAttention, SEMANTIC_EXTRACTORS,  # nb:skip
                   component_parameter_table)  # nb:skip
from Loss import VGGFeatureExtractor, IdentityFeature  # nb:skip
from metrics import PROTOCOL, ssim_y  # nb:skip
from Trainer import build_model, load_weights, read_json, write_json, environment_info  # nb:skip
from Inferencer import read_rows, write_rows  # nb:skip


# ==========================================
# 0. small helpers
# ==========================================

def fmt(v, d=2):
    return '--' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{d}f}'


def write_tex(rows, cols, path, digits):
    """cols: list of keys; digits: {key: decimals}.  One LaTeX row per dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('% ' + ' & '.join(cols) + '\n')
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(fmt(v, digits[c]) if (c in digits and isinstance(v, (int, float))) else str(v if v is not None else '--'))
            f.write(' & '.join(cells) + ' \\\\\n')


def save_table(rows, name, cfg, cols=None, digits=None):
    if not rows:
        print(f'[table] {name}: nothing to write yet')
        return
    os.makedirs(cfg.tables_dir, exist_ok=True)
    write_rows(rows, os.path.join(cfg.tables_dir, name + '.csv'))
    cols = cols or list(rows[0].keys())
    digits = digits or {c: (3 if 'ssim' in c.lower() or 'lpips' in c.lower() or 'dists' in c.lower() else 2) for c in cols}
    write_tex(rows, cols, os.path.join(cfg.tables_dir, name + '.tex'), digits)
    print(f'[table] {name}: {len(rows)} rows -> {cfg.tables_dir}')


def per_image(cfg, run_id, bench, tag=''):
    return read_rows(os.path.join(cfg.run_dir(run_id), 'eval', bench + tag, 'per_image.csv'))


def run_eval(cfg, run_id, bench, key='psnr'):
    rec = read_json(os.path.join(cfg.run_dir(run_id), 'run.json'), {})
    return rec.get('eval', {}).get(bench, {}).get(key)


# ==========================================
# 1. statistics
# ==========================================

def bootstrap_mean_ci(x, n=10000, seed=7, alpha=0.05):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return (float('nan'),) * 3
    rng = np.random.default_rng(seed)
    means = []
    chunk = max(1, min(n, 20_000_000 // max(1, len(x))))
    done = 0
    while done < n:
        k = min(chunk, n - done)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        means.append(x[idx].mean(1))
        done += k
    means = np.concatenate(means)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a, b, n_boot=10000, seed=7):
    """Two-sided Wilcoxon signed-rank on paired per-image values a (ours) vs b."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a - b
    res = dict(n=len(d), mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(d.mean()),
               median_diff=float(np.median(d)), win_rate=float((d > 0).mean()))
    m, lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    res.update(diff_ci_low=lo, diff_ci_high=hi)
    if len(d) >= 6 and np.any(d != 0):
        w = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        res.update(wilcoxon_W=float(w.statistic), p_value=float(w.pvalue))
        nz = d[d != 0]
        r = np.argsort(np.argsort(np.abs(nz))) + 1.0
        res['rank_biserial'] = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
    else:
        res.update(wilcoxon_W=None, p_value=None, rank_biserial=None)
    return res


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    order = sorted(idx, key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvals[i]))
        adj[i] = running
    return adj


def join_by_key(rows_a, rows_b):
    mb = {r['key'] if isinstance(r['key'], str) else str(r['key']): r for r in rows_b}
    out = []
    for r in rows_a:
        k = r['key'] if isinstance(r['key'], str) else str(r['key'])
        if k in mb:
            out.append((r, mb[k]))
    return out


# ==========================================
# 2. Step 0 - facts (no training)
# ==========================================

SUBMITTED_CODE_FACTS = {   # read from the submitted notebook/config (commit 1d4282b)
    'loss_weights_used_by_notebook': 'pix 5.0, fft 1.0 (HYPERPARAMS fft_weight), perc 1.0, ssim 1.0, id 1.0, edge 1.0 (constructor default, '
                                     'edge_weight not passed) -- the manuscript states fft 0.5 and edge 0.5',
    'ssim_loss': 'pyiqa.create_metric("ssim") without as_loss=True: pyiqa evaluates it under torch.set_grad_enabled(False), '
                 'so the SSIM term added a constant and no gradient (see check below)',
    'frozen_networks': 'models.vgg19(weights=None) in the SIM extractor and in the perceptual loss, models.resnet50(weights=None) '
                       'in the identity loss: no ImageNet weights were loaded by the released code; the identity-loss ResNet-50 '
                       'was left in train mode (BatchNorm batch statistics)',
    'optimisation': 'HYPERPARAMS batch_size 64, epochs 3000, Adam lr 1e-4, CosineAnnealingLR(T_max=3000) stepped once per epoch, '
                    'clip_grad_norm 1.0, no autocast/GradScaler (FP32 training)',
    'model_selection': 'best.pth = highest mean PSNR on centre 256x256 crops of GoPro test + DeRaindrop test_b + RESIDE-6K test '
                       '(test images used for checkpoint selection)',
    'inference': 'full-resolution forward, pyiqa psnr/ssim/lpips per image, but at most the first 256 sorted images of each test '
                 'folder (max_images=256); ground truth resized to the input size when they differ',
    'data_rainy': 'JRDR rain_data_train_Heavy (= RainTrainH): input=norain/ (clean) and gt=rain/X2/ (rainy) -> direction swapped; '
                  'pairs formed by zipping two independently sorted lists -> scene ids differ in all 1800 pairs',
    'data_raining': 'adeshpal/image-deraining Dataset/input (rain-001..080) = images 1-80 of the Rain100L test set',
    'data_kohler': 'training category "blurred" = kohlerdeblur/kohler/blur|sharp (all 48 pairs); Config.py TEST_DATA_CONFIG '
                   'kohler_blur = the same folders; the notebook test entry is empty',
    'data_snow_sidd': 'DATA_CONFIG snow has empty paths (no snow training data); no SIDD path in any configuration; the snow '
                      'test folder came from a private gdown archive (/content/input, /content/gt)',
    'data_hide': 'DATA_CONFIG hide = DBlur/HIDE/train, a folder that does not exist in the DBlur collection (HIDE is test-only '
                 'there, 2025 pairs); the loader silently skipped empty folders, so no HIDE image was trained on and the '
                 'submitted pool had 12 non-empty categories (hide and snow empty)',
    'srb_gate': 'resonance_gate ends with Sigmoid and is not zero-initialised (G ~ 0.5 at initialisation, see check below)',
}


def step0_audit(cfg, roots=None, device='cpu'):
    out_dir = cfg.analysis_dir('step0_facts')
    os.makedirs(out_dir, exist_ok=True)
    facts = dict(submitted_code=SUBMITTED_CODE_FACTS, metric_protocol=PROTOCOL, environment=environment_info())
    torch.manual_seed(0)
    # --- metric introspection
    try:
        import pyiqa
        s = pyiqa.create_metric('ssim', device='cpu')
        p = pyiqa.create_metric('psnr', device='cpu')
        facts['pyiqa_ssim_options'] = dict(test_y_channel=s.net.test_y_channel, color_space=s.net.color_space,
                                           downsample=s.net.downsample, crop_border=s.net.crop_border)
        facts['pyiqa_psnr_options'] = dict(test_y_channel=p.net.test_y_channel, crop_border=p.net.crop_border)
        # --- gradient of the submitted SSIM loss
        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        gt = torch.rand(2, 3, 64, 64)
        l_sub = 1.0 - s(pred, gt).mean()
        l_new = 1.0 - ssim_y(pred, gt).mean()
        l_new.backward()
        facts['ssim_loss_gradient'] = dict(submitted_requires_grad=bool(l_sub.requires_grad),
                                           revised_grad_norm=float(pred.grad.norm()),
                                           revised_vs_pyiqa_ssim_abs_diff=float((ssim_y(pred.detach(), gt) - s(pred.detach(), gt)).abs().max()))
    except Exception as e:
        facts['pyiqa_check_error'] = str(e)
    # --- gate initialisation
    srb = SpectralResonanceBlock(64)
    g = srb.trace(torch.randn(4, 64, 32, 32))['gate']
    facts['srb_gate_at_initialisation'] = dict(mean=float(g.mean()), std=float(g.std()), min=float(g.min()), max=float(g.max()))
    # --- parameter accounting of the paper architecture (Table 10)
    m = KASIRNet(extractor_pretrained=False)
    facts['parameters_M'] = component_parameter_table(m)
    # --- submitted checkpoints
    for tag, path, flags in (('submitted_full', cfg.LEGACY_CHECKPOINT, {}), ('submitted_noSIM', cfg.LEGACY_NOSIM_CHECKPOINT, None)):
        if not path or not os.path.exists(path):
            facts[f'{tag}_checkpoint'] = f'not found at {path!r}'
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck)
        info = {k: ck[k] for k in ('epoch', 'best_psnr', 'current_psnr', 'iteration') if isinstance(ck, dict) and k in ck}
        info['n_tensors'] = len(sd)
        info['n_parameters_M'] = sum(v.numel() for v in sd.values() if v.dtype.is_floating_point) / 1e6
        try:
            mm = KASIRNet(extractor_pretrained=False, **({} if flags is not None else dict(use_sim=not any(k.startswith('vgg_extractor') for k in sd))))
            mm.load_state_dict(sd, strict=True)
            info['loads_strictly_into_current_architecture'] = True
            info['flags'] = mm.flags
        except Exception as e:
            info['loads_strictly_into_current_architecture'] = False
            info['load_error'] = str(e)[:400]
        if any(k.startswith('vgg_extractor') for k in sd):
            ref = SEMANTIC_EXTRACTORS['vgg19'](pretrained=True).state_dict()
            diffs = [float((sd['vgg_extractor.' + k].float() - v.float()).abs().max()) for k, v in ref.items()
                     if 'vgg_extractor.' + k in sd and v.dtype.is_floating_point and k not in ('mean', 'std')]
            info['vgg_max_abs_diff_to_imagenet'] = max(diffs) if diffs else None
            info['vgg_is_imagenet'] = bool(diffs and max(diffs) < 1e-5)
        facts[f'{tag}_checkpoint'] = info
    # --- data facts (if the sources are available)
    if roots is not None:
        facts['data'] = _data_facts(cfg, roots)
    write_json(facts, os.path.join(out_dir, 'facts.json'))
    with open(os.path.join(out_dir, 'facts.md'), 'w') as f:
        f.write('# Step 0 facts (generated)\n\n')
        for k, v in facts.items():
            f.write(f'## {k}\n```\n{json.dumps(v, indent=2, default=str)}\n```\n\n')
    print(f'[step0] facts written to {out_dir}')
    return facts


def _data_facts(cfg, roots):
    out = {}
    try:
        d = resolve_path(roots, ('jrdr', 'JRDR/rain_data_train_Heavy'))
        A, B = list_images(os.path.join(d, 'norain')), list_images(os.path.join(d, 'rain', 'X2'))

        def hf(f):
            x = np.asarray(Image.open(f).convert('L'), np.float32) / 255.
            return float(np.abs(np.diff(x, axis=1)).mean() + np.abs(np.diff(x, axis=0)).mean())
        import re
        key = lambda f: int(re.findall(r'\d+', os.path.basename(f))[0])
        out['jrdr_heavy'] = dict(hf_energy_norain=float(np.mean([hf(f) for f in A[:50]])),
                                 hf_energy_rain_X2=float(np.mean([hf(f) for f in B[:50]])),
                                 sorted_zip_pairs_with_different_scene=int(sum(key(a) != key(b) for a, b in zip(A, B))),
                                 n_pairs=len(A))
    except Exception as e:
        out['jrdr_heavy'] = f'unavailable: {e}'
    man = read_json(os.path.join(cfg.manifest_dir, 'train_manifest.json'))
    if man:
        out['leakage_audit_of_revised_pool'] = man.get('audit')
        out['revised_pool'] = {k: dict(n_train=v.get('n_train'), n_val=v.get('n_val'), removed=v.get('n_removed_overlap'))
                               for k, v in man['categories'].items()}
    if 'legacy_raining' in roots:
        try:
            aud = LeakageAuditor(os.path.join(cfg.manifest_dir, 'thumb_cache'), cfg.OVERLAP_THRESHOLD)
            tr = list_images(resolve_path(roots, ('legacy_raining', 'Dataset/target')))
            te = [g for _, g, _ in pair_files(resolve_path(roots, ('rain100l', '.')), resolve_path(roots, ('rain100l', '.')),
                                              ('regex', r'^rain-(\d+)$', r'^norain-(\d+)$'))[0]]
            dup, _ = aud.duplicates(tr, te, 'legacy_raining', 'rain100l')
            out['submitted_raining_vs_rain100l_test'] = dict(n_train=len(tr), n_duplicates=int(dup.sum()))
        except Exception as e:
            out['submitted_raining_vs_rain100l_test'] = f'unavailable: {e}'
    return out


# ==========================================
# 3. cost (TODO-13)
# ==========================================

def count_macs(model, size, device):
    from torch.utils.flop_counter import FlopCounterMode
    WindowAttention.use_fused = False   # explicit matmuls are counted by FlopCounterMode
    try:
        x = torch.rand(1, 3, *size, device=device)
        fc = FlopCounterMode(display=False, depth=None)
        with torch.no_grad(), fc:
            model(x)
        total = fc.get_total_flops() / 2 / 1e9
        per = fc.get_flop_counts()
        ext = 0.0
        for k, v in per.items():
            if k.endswith('vgg_extractor') or k.endswith('semantic_extractor'):
                ext = sum(v.values()) / 2 / 1e9
        return dict(total_GMACs=total, extractor_GMACs=ext, restoration_GMACs=total - ext,
                    note='FFT operations are not counted by torch.utils.flop_counter')
    finally:
        WindowAttention.use_fused = True


def profile_latency(model, size, device, dtype, warmup, repeats):
    x = torch.rand(1, 3, *size, device=device)
    amp = dtype == 'fp16' and device.type == 'cuda'
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(repeats):
                model(x)
            en.record()
            torch.cuda.synchronize()
            ms = st.elapsed_time(en) / repeats
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
        else:
            t0 = time.time()
            for _ in range(repeats):
                model(x)
            ms, mem = (time.time() - t0) * 1000 / repeats, None
    return dict(size=f'{size[0]}x{size[1]}', precision=dtype, latency_ms=ms, peak_memory_GB=mem)


def cost_report(cfg, run=None, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run = run or cfg.full_run()
    out_dir = cfg.analysis_dir('cost')
    model = build_model(cfg, run)
    ck = os.path.join(cfg.ckpt_dir(run['run_id']), 'final.pth')
    if os.path.exists(ck):
        load_weights(model, ck)
    model = model.to(device).eval()
    res = dict(run_id=run['run_id'], parameters_M=component_parameter_table(model), environment=environment_info())
    # loss-only networks (training time only, never loaded for inference)
    vl, il = VGGFeatureExtractor(), IdentityFeature()
    res['loss_networks_M'] = dict(vgg19_perceptual=sum(p.numel() for p in vl.parameters()) / 1e6,
                                  resnet50_identity=sum(p.numel() for p in il.parameters()) / 1e6)
    res['instantiated_at_training_M'] = res['parameters_M']['loaded at inference'] + sum(res['loss_networks_M'].values())
    # SRB vs plain-conv replacement block
    srb_p = sum(p.numel() for m in model.modules() if isinstance(m, SpectralResonanceBlock) for p in m.parameters())
    plain = KASIRNet(extractor_pretrained=False, use_srb=False)
    plain_p = sum(p.numel() for m in plain.modules() if isinstance(m, PlainConvBlock) for p in m.parameters())
    res['srb_vs_plain_conv_M'] = dict(srb_blocks=srb_p / 1e6, plain_conv_blocks=plain_p / 1e6, difference=(srb_p - plain_p) / 1e6)
    res['macs'] = {f'{h}x{w}': count_macs(model, (h, w), device) for (h, w) in [(256, 256), (128, 128)]}
    rows = []
    for size in cfg.PROFILE_SIZES:
        for dt in ('fp32', 'fp16'):
            if dt == 'fp16' and device.type != 'cuda':
                continue
            try:
                rows.append(profile_latency(model, size, device, dt, cfg.PROFILE_WARMUP if device.type == 'cuda' else 1,
                                            cfg.PROFILE_REPEATS if device.type == 'cuda' else 2))
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(size=f'{size[0]}x{size[1]}', precision=dt, latency_ms=None, peak_memory_GB='OOM'))
    res['inference_profile'] = rows
    # peak memory of one training step (batch BATCH_SIZE, patch, AMP, all loss terms)
    if device.type == 'cuda':
        try:
            from Loss import OmniRestorationLoss  # nb:skip
            tm = build_model(cfg, run).to(device).train()
            crit = OmniRestorationLoss(cfg.loss_weights(run), amp=cfg.AMP).to(device)
            opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=cfg.LR)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.rand(cfg.BATCH_SIZE, 3, cfg.PATCH_SIZE, cfg.PATCH_SIZE, device=device)
            for _ in range(3):
                opt.zero_grad()
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.AMP):
                    y, _ = tm(x)
                l, _ = crit(y, x)
                l.backward()
                opt.step()
            res['training_step_peak_memory_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
            del tm, crit, opt
        except Exception as e:
            res['training_step_peak_memory_GB'] = f'failed: {e}'
    write_json(res, os.path.join(out_dir, 'cost.json'))
    save_table([dict(size=r['size'], precision=r['precision'], latency_ms=r['latency_ms'], peak_memory_GB=r['peak_memory_GB'])
                for r in rows], 'table10_latency_memory', cfg, digits=dict(latency_ms=1, peak_memory_GB=2))
    return res


# ==========================================
# 4. significance (TODO-12C) and perceptual metrics (TODO-14)
# ==========================================

def significance_report(cfg, evaluator, run_id):
    ci_rows, test_rows, perc_rows = [], [], []
    for bench in cfg.MAIN_BENCHMARKS + ['rain100h']:
        ours = per_image(cfg, run_id, bench)
        if not ours:
            continue
        row = dict(benchmark=bench, n=len(ours))
        for k in ('psnr', 'ssim', 'lpips', 'dists'):
            if k in ours[0]:
                m, lo, hi = bootstrap_mean_ci([r[k] for r in ours], cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_SEED)
                row.update({k: m, f'{k}_ci_low': lo, f'{k}_ci_high': hi})
        ci_rows.append(row)
        perc_rows.append(dict(benchmark=bench, method='KASIRNet', n=len(ours), **{k: row.get(k) for k in ('psnr', 'ssim', 'lpips', 'dists')},
                              divergent_lpips_pct=None, divergent_dists_pct=None))
        for method, folder in cfg.COMPETITOR_OUTPUTS.get(bench, {}).items():
            comp = evaluator.score_folder(bench, method, folder,
                                          os.path.join(cfg.analysis_dir('competitors'), bench, method.replace(' ', '_') + '.csv'))
            pairs = join_by_key(ours, comp)
            if len(pairs) < 2:
                continue
            for k in ('psnr', 'ssim', 'lpips', 'dists'):
                t = paired_test([a[k] for a, _ in pairs], [b[k] for _, b in pairs], cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_SEED)
                test_rows.append(dict(benchmark=bench, competitor=method, metric=k, **t))
            div_l = np.mean([(a['psnr'] > b['psnr']) and (a['lpips'] > b['lpips']) for a, b in pairs]) * 100
            div_d = np.mean([(a['psnr'] > b['psnr']) and (a['dists'] > b['dists']) for a, b in pairs]) * 100
            perc_rows.append(dict(benchmark=bench, method=method, n=len(pairs),
                                  **{k: float(np.mean([b[k] for _, b in pairs])) for k in ('psnr', 'ssim', 'lpips', 'dists')},
                                  divergent_lpips_pct=float(div_l), divergent_dists_pct=float(div_d)))
    adj = holm([r['p_value'] for r in test_rows])
    for r, a in zip(test_rows, adj):
        r['p_holm'] = a
    save_table(ci_rows, 'table9_bootstrap_ci', cfg)
    save_table(test_rows, 'table9_wilcoxon_vs_competitors', cfg,
               digits=dict(n=0, mean_a=3, mean_b=3, mean_diff=3, median_diff=3, win_rate=3, diff_ci_low=3, diff_ci_high=3,
                           wilcoxon_W=1, p_value=4, rank_biserial=3, p_holm=4))
    save_table(perc_rows, 'table11_perceptual', cfg,
               digits=dict(n=0, psnr=2, ssim=3, lpips=3, dists=3, divergent_lpips_pct=1, divergent_dists_pct=1))
    return ci_rows, test_rows, perc_rows


# ==========================================
# 5. prompt probe + mixed degradations (TODO-16)
# ==========================================

@torch.no_grad()
def collect_descriptors(cfg, roots, model, bench, n, device, field='input'):
    ds = BenchmarkDataset(cfg, roots, bench, max_images=n)
    P = []
    for i in range(len(ds)):
        x = ds[i][field].unsqueeze(0).to(device)
        P.append(model.prompt_learner(model.check_image_size(x)[0]).float().cpu().numpy()[0])
    return np.stack(P)


def prompt_probe_report(cfg, roots, evaluator, model, run_id):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import confusion_matrix
    out_dir = cfg.analysis_dir('prompt_probe')
    os.makedirs(out_dir, exist_ok=True)
    dev = evaluator.device
    fams, X, y = [], [], []
    for f in cfg.PROBE_FAMILIES:
        try:
            P = collect_descriptors(cfg, roots, model, cfg.PROBE_FAMILIES[f], cfg.PROBE_IMAGES_PER_FAMILY, dev)
        except (FileNotFoundError, KeyError) as e:
            print(f'[probe] family {f} skipped: {e}')
            continue
        X.append(P); y += [len(fams)] * len(P); fams.append(f)
    X, y = np.concatenate(X), np.array(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    cv = StratifiedKFold(max(2, min(5, int(np.bincount(y).min()))), shuffle=True, random_state=cfg.SEED)
    acc = cross_val_score(clf, X, y, cv=cv)
    pred = cross_val_predict(clf, X, y, cv=cv)
    cm = confusion_matrix(y, pred, labels=list(range(len(fams))))
    clf.fit(X, y)
    probe = dict(families=fams, n_per_family=int(np.bincount(y).min()), cv_accuracy_mean=float(acc.mean()),
                 cv_accuracy_std=float(acc.std()), fold_accuracies=acc.tolist(), confusion_matrix=cm.tolist(),
                 per_class_recall={f: float(cm[i, i] / max(1, cm[i].sum())) for i, f in enumerate(fams)})
    # mixed degradations: probe probabilities + restoration with normal / first-degradation descriptor
    mixed_rows, projections = [], {'single': (X, y)}
    for mb in [b for b in cfg.EXTRA_BENCHMARKS if b.startswith('mix_')]:
        try:
            spec = cfg.benchmarks()[mb]
            Pm = collect_descriptors(cfg, roots, model, mb, cfg.PROBE_IMAGES_PER_FAMILY, dev)
            Pf = collect_descriptors(cfg, roots, model, mb, cfg.PROBE_IMAGES_PER_FAMILY, dev, field='first')
            pr = clf.predict_proba(Pm)
            if not all(f in fams for f in spec['families']):
                continue
            fi = [fams.index(f) for f in spec['families']]
            top2 = np.argsort(-pr, 1)[:, :2]
            norm = evaluator.evaluate(model, run_id, mb, metric_names=('psnr', 'ssim'))
            ctrl = evaluator.evaluate(model, run_id, mb, prompt_mode='first', metric_names=('psnr', 'ssim'))
            cos = float(np.mean(np.sum(Pm * Pf, 1) / (np.linalg.norm(Pm, axis=1) * np.linalg.norm(Pf, axis=1) + 1e-8)))
            mixed_rows.append(dict(benchmark=mb, families='+'.join(spec['families']), n=norm['n_images'],
                                   input_psnr=norm['input_psnr'], input_ssim=norm['input_ssim'],
                                   psnr=norm['psnr'], ssim=norm['ssim'],
                                   psnr_first_descriptor=ctrl['psnr'], ssim_first_descriptor=ctrl['ssim'],
                                   prob_first_family=float(pr[:, fi[0]].mean()), prob_second_family=float(pr[:, fi[1]].mean()),
                                   both_in_top2_pct=float(np.mean([set(fi) <= set(t) for t in top2]) * 100),
                                   cos_mixed_vs_first_descriptor=cos))
            projections[mb] = (Pm, None)
        except Exception as e:
            print(f'[probe] {mb} skipped: {e}')
    # TODO-17: to which known degradation does the descriptor of an UNSEEN degradation look closest?
    routing = {}
    for zb in cfg.ZERO_SHOT_BENCHMARKS:
        try:
            Pz = collect_descriptors(cfg, roots, model, zb, cfg.PROBE_IMAGES_PER_FAMILY, dev)
            pr = clf.predict_proba(Pz)
            top = np.bincount(pr.argmax(1), minlength=len(fams)) / len(pr)
            routing[zb] = dict(n=len(Pz), mean_probability={f: float(pr[:, i].mean()) for i, f in enumerate(fams)},
                               top1_share={f: float(top[i]) for i, f in enumerate(fams)},
                               routed_to=fams[int(top.argmax())], routed_share=float(top.max()))
            projections[zb] = (Pz, None)
        except Exception as e:
            print(f'[probe] zero-shot routing of {zb} skipped: {e}')
    write_json(routing, os.path.join(out_dir, 'zero_shot_routing.json'))
    write_json(probe, os.path.join(out_dir, 'probe.json'))
    save_table([dict(families=', '.join(fams), n_per_family=probe['n_per_family'], cv_accuracy_mean=probe['cv_accuracy_mean'] * 100,
                     cv_accuracy_std=probe['cv_accuracy_std'] * 100, **{f'recall_{k}': v * 100 for k, v in probe['per_class_recall'].items()})],
               'table13_prompt_probe', cfg, digits={**{'n_per_family': 0, 'cv_accuracy_mean': 1, 'cv_accuracy_std': 1},
                                                    **{f'recall_{k}': 1 for k in fams}})
    save_table(mixed_rows, 'table13_mixed_degradations', cfg,
               digits=dict(n=0, input_psnr=2, input_ssim=3, psnr=2, ssim=3, psnr_first_descriptor=2, ssim_first_descriptor=3,
                           prob_first_family=3, prob_second_family=3, both_in_top2_pct=1, cos_mixed_vs_first_descriptor=3))
    try:
        from sklearn.decomposition import PCA
        pca = PCA(2, random_state=0).fit(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        Z = pca.transform(X)
        for i, f in enumerate(fams):
            ax.scatter(*Z[y == i].T, s=12, alpha=.7, label=f)
        for mb, (Pm, _) in projections.items():
            if mb != 'single':
                ax.scatter(*pca.transform(Pm).T, s=14, marker='x', alpha=.8, label=mb)
        ax.legend(fontsize=7); ax.set_title('Visual descriptor p (PCA): single vs mixed degradations')
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'descriptor_pca.png'), dpi=200); plt.close(fig)
    except Exception as e:
        print('[probe] figure skipped:', e)
    return probe, mixed_rows


# ==========================================
# 6. severity (TODO-09A) + frozen-feature stability
# ==========================================

SEVERITY_LEVELS = [('blur', 'GoPro tercile 1 (mild)', 'gopro', 0), ('blur', 'GoPro tercile 2', 'gopro', 1),
                   ('blur', 'GoPro tercile 3 (severe)', 'gopro', 2),
                   ('rain', 'Rain100L (light)', 'rain100l', None), ('rain', 'Rain100H (heavy)', 'rain100h', None),
                   ('snow', 'Snow100K-S', 'snow100k_s', None), ('snow', 'Snow100K-M', 'snow100k_m', None), ('snow', 'Snow100K-L', 'snow100k_l', None),
                   ('noise', 'Gaussian sigma=15', 'cbsd68_g15', None), ('noise', 'Gaussian sigma=25', 'cbsd68_g25', None),
                   ('noise', 'Gaussian sigma=50', 'cbsd68_g50', None)]


def _tercile_keys(rows):
    v = np.array([r['input_psnr'] for r in rows])
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    # higher input PSNR = milder blur -> tercile 0 = mildest
    t = [0 if x >= q2 else (1 if x >= q1 else 2) for x in v]
    return {str(r['key']): ti for r, ti in zip(rows, t)}


def severity_report(cfg, run_ids):
    """run_ids: models to compare (e.g. full model, full_s1, noSIM_s1); they must have been
    evaluated on the severity benchmarks (main() does it)."""
    rows = []
    ref = per_image(cfg, run_ids[0], 'gopro')
    terc = _tercile_keys(ref) if ref else {}
    for fam, label, bench, tq in SEVERITY_LEVELS:
        row = dict(family=fam, level=label)
        for rid in run_ids:
            pr = per_image(cfg, rid, bench)
            if tq is not None:
                pr = [r for r in pr if terc.get(str(r['key'])) == tq]
            if not pr:
                continue
            row['n'] = len(pr)
            row['input_psnr'] = float(np.mean([r['input_psnr'] for r in pr]))
            row[f'psnr_{rid}'] = float(np.mean([r['psnr'] for r in pr]))
            row[f'ssim_{rid}'] = float(np.mean([r['ssim'] for r in pr]))
        if len(run_ids) >= 3 and f'psnr_{run_ids[1]}' in row and f'psnr_{run_ids[2]}' in row:
            row['delta_SIM_psnr'] = row[f'psnr_{run_ids[1]}'] - row[f'psnr_{run_ids[2]}']
        rows.append(row)
    save_table(rows, 'table8_sim_severity', cfg)
    return rows


@torch.no_grad()
def feature_stability_report(cfg, roots, device, n_images=60):
    """Cosine similarity between frozen-extractor features of the degraded input and of the
    clean image (per position, averaged), per scale and per severity level: measures how far the
    'stable semantic prior' assumption holds for each extractor (Reviewer 3, comment 3).
    GoPro levels are terciles of the input PSNR within the sampled images."""
    rows = []
    benches = []
    for fam, label, bench, tq in SEVERITY_LEVELS:
        if bench not in [b for b, _ in benches]:
            benches.append((bench, fam))
    for ename, cls in SEMANTIC_EXTRACTORS.items():
        ext = cls(pretrained=True).to(device).eval()
        for bench, fam in benches:
            try:
                ds = BenchmarkDataset(cfg, roots, bench, max_images=n_images)
            except Exception as e:
                print(f'[stability] {bench} skipped: {e}')
                continue
            sims, pin = [], []
            for i in range(len(ds)):
                it = ds[i]
                x, g = it['input'].unsqueeze(0).to(device), it['gt'].unsqueeze(0).to(device)
                pin.append(float(10 * torch.log10(1 / (((x - g) ** 2).mean() + 1e-8))))
                if max(x.shape[2:]) > 1024:
                    sc = 1024 / max(x.shape[2:])
                    x = F.interpolate(x, scale_factor=sc, mode='bilinear', antialias=True)
                    g = F.interpolate(g, scale_factor=sc, mode='bilinear', antialias=True)
                sims.append([float(F.cosine_similarity(a, b, dim=1).mean()) for a, b in zip(ext(x), ext(g))])
            if not sims:
                continue
            S, P = np.array(sims), np.array(pin)
            if bench == 'gopro':
                q1, q2 = np.quantile(P, [1 / 3, 2 / 3])
                groups = [('GoPro tercile 1 (mild)', P >= q2), ('GoPro tercile 2', (P >= q1) & (P < q2)), ('GoPro tercile 3 (severe)', P < q1)]
            else:
                groups = [(next(l for f, l, b, t in SEVERITY_LEVELS if b == bench), np.ones(len(P), bool))]
            for label, mask in groups:
                s = S[mask]
                row = dict(extractor=ename, family=fam, level=label, n=int(mask.sum()), input_psnr=float(P[mask].mean()))
                for k in range(s.shape[1]):
                    row[f'cos_scale{k + 1}'] = float(s[:, k].mean())
                row['cos_mean'] = float(s.mean())
                rows.append(row)
        del ext
    save_table(rows, 'table8_feature_stability', cfg,
               digits={**{f'cos_scale{k}': 3 for k in range(1, 5)}, 'cos_mean': 3, 'n': 0, 'input_psnr': 2})
    return rows


# ==========================================
# 7. spectral analyses (Reviewer 3 c.2, Reviewer 5 c.2, TODO-05)
# ==========================================

def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


@torch.no_grad()
def amplitude_phase_attribution(x, y):
    """Oppenheim-Lim style swap test per image.  Returns PSNR of the degraded input, of the input
    with the clean amplitude restored (degraded phase kept) and with the clean phase restored
    (degraded amplitude kept), plus relative amplitude error and amplitude-weighted phase error."""
    X, Y = torch.fft.fft2(x.double()), torch.fft.fft2(y.double())
    h_amp = torch.fft.ifft2(torch.polar(Y.abs(), X.angle())).real.clamp(0, 1)
    h_pha = torch.fft.ifft2(torch.polar(X.abs(), Y.angle())).real.clamp(0, 1)
    ps = lambda a: float(10 * torch.log10(1 / (((a - y.double()) ** 2).mean() + 1e-12)))
    amp_err = float((X.abs() - Y.abs()).norm() / (Y.abs().norm() + 1e-12))
    ph_err = float((Y.abs() * _wrap(X.angle() - Y.angle()).abs()).sum() / (Y.abs().sum() + 1e-12))
    return dict(psnr_input=ps(x), psnr_fix_amplitude=ps(h_amp), psnr_fix_phase=ps(h_pha),
                rel_amplitude_error=amp_err, weighted_phase_error_rad=ph_err)


@torch.no_grad()
def spectral_report(cfg, roots, model, device, benches=None, n_images=30):
    out_rows, srb_rows = [], []
    benches = benches or (cfg.MAIN_BENCHMARKS + ['rain100h', 'snow100k_s', 'snow100k_m', 'cbsd68_g15', 'cbsd68_g25', 'cbsd68_g50'])
    srbs = {n: m for n, m in model.named_modules() if isinstance(m, SpectralResonanceBlock)}
    captured = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, n=n: captured.__setitem__(n, inp[0].detach())) for n, m in srbs.items()]
    try:
        for bench in benches:
            try:
                ds = BenchmarkDataset(cfg, roots, bench, max_images=n_images)
            except Exception as e:
                print(f'[spectral] {bench} skipped: {e}')
                continue
            acc, layer = [], {}
            for i in range(len(ds)):
                it = ds[i]
                acc.append(amplitude_phase_attribution(it['input'], it['gt']))
                if srbs:
                    x = it['input'].unsqueeze(0).to(device)
                    if max(x.shape[2:]) > 768:
                        s = 768 / max(x.shape[2:])
                        x = F.interpolate(x, scale_factor=s, mode='bilinear', antialias=True)
                    captured.clear()
                    model(x)
                    for n, inp in captured.items():
                        t = srbs[n].trace(inp)
                        w = t['F_in'].abs()
                        d_gate = float((w * _wrap(t['F_gated'].angle() - t['F_in'].angle()).abs()).sum() / w.sum())
                        d_mix = float((w * _wrap(t['F_out'].angle() - t['F_in'].angle()).abs()).sum() / w.sum())
                        L = layer.setdefault(n, dict(gate_mean=[], gate_std=[], phase_dev_gate=[], phase_dev_output=[]))
                        L['gate_mean'].append(float(t['gate'].mean())); L['gate_std'].append(float(t['gate'].std()))
                        L['phase_dev_gate'].append(d_gate); L['phase_dev_output'].append(d_mix)
            r = dict(benchmark=bench, n=len(acc), **{k: float(np.mean([a[k] for a in acc])) for k in acc[0]})
            gi, ga, gp = r['psnr_input'], r['psnr_fix_amplitude'], r['psnr_fix_phase']
            r['amplitude_share'] = float(max(0.0, ga - gi) / (max(0.0, ga - gi) + max(0.0, gp - gi) + 1e-9))
            out_rows.append(r)
            for n, L in layer.items():
                srb_rows.append(dict(benchmark=bench, srb=n, **{k: float(np.mean(v)) for k, v in L.items()}))
    finally:
        for h in hooks:
            h.remove()
    save_table(out_rows, 'spectral_amplitude_phase_attribution', cfg,
               digits=dict(n=0, psnr_input=2, psnr_fix_amplitude=2, psnr_fix_phase=2, rel_amplitude_error=3,
                           weighted_phase_error_rad=3, amplitude_share=3))
    save_table(srb_rows, 'spectral_srb_phase_and_gates', cfg,
               digits=dict(gate_mean=3, gate_std=3, phase_dev_gate=6, phase_dev_output=3))
    return out_rows, srb_rows


def protocol_effect_report(cfg, evaluator, model, run_id, benches=None):
    """Test-set effect of the inference protocol (reported only; the protocol itself was chosen on the
    held-out validation pairs): every main benchmark with the selected protocol and with 'full'."""
    benches = benches or cfg.MAIN_BENCHMARKS
    sel, rows = evaluator.protocol, []
    if sel == 'full':
        print('[protocol] the selected protocol is the full-image pass: nothing to compare')
        return rows
    for b in benches:
        try:
            a = evaluator.evaluate(model, run_id, b, metric_names=('psnr', 'ssim'))
            f = evaluator.evaluate(model, run_id, b, metric_names=('psnr', 'ssim'), protocol='full')
        except FileNotFoundError as e:
            print(f'[protocol] {b} skipped: {e}')
            continue
        rows.append(dict(benchmark=b, n=a['n_images'], protocol=sel, psnr=a['psnr'], ssim=a['ssim'],
                         psnr_full_image=f['psnr'], ssim_full_image=f['ssim'], delta_psnr=a['psnr'] - f['psnr'],
                         delta_ssim=a['ssim'] - f['ssim']))
    save_table(rows, 'table_inference_protocol_test', cfg, digits=dict(n=0, psnr=2, psnr_full_image=2, delta_psnr=2,
                                                                       ssim=3, ssim_full_image=3, delta_ssim=3))
    return rows


@torch.no_grad()
def srb_consistency_report(cfg, roots, model, device, benches=None, n_images=10, P=256):
    """Why the inference protocol matters (R3-2 / R5-2).
    (a) Exact spatial form of every SRB.  The real 1x1 conv W = [[A, B], [C, D]] on [Re F; Im F] equals
        alpha F + beta conj(F) with alpha = ((A+D) + i(C-B))/2, beta = ((A-D) + i(C+B))/2, i.e. in image space
        y = a_r x + a_i Qx + b_r Rx + b_i QRx  (R: point reflection x(-n mod N), Q: quadrature transform =
        irFFT2(i rFFT2(.))).  The anti-linear terms (beta) couple every position with the opposite position
        of the WHOLE map: inside a 256x256 training patch, or across a whole 720x1280 test frame.
        Table: energy share of each term per SRB (from the weights only).
    (b) Train/test gap of the resonance gate: input of the gate (mean un-normalised amplitude) and gate
        value on full test images vs on their P x P tiles (the training regime)."""
    srbs = {n: m for n, m in model.named_modules() if isinstance(m, SpectralResonanceBlock)}
    rows_a = []
    for n, m in srbs.items():
        Wt = m.freq_conv.weight.detach().float()[:, :, 0, 0].cpu()
        c = Wt.shape[0] // 2
        A, B, Cm, D = Wt[:c, :c], Wt[:c, c:], Wt[c:, :c], Wt[c:, c:]
        e = dict(identity_mix=((A + D) / 2).pow(2).sum(), quadrature=((Cm - B) / 2).pow(2).sum(),
                 reflection=((A - D) / 2).pow(2).sum(), reflection_quadrature=((Cm + B) / 2).pow(2).sum())
        tot = float(sum(e.values()))
        rows_a.append(dict(srb=n, channels=c, **{f'share_{k}': float(v) / tot for k, v in e.items()},
                           anti_linear_share=float(e['reflection'] + e['reflection_quadrature']) / tot,
                           bias_norm=float(m.freq_conv.bias.detach().norm()) if m.freq_conv.bias is not None else 0.0))
    save_table(rows_a, 'spectral_srb_operator_decomposition', cfg, digits=dict(channels=0))
    benches = benches or cfg.MAIN_BENCHMARKS
    cap = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, n=n: cap.__setitem__(n, inp[0].detach())) for n, m in srbs.items()]
    acc = {}
    try:
        for bench in benches:
            try:
                ds = BenchmarkDataset(cfg, roots, bench, max_images=n_images)
            except Exception as e:
                print(f'[srb-consistency] {bench} skipped: {e}')
                continue
            for i in range(len(ds)):
                x = ds[i]['input'].unsqueeze(0).to(device)
                H, W = x.shape[2:]
                if H <= P or W <= P:
                    continue
                x = x[:, :, :min(H, 720) // 16 * 16, :min(W, 1280) // 16 * 16]
                cap.clear(); model(x); full = dict(cap)
                tiles = torch.cat([x[:, :, a:a + P, b:b + P] for a in range(0, x.shape[2] - P + 1, P)
                                   for b in range(0, x.shape[3] - P + 1, P)])
                cap.clear(); model(tiles); tl = dict(cap)
                for n, m in srbs.items():
                    gf, gt_ = m.trace(full[n])['gate'], m.trace(tl[n])['gate']
                    sf = torch.fft.rfft2(full[n].float()).abs().mean().item()
                    st = torch.fft.rfft2(tl[n].float()).abs().mean().item()
                    L = acc.setdefault((bench, n), dict(gate_input_full=[], gate_input_tile=[], gate_full=[], gate_tile=[], gate_abs_diff=[]))
                    L['gate_input_full'].append(sf); L['gate_input_tile'].append(st)
                    L['gate_full'].append(float(gf.mean())); L['gate_tile'].append(float(gt_.mean()))
                    L['gate_abs_diff'].append(float((gf.mean(0) - gt_.mean(0)).abs().mean()))
    finally:
        for h in hooks:
            h.remove()
    rows_b = []
    for (bench, n), L in acc.items():
        r = dict(benchmark=bench, srb=n, n=len(L['gate_full']), **{k: float(np.mean(v)) for k, v in L.items()})
        r['gate_input_ratio_full_over_tile'] = r['gate_input_full'] / max(r['gate_input_tile'], 1e-12)
        rows_b.append(r)
    save_table(rows_b, 'spectral_srb_train_test_gap', cfg, digits=dict(n=0))
    return rows_a, rows_b


# ==========================================
# 8. qualitative figures (TODO-06)
# ==========================================

def _edges(t):
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    g = F.pad(t.mean(0, keepdim=True).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    ex, ey = F.conv2d(g, sx), F.conv2d(g, sx.transpose(2, 3))
    return torch.sqrt(ex ** 2 + ey ** 2)[0, 0].numpy()


def _spectrum(t):
    return torch.log1p(torch.fft.fftshift(torch.fft.fft2(t.mean(0))).abs()).numpy()


@torch.no_grad()
def qualitative_figures(cfg, roots, evaluator, model, run_id):
    out_dir = cfg.analysis_dir('figures')
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for fig_id, bench in (('S1', 'rain100l'), ('S2', 'snow100k_l')):
        rows = per_image(cfg, run_id, bench)
        if not rows:
            print(f'[figures] {bench}: evaluate the run first')
            continue
        comps = {m: f for m, f in cfg.FIGURE_COMPETITORS.get(bench, {}).items() if os.path.isdir(f)}
        gains = np.array([r['psnr'] - r['input_psnr'] for r in rows])
        order = np.argsort(gains)
        picks = [rows[order[int(q * (len(rows) - 1))]] for q in (0.25, 0.5, 0.75)]   # representative, not cherry-picked
        ds = BenchmarkDataset(cfg, roots, bench)
        by_key = {str(ds.items[i][2]): i for i in range(len(ds))}
        cols = ['Input'] + list(comps)[:3] + ['KASIRNet', 'GT']
        fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 2.3 * 3))
        sel = []
        for ri, r in enumerate(picks):
            it = ds[by_key[str(r['key'])]]
            out, _ = model(it['input'].unsqueeze(0).to(evaluator.device))
            out = out[0].float().cpu()
            panels = [('Input', it['input'])]
            for m in list(comps)[:3]:
                f = next((os.path.join(comps[m], fn) for fn in os.listdir(comps[m])
                          if os.path.splitext(fn)[0].lower() in (os.path.splitext(it['name'])[0].lower(), str(it['key']).lower())), None)
                panels.append((m, TF.to_tensor(Image.open(f).convert('RGB')) if f else None))
            panels += [('KASIRNet', out), ('GT', it['gt'])]
            for ci, (title, t) in enumerate(panels):
                ax = axes[ri, ci]
                ax.axis('off')
                if t is None:
                    continue
                ax.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
                if title != 'GT':
                    m = evaluator.light(t.unsqueeze(0), it['gt'].unsqueeze(0))
                    ax.set_title(f"{title}\n{m['psnr'][0]:.2f} dB / {m['ssim'][0]:.3f}", fontsize=8)
                else:
                    ax.set_title('GT', fontsize=8)
            sel.append(dict(image=it['name'], psnr_gain=float(r['psnr'] - r['input_psnr'])))
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(out_dir, f'Fig{fig_id}_{bench}.{ext}'), dpi=300)
        plt.close(fig)
        # image / edge / spectrum rows for the median image
        it = ds[by_key[str(picks[1]['key'])]]
        out, _ = model(it['input'].unsqueeze(0).to(evaluator.device))
        cols2 = [('Input', it['input'])] + [(m, None) for m in list(comps)[:3]] + [('KASIRNet', out[0].float().cpu()), ('GT', it['gt'])]
        for j, (m, _) in enumerate(cols2):
            if m in comps:
                f = next((os.path.join(comps[m], fn) for fn in os.listdir(comps[m])
                          if os.path.splitext(fn)[0].lower() in (os.path.splitext(it['name'])[0].lower(), str(it['key']).lower())), None)
                cols2[j] = (m, TF.to_tensor(Image.open(f).convert('RGB')) if f else None)
        fig, axes = plt.subplots(3, len(cols2), figsize=(2.6 * len(cols2), 2.3 * 3))
        for ci, (title, t) in enumerate(cols2):
            for ri in range(3):
                axes[ri, ci].axis('off')
            if t is None:
                continue
            axes[0, ci].imshow(t.permute(1, 2, 0).clamp(0, 1).numpy()); axes[0, ci].set_title(title, fontsize=8)
            axes[1, ci].imshow(_edges(t), cmap='gray')
            axes[2, ci].imshow(_spectrum(t), cmap='magma')
        axes[1, 0].text(-0.08, 0.5, 'Sobel edges', transform=axes[1, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
        axes[2, 0].text(-0.08, 0.5, 'log |FFT|', transform=axes[2, 0].transAxes, rotation=90, va='center', ha='right', fontsize=8)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(out_dir, f'Fig{fig_id}_{bench}_image_edge_spectrum.{ext}'), dpi=300)
        plt.close(fig)
        write_json(dict(benchmark=bench, rule='images at the 25th/50th/75th percentile of the PSNR gain of KASIRNet over the input',
                        columns=cols, selected=sel), os.path.join(out_dir, f'Fig{fig_id}_{bench}.json'))
        made.append(fig_id)
    return made


# ==========================================
# 9. ablation tables (reduced budget)
# ==========================================

def _scores(cfg, rid, benches):
    s = {b: run_eval(cfg, rid, b) for b in benches}
    return None if any(v is None for v in s.values()) else s


def ablation_tables(cfg):
    B = cfg.ABLATION_BENCHMARKS
    ref = _scores(cfg, 'full_s1', B)
    tables = {}

    def delta_row(rid, label):
        s = _scores(cfg, rid, B)
        if s is None or ref is None:
            return None
        d = {b: s[b] - ref[b] for b in B}
        row = dict(variant=label, run_id=rid, **{f'psnr_{b}': s[b] for b in B}, **{f'd_{b}': d[b] for b in B})
        row['max_drop'] = min(d.values())
        row['avg_delta'] = float(np.mean(list(d.values())))
        row['ssim_avg'] = float(np.mean([run_eval(cfg, rid, b, 'ssim') for b in B]))
        return row

    # Table 6: components (seed 1)
    t6 = []
    for rid, label in [('full_s1', 'Full KASIRNet'), ('noSIM_s1', 'w/o SIM'), ('noSRB_s1', 'w/o SRB (plain conv)'),
                       ('noSCWAM_s1', 'w/o SCWAM+Meshgrid (local conv)'), ('noPrompt_s1', 'w/o Prompt/SFT'),
                       ('noMeshgrid_s1', 'w/o Meshgrid'), ('noGCB_s1', 'w/o GCB'), ('depth3_s1', 'depth: 3 scales'),
                       ('width32_s1', 'width: C=32')]:
        r = delta_row(rid, label)
        if r:
            t6.append(r)
    save_table(t6, 'table6_component_ablation', cfg); tables['t6'] = t6
    # Table 7: loss terms
    t7 = [r for r in (delta_row(f'no_{k}_s1', f'w/o L_{k}') for k in cfg.LOSS_WEIGHTS) if r]
    save_table(t7, 'table7_loss_ablation', cfg); tables['t7'] = t7
    # SCWAM design
    ts = [r for r in (delta_row(rid, lab) for rid, lab in [('scwam_noShift_s1', 'no cyclic shift'),
                                                            ('scwam_swinMask_s1', 'Swin masked shift'),
                                                            ('scwam_RPB_s1', 'relative position bias instead of Meshgrid')]) if r]
    save_table(ts, 'table_scwam_design', cfg); tables['scwam'] = ts
    # Table 9 A: seeds
    t9 = []
    full_avgs = None
    for name in ('full', 'noSIM', 'noSRB', 'noSCWAM', 'noPrompt'):
        per_seed = [_scores(cfg, f'{name}_s{s}', B) for s in cfg.ABLATION_SEEDS]
        per_seed = [p for p in per_seed if p]
        if not per_seed:
            continue
        row = dict(variant=name, n_seeds=len(per_seed))
        for b in B:
            v = [p[b] for p in per_seed]
            row[f'{b}_mean'] = float(np.mean(v)); row[f'{b}_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        avg = [float(np.mean([p[b] for b in B])) for p in per_seed]
        row['avg_mean'] = float(np.mean(avg)); row['avg_std'] = float(np.std(avg, ddof=1)) if len(avg) > 1 else None
        if name == 'full':
            full_avgs = avg
        elif full_avgs and len(full_avgs) == len(avg) and len(avg) > 1:
            from scipy.stats import ttest_rel
            d = np.array(avg) - np.array(full_avgs)
            row['delta_avg_mean'] = float(d.mean()); row['delta_avg_std'] = float(d.std(ddof=1))
            row['paired_t_p'] = float(ttest_rel(avg, full_avgs).pvalue)
        # per-image paired Wilcoxon (seed 1) on each benchmark
        if name != 'full':
            for b in B:
                pairs = join_by_key(per_image(cfg, f'{name}_s1', b), per_image(cfg, 'full_s1', b))
                if len(pairs) >= 6:
                    t = paired_test([a['psnr'] for a, _ in pairs], [c['psnr'] for _, c in pairs], cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_SEED)
                    row[f'{b}_wilcoxon_p'] = t['p_value']
        t9.append(row)
    save_table(t9, 'table9_seed_variance', cfg); tables['t9'] = t9
    # Table 9 B: interactions  I = d(A+B) - (dA + dB)  (negative = the pair overlaps less than additive)
    t9b = []
    for rid, a, b_ in [('noSIM+noSRB_s1', 'noSIM_s1', 'noSRB_s1'), ('noSIM+noSCWAM_s1', 'noSIM_s1', 'noSCWAM_s1'),
                       ('noSRB+noSCWAM_s1', 'noSRB_s1', 'noSCWAM_s1')]:
        r, ra, rb = delta_row(rid, rid), delta_row(a, a), delta_row(b_, b_)
        if r and ra and rb:
            t9b.append(dict(pair=rid.replace('_s1', ''), d_pair=r['avg_delta'], d_first=ra['avg_delta'], d_second=rb['avg_delta'],
                            additive_prediction=ra['avg_delta'] + rb['avg_delta'],
                            interaction=r['avg_delta'] - (ra['avg_delta'] + rb['avg_delta'])))
    r, ra, rf = delta_row('noSRB+noFFT_s1', 'noSRB+noFFT'), delta_row('noSRB_s1', 'noSRB'), delta_row('no_fft_s1', 'no_fft')
    if r and ra and rf:
        t9b.append(dict(pair='noSRB+noL_fft', d_pair=r['avg_delta'], d_first=ra['avg_delta'], d_second=rf['avg_delta'],
                        additive_prediction=ra['avg_delta'] + rf['avg_delta'],
                        interaction=r['avg_delta'] - (ra['avg_delta'] + rf['avg_delta'])))
    save_table(t9b, 'table9_interactions', cfg); tables['t9b'] = t9b
    # Table 12: loss-weight sensitivity
    t12 = []
    for k, w in cfg.LOSS_WEIGHTS.items():
        row = dict(term=k, default=w)
        for f in (0.5, 2.0):
            s = _scores(cfg, f'w_{k}_x{f:g}_s1', B)
            if s and ref:
                for b in B:
                    row[f'x{f:g}_d_{b}'] = s[b] - ref[b]
                row[f'x{f:g}_d_avg'] = float(np.mean([s[b] - ref[b] for b in B]))
        if len(row) > 2:
            t12.append(row)
    save_table(t12, 'table12_loss_sensitivity', cfg); tables['t12'] = t12
    # Table 8 B: extractors
    t8 = []
    for rid, label in [('full_s1', 'VGG-19 (paper)'), ('ext_resnet50_s1', 'ResNet-50 layer1-4'), ('ext_convnextT_s1', 'ConvNeXt-T stages 1-4')]:
        s = _scores(cfg, rid, B)
        if s:
            rec = read_json(os.path.join(cfg.run_dir(rid), 'run.json'), {})
            t8.append(dict(extractor=label, **{f'psnr_{b}': s[b] for b in B}, avg=float(np.mean(list(s.values()))),
                           trainable_M=rec.get('parameters_M', {}).get('trainable total'),
                           frozen_M=rec.get('parameters_M', {}).get('frozen extractor')))
    save_table(t8, 'table8_extractors', cfg); tables['t8'] = t8
    return tables


# ==========================================
# 10. main tables, Rain100H, zero-shot, Section 4.1
# ==========================================

def zero_shot_report(cfg, evaluator, run_id):
    """TODO-17, answered without embellishment: for each unseen degradation, what the model does to the image
    compared with leaving it untouched (per-image delta PSNR with a bootstrap CI, Wilcoxon signed-rank test vs the
    input, share of images improved / unchanged / degraded, perceptual change), to which known degradation the
    Prompt Learner routes it, and the same numbers for an all-in-one model that is zero-shot on it too.
    The verdict column is derived from the statistics only."""
    cats = cfg.train_categories()
    routing = read_json(os.path.join(cfg.analysis_dir('prompt_probe'), 'zero_shot_routing.json'), {})
    thr, rows = cfg.ZERO_SHOT_CHANGE_DB, []

    def describe(name, outs, ins_key):
        d = np.array([r['psnr'] - r[ins_key] for r in outs])
        t = paired_test([r['psnr'] for r in outs], [r[ins_key] for r in outs], cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_SEED)
        sig = t.get('p_value') is not None and t['p_value'] < 0.05
        verdict = ('improves the input' if sig and t['diff_ci_low'] > 0 else 'degrades the input' if sig and t['diff_ci_high'] < 0
                   else 'no significant change (behaves like the identity)')
        return dict(method=name, n=len(outs), psnr=float(np.mean([r['psnr'] for r in outs])), delta_psnr=float(d.mean()),
                    delta_ci_low=t['diff_ci_low'], delta_ci_high=t['diff_ci_high'], p_wilcoxon_vs_input=t.get('p_value'),
                    improved_pct=float((d > thr).mean() * 100), unchanged_pct=float((np.abs(d) <= thr).mean() * 100),
                    degraded_pct=float((d < -thr).mean() * 100), verdict=verdict)

    for b in cfg.ZERO_SHOT_BENCHMARKS:
        ours = per_image(cfg, run_id, b)
        if not ours:
            continue
        fam = cfg.benchmarks()[b]['family']
        seen = sorted(n for n, c in cats.items() if c['family'] == fam)
        base = dict(benchmark=b, degradation=fam, training_exposure=('SEEN in training: ' + ', '.join(seen)) if seen else 'zero-shot (never in training)',
                    input_psnr=float(np.mean([r['input_psnr'] for r in ours])), input_ssim=float(np.mean([r['input_ssim'] for r in ours])))
        for k in ('lpips', 'dists'):
            if f'input_{k}' in ours[0]:
                base[f'input_{k}'] = float(np.mean([r[f'input_{k}'] for r in ours]))
        rt = routing.get(b)
        if rt:
            base.update(prompt_routed_to=rt['routed_to'], prompt_routed_share_pct=rt['routed_share'] * 100)
        r = dict(base, **describe('KASIRNet', ours, 'input_psnr'))
        r.update(ssim=float(np.mean([x['ssim'] for x in ours])),
                 **{k: float(np.mean([x[k] for x in ours])) for k in ('lpips', 'dists') if k in ours[0]})
        rows.append(r)
        for method, sub in cfg.ZERO_SHOT_COMPETITORS.get(b, {}).items():
            folder = os.path.join(cfg.COMPETITOR_ROOT, b, sub)
            path = os.path.join(cfg.analysis_dir('competitors'), b, method.replace(' ', '_').replace('/', '-') + '.csv')
            comp = evaluator.score_folder(b, method, folder, path) if evaluator is not None else read_rows(path)
            pairs = join_by_key(ours, comp)
            if len(pairs) >= 2:
                cr = [dict(c, input_psnr=o['input_psnr']) for o, c in pairs]
                rows.append(dict(base, **describe(method, cr, 'input_psnr'), ssim=float(np.mean([c['ssim'] for c in cr])),
                                 **{k: float(np.mean([c[k] for c in cr])) for k in ('lpips', 'dists') if k in cr[0]}))
    save_table(rows, 'table14_zero_shot', cfg,
               digits=dict(n=0, input_psnr=2, input_ssim=3, input_lpips=3, input_dists=3, psnr=2, ssim=3, lpips=3, dists=3,
                           delta_psnr=2, delta_ci_low=2, delta_ci_high=2, p_wilcoxon_vs_input=4, improved_pct=1,
                           unchanged_pct=1, degraded_pct=1, prompt_routed_share_pct=1))
    return rows


def main_tables(cfg, run_id):
    rows = []
    for b in cfg.MAIN_BENCHMARKS + ['rain100h']:
        e = read_json(os.path.join(cfg.run_dir(run_id), 'eval', b, 'summary.json'))
        if e:
            rows.append(dict(benchmark=b, n=e['n_images'], psnr=e['psnr'], ssim=e['ssim'], lpips=e.get('lpips'), dists=e.get('dists'),
                             psnr_y=e.get('psnr_y'), ssim_y=e.get('ssim_y'),
                             input_psnr=e['input_psnr'], input_ssim=e['input_ssim'], gt_resized=e.get('n_gt_resized')))
    if rows:
        main = [r for r in rows if r['benchmark'] in cfg.ABLATION_BENCHMARKS]
        if len(main) == len(cfg.ABLATION_BENCHMARKS):
            rows.append(dict(benchmark='mean(GoPro, Rain100L, SIDD)', n=None, psnr=float(np.mean([r['psnr'] for r in main])),
                             ssim=float(np.mean([r['ssim'] for r in main]))))
    save_table(rows, 'table4_5_main_results', cfg, digits=dict(n=0, psnr=2, ssim=3, lpips=3, dists=3, psnr_y=2, ssim_y=3,
                                                              input_psnr=2, input_ssim=3, gt_resized=0))
    zs = zero_shot_report(cfg, None, run_id)
    return rows, zs


def section41_report(cfg):
    runs = []
    for rid, run in cfg.all_runs().items():
        if rid == cfg.full_alias():
            continue
        rec = read_json(os.path.join(cfg.run_dir(rid), 'run.json'))
        if rec and 'budget' in rec:
            runs.append(dict(run_id=rid, group=rec['group'], seed=rec['seed'], status=rec['status'],
                             epochs=rec['budget']['epochs'], iterations=rec['budget']['iterations_done'],
                             total_iters=rec['budget']['total_iters'], wall_clock_h=rec.get('wall_clock_hours'),
                             gpu=rec.get('environment', {}).get('gpu'), selected=rec.get('selection', {}).get('selected'),
                             best_val_psnr=rec.get('selection', {}).get('best_val_psnr')))
    save_table(runs, 'section41_runs', cfg, digits=dict(seed=0, epochs=0, iterations=0, total_iters=0, wall_clock_h=2, best_val_psnr=2))
    full = read_json(os.path.join(cfg.run_dir(cfg.report_run_id()), 'run.json'), {})
    reference = read_json(os.path.join(cfg.run_dir(cfg.FULL_RUN_ID), 'run.json'), {})
    man = read_json(os.path.join(cfg.manifest_dir, 'train_manifest.json'), {})
    rep = dict(full_run=full, reference_run_of_ablations=reference,
               inference_protocol=read_json(os.path.join(cfg.analysis_dir('inference_protocol'), 'selection.json'), {}),
               data_categories=man.get('categories'), leakage_audit=man.get('audit'), metric_protocol=PROTOCOL,
               total_gpu_hours_all_runs=float(np.nansum([r['wall_clock_h'] or 0 for r in runs])))
    write_json(rep, os.path.join(cfg.tables_dir, 'section41_reproducibility.json'))
    return rep
