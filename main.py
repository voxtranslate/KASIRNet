# ==========================================================================
# KASIRNet - main(): full training -> full inference -> analyses -> ablations
# -> tables.  Stages are idempotent: re-running the notebook in a new Colab
# session resumes the current training run from its last checkpoint and skips
# everything that is already finished.
# ==========================================================================
import os
import gc
import numpy as np
import torch

from Config import Config  # nb:skip
from Datasets import (resolve_sources, spec_sources, build_training_manifest, BenchmarkDataset, DatasetCache,  # nb:skip
                      manifest_is_current, overlap_benchmarks, manifest_used_files, validate_data, resolve_path,  # nb:skip
                      benchmark_files, validation_signature)  # nb:skip
from Trainer import Trainer, read_json, write_json, try_lock, release_lock  # nb:skip
from Inferencer import Evaluator, KASIRNetInferencer, select_inference_protocol  # nb:skip
from metrics import MetricSuite  # nb:skip
from Competitors import competitor_plan, competitor_folders, generate_competitor_outputs  # nb:skip
from Analysis import (step0_audit, cost_report, significance_report, prompt_probe_report, severity_report,  # nb:skip
                      feature_stability_report, spectral_report, qualitative_figures, ablation_tables,  # nb:skip
                      main_tables, section41_report, srb_consistency_report, protocol_effect_report, zero_shot_report,  # nb:skip
                      SEVERITY_LEVELS)  # nb:skip


WEIGHT_EXT = ('.pth', '.pt', '.ckpt', '.safetensors', '.bin', '.pkl')


def sync_weight_cache(cfg, direction):
    """Pretrained weights (torch hub cache: torchvision, pyiqa LPIPS/DISTS, competitor checkpoints) <->
    Google Drive (Config.WEIGHTS_CACHE), so that later sessions do not download them again."""
    import shutil
    hub, drive = torch.hub.get_dir(), cfg.WEIGHTS_CACHE
    if not drive:
        return 0
    src, dst = (drive, hub) if direction == 'restore' else (hub, drive)
    n = 0
    for dp, _, fs in os.walk(src):
        for f in fs:
            if not f.endswith(WEIGHT_EXT):
                continue
            a = os.path.join(dp, f)
            b = os.path.join(dst, os.path.relpath(a, src))
            if os.path.exists(b) and os.path.getsize(b) == os.path.getsize(a):
                continue
            try:
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copyfile(a, b + '.tmp')
                os.replace(b + '.tmp', b)
                n += 1
            except OSError as e:
                print(f'[weights] could not copy {a}: {e}')
    return n


def prefetch_weights(cfg):
    """Downloads (or restores from Drive) every pretrained network the pipeline uses, before any long
    stage: VGG-19 (SIM, perceptual loss), ResNet-50 (identity loss, TODO-09B), ConvNeXt-T (TODO-09B),
    AlexNet + LPIPS v0.1 and VGG-16 + DISTS (metrics).  A missing network stops the run here."""
    import torchvision.models as tvm
    restored = sync_weight_cache(cfg, 'restore')
    jobs = {'VGG-19 (IMAGENET1K_V1)': lambda: tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1),
            'ResNet-50 (IMAGENET1K_V1)': lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
            'ConvNeXt-T (IMAGENET1K_V1)': lambda: tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
            'LPIPS-Alex + DISTS (pyiqa)': lambda: MetricSuite('cpu', ('lpips', 'dists'))}
    failed = []
    for name, fn in jobs.items():
        for attempt in range(3):
            try:
                fn()
                break
            except Exception as e:
                if attempt == 2:
                    failed.append(f'{name}: {e}')
    stored = sync_weight_cache(cfg, 'store')
    print(f'[weights] pretrained networks ready ({restored} restored from Drive, {stored} newly cached in {cfg.WEIGHTS_CACHE})')
    if failed:
        raise RuntimeError('pretrained weights could not be downloaded:\n  ' + '\n  '.join(failed))


def setup_environment(cfg):
    try:
        from google.colab import drive  # noqa
        if not os.path.ismount(cfg.DRIVE_MOUNT):
            drive.mount(cfg.DRIVE_MOUNT)
    except ImportError:
        pass
    os.makedirs(cfg.OUTPUT_ROOT, exist_ok=True)
    os.makedirs(cfg.DATA_ROOT, exist_ok=True)
    write_json(cfg.to_dict(), os.path.join(cfg.OUTPUT_ROOT, 'config_snapshot.json'))


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
        self._roots, self._origin, self._manifest, self._evaluator, self._protocol = {}, {}, None, None, None
        self.cache = DatasetCache(cfg)
        self.stop = False

    # ------------------------------------------------------------ helpers
    def roots(self, keys, need_full=False):
        """Local roots of the data sources: Drive cache first, original download otherwise.
        need_full: a source restored from a 'used' cache is replaced by the complete source."""
        for k in sorted(set(keys)):
            partial = k in self._roots and self._origin.get(k) == 'used'
            if k in self._roots and not (need_full and partial):
                continue
            try:
                self._roots.update(resolve_sources(self.cfg, [k], self.cache, need_full))
                m = self.cache.info(k)
                from_cache = self._roots[k].startswith(self.cache.local)
                self._origin[k] = (m['mode'] if (m and from_cache) else 'origin')
            except Exception as e:
                print(f'[data] source {k} unavailable: {e}')
        return self._roots

    def bench_roots(self, benches):
        b = self.cfg.benchmarks()
        return self.roots([k for n in benches for k in spec_sources(b[n], b)])

    def manifest(self):
        """Frozen training manifest (built once from the complete sources, then reused)."""
        if self._manifest is None:
            cats = self.cfg.train_categories()
            b = self.cfg.benchmarks()
            keys = [k for s in cats.values() for k in spec_sources(s)]
            keys += [k for s in cats.values() for n in s.get('exclude_overlap_with', []) for k in spec_sources(b[n], b)]
            keys += [k for names in overlap_benchmarks(self.cfg).values() for n in names for k in spec_sources(b[n], b)]
            current = manifest_is_current(self.cfg)
            self._manifest = build_training_manifest(self.cfg, self.roots(keys, need_full=not current))
        return self._manifest

    def all_source_keys(self):
        b = self.cfg.benchmarks()
        keys = [k for s in self.cfg.train_categories().values() for k in spec_sources(s)]
        keys += [k for n in b for k in spec_sources(b[n], b)]
        return sorted(set(keys + ['jrdr', 'rain100l', 'legacy_raining']))

    def cache_benchmark_sources(self):
        """Sources read only by benchmarks are cached right after their download (their files are known
        without the training manifest), so a later failure in the audit cannot waste the download."""
        train_keys = {k for s in self.cfg.train_categories().values() for k in spec_sources(s)}
        files = {}
        for name in self.cfg.benchmarks():
            try:
                for k, r in benchmark_files(self.cfg, self._roots, name):
                    files.setdefault(k, set()).add(r)
            except (FileNotFoundError, KeyError):
                pass
        for k, root in sorted(self._roots.items()):
            if self._origin.get(k) == 'origin' and k not in train_keys and k in files:
                self.cache.store(k, root, files[k])

    def cache_sources(self):
        """Pack every source downloaded in this session into the Drive cache."""
        if self.cfg.DATA_CACHE_MODE == 'off':
            return
        used = manifest_used_files(self.cfg, self._roots, self.manifest())
        try:   # the Step-0 audit reads the complete RainTrainH folders
            d = resolve_path(self._roots, ('jrdr', 'JRDR/rain_data_train_Heavy'))
            for dp, _, fs in os.walk(d):
                used.setdefault('jrdr', set()).update(os.path.relpath(os.path.join(dp, f), self._roots['jrdr']) for f in fs)
        except (FileNotFoundError, KeyError):
            pass
        for k, root in sorted(self._roots.items()):
            if self._origin.get(k) == 'origin':
                self.cache.store(k, root, None if k == 'legacy_raining' else used.get(k))

    def evaluator(self):
        if self._evaluator is None:
            self._evaluator = Evaluator(self.cfg, self._roots, self.device)
        self._evaluator.roots = self._roots
        p = self.protocol()
        if p:
            self._evaluator.protocol = p
        return self._evaluator

    def protocol(self):
        """Inference protocol of every evaluation.  'auto': chosen once on the held-out validation pairs
        with the reported model and then frozen for all runs (None until that model is trained)."""
        cfg = self.cfg
        if cfg.INFERENCE_PROTOCOL != 'auto':
            return cfg.INFERENCE_PROTOCOL
        if self._protocol:
            return self._protocol
        rid = cfg.report_run_id()   # selected once with the reported model, then frozen for every run
        rec = read_json(os.path.join(cfg.run_dir(rid), 'run.json'), {})
        out_dir = cfg.analysis_dir('inference_protocol')
        key = [cfg.INFERENCE_CANDIDATES, list(cfg.INFERENCE_SELECT_MAX_SIZE), cfg.INFERENCE_SELECT_MIN_GAIN, rid,
               rec.get('selection', {}).get('selected')]
        sel = read_json(os.path.join(out_dir, 'selection.json'), {})
        if sel.get('key') == key:
            self._protocol = sel['selected']
            return self._protocol
        if rec.get('status') != 'complete':
            return None            # not selectable yet: evaluations wait (evaluate_many defers them)
        model = Evaluator(cfg, self._roots, self.device).load_run_model(cfg.all_runs()[rid])
        sel = select_inference_protocol(cfg, model, self.manifest()['val'], self.device, out_dir)
        sel['key'] = key
        write_json(sel, os.path.join(out_dir, 'selection.json'))
        del model
        self.free()
        self._protocol = sel['selected']
        return self._protocol

    def complete(self, run_id):
        return read_json(os.path.join(self.cfg.run_dir(run_id), 'run.json'), {}).get('status') == 'complete'

    def model(self, run_id):
        return self.evaluator().load_run_model(self.cfg.all_runs()[run_id])

    @staticmethod
    def free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_many(self, run_id, benches, metric_names=None, save=False):
        if self.protocol() is None:
            print(f'[eval] {run_id}: deferred until {self.cfg.report_run_id()} is trained and the inference protocol is '
                  f'selected (it is evaluated when the notebook runs again)')
            return
        ev = self.evaluator()
        model = None
        for b in benches:
            if ev.is_done(run_id, b):
                continue
            self.bench_roots([b])
            try:
                model = model or self.model(run_id)
                ev.evaluate(model, run_id, b, metric_names=metric_names,
                            save_images=save and self.cfg.SAVE_RESTORED_IMAGES and b in self.cfg.MAIN_BENCHMARKS and not b.startswith('snow'))
            except FileNotFoundError as e:
                print(f'[eval] {run_id} / {b} skipped: {e}')
        del model
        self.free()

    # ------------------------------------------------------------- stages
    def stage_audit(self):
        """Step 0: pretrained weights, all sources (Drive cache or download), frozen training manifest
        with the leakage audit, validation of every training category and benchmark, Step-0 facts, Drive cache."""
        cfg = self.cfg
        prefetch_weights(cfg)
        keys = self.all_source_keys()
        self.cache.inventory(keys)
        self.roots(keys, need_full=not manifest_is_current(cfg))
        self.cache_benchmark_sources()
        missing = [k for k in keys if k not in self._roots]
        if missing:
            raise RuntimeError(f'data sources not available in this session: {missing}. The reason is printed above '
                               f'(e.g. Google-Drive "quota exceeded": run again later, or place the archive in '
                               f'{cfg.DATA_CACHE_ROOT}/archives/). Everything downloaded so far is kept.')
        man = self.manifest()
        vsig = validation_signature(cfg)
        if read_json(os.path.join(cfg.manifest_dir, 'data_validation.sig.json'), {}).get('signature') != vsig:
            validate_data(cfg, self._roots, man)     # again whenever a benchmark or category definition changed
            write_json(dict(signature=vsig), os.path.join(cfg.manifest_dir, 'data_validation.sig.json'))
        if not os.path.exists(os.path.join(cfg.analysis_dir('step0_facts'), 'facts.json')):
            step0_audit(cfg, self._roots, self.device)
        self.cache_sources()

    def train(self, run):
        """Train one run unless another live session holds it.  Returns its status."""
        rid = run['run_id']
        if self.complete(rid):
            return 'complete'
        if not try_lock(self.cfg, rid):
            print(f'[{rid}] being trained by another session -> skipped here')
            return 'locked'
        try:
            rec = Trainer(self.cfg, run, self.manifest(), self.device).fit()
        finally:
            release_lock(self.cfg, rid)
            self.free()
        return rec.get('status')

    def stage_train_full(self):
        st = self.train(self.cfg.full_run())
        if st == 'paused':
            self.stop = True

    def stage_train_main(self):
        """Final model of Tables 4-5: warm restart of KASIRNet_full (Config.MAIN_*)."""
        cfg = self.cfg
        if cfg.MAIN_EPOCHS <= 0:
            return
        if cfg.MAIN_INIT_FROM and not self.complete(cfg.MAIN_INIT_FROM):
            print(f'[train_main] waits for {cfg.MAIN_INIT_FROM}'); return
        st = self.train(cfg.main_run())
        if st == 'paused':
            self.stop = True

    def stage_eval_full(self):
        rid = self.cfg.report_run_id()
        if not self.complete(rid):
            print(f'[eval_full] {rid} not trained yet'); return
        benches = self.cfg.MAIN_BENCHMARKS + [b for b in self.cfg.EXTRA_BENCHMARKS if not b.startswith('mix_')]
        self.evaluate_many(rid, benches, save=True)

    def stage_competitors(self):
        """Public competitor models with their official weights restore every test set they apply to
        (COMPETITOR_ROOT/<benchmark>/<method>/, resumable); scored later with the same metric function."""
        cfg = self.cfg
        if not cfg.RUN_COMPETITORS:
            return
        for method, bench in competitor_plan(cfg):
            try:
                self.bench_roots([bench])
                generate_competitor_outputs(cfg, self._roots, method, bench, self.device)
            except Exception as e:
                print(f'[competitors] {method} on {bench} skipped: {e}')
            self.free()
        self.merge_competitors()

    def merge_competitors(self):
        """Generated competitor folders join the ones held by the authors (significance tests, perceptual
        table, Figs. S1/S2: up to 3 methods with outputs)."""
        cfg = self.cfg
        for b, methods in competitor_folders(cfg).items():
            for m, folder in methods.items():
                if os.path.isdir(folder):
                    cfg.COMPETITOR_OUTPUTS.setdefault(b, {}).setdefault(m, folder)
        for b in list(cfg.FIGURE_COMPETITORS):
            have = {m: f for m, f in cfg.FIGURE_COMPETITORS[b].items() if os.path.isdir(f)}
            for m, f in cfg.COMPETITOR_OUTPUTS.get(b, {}).items():
                if len(have) < 3 and os.path.isdir(f):
                    have.setdefault(m, f)
            cfg.FIGURE_COMPETITORS[b] = have

    def stage_analyses_full(self):
        cfg, rid = self.cfg, self.cfg.report_run_id()
        if not self.complete(rid):
            print(f'[analyses_full] {rid} not trained yet'); return
        self.bench_roots(cfg.MAIN_BENCHMARKS + cfg.EXTRA_BENCHMARKS)
        self.merge_competitors()
        ev = self.evaluator()
        steps = dict(
            cost=lambda: cost_report(cfg, cfg.all_runs()[rid], self.device),
            significance=lambda: significance_report(cfg, ev, rid),
            probe=lambda: prompt_probe_report(cfg, self._roots, ev, self.model(rid), rid),
            zero_shot=lambda: zero_shot_report(cfg, ev, rid),
            spectral=lambda: spectral_report(cfg, self._roots, self.model(rid), self.device),
            srb_consistency=lambda: srb_consistency_report(cfg, self._roots, self.model(rid), self.device),
            protocol_effect=lambda: protocol_effect_report(cfg, ev, self.model(rid), rid),
            figures=lambda: qualitative_figures(cfg, self._roots, ev, self.model(rid), rid),
            stability=lambda: feature_stability_report(cfg, self._roots, self.device),
            explain=lambda: self.explainability(rid),
            tables=lambda: main_tables(cfg, rid))
        done_path = os.path.join(cfg.analysis_dir('status'), f'{rid}.json')
        done = read_json(done_path, {})
        for name, fn in steps.items():
            if done.get(name):
                continue
            try:
                fn()
                done[name] = True
                write_json(done, done_path)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'[analyses_full] {name} failed: {e}')
            self.free()

    def explainability(self, rid):
        model = self.model(rid)
        inf = KASIRNetInferencer(model, self.cfg.analysis_dir('explainability'), self.evaluator().metrics,
                                 explain_max_size=512, enable_occlusion=self.cfg.EXPLAIN_OCCLUSION)
        for b in self.cfg.MAIN_BENCHMARKS:
            try:
                ds = BenchmarkDataset(self.cfg, self._roots, b, max_images=self.cfg.EXPLAIN_IMAGES_PER_BENCHMARK)
                inf.run_dataset_inference(b, [ds[i] for i in range(len(ds))])
            except Exception as e:
                print(f'[explain] {b} skipped: {e}')
        inf.finalize_global_reports()

    def stage_ablations(self):
        cfg = self.cfg
        plan = cfg.ablation_plan()
        full = read_json(os.path.join(cfg.run_dir(cfg.FULL_RUN_ID), 'run.json'), {})
        todo = [r for r in plan if not self.complete(r['run_id'])]
        if full.get('budget'):
            hist = []
            ck = os.path.join(cfg.run_dir(cfg.FULL_RUN_ID), 'history_train.csv')
            if os.path.exists(ck):
                import csv
                hist = [float(r['sec_per_iter']) for r in csv.DictReader(open(ck))]
            if hist:
                it = full['budget']['iters_per_epoch'] * cfg.ABLATION_EPOCHS
                print(f'[ablations] {len(todo)}/{len(plan)} runs left x {it} iterations x {np.median(hist):.3f} s/it '
                      f'= about {len(todo) * it * np.median(hist) / 3600:.0f} GPU-hours')
        self.bench_roots(cfg.ABLATION_BENCHMARKS)
        if cfg.full_alias() and self.complete(cfg.FULL_RUN_ID):   # the reference "full, seed 1" of every ablation table
            self.evaluate_many(cfg.FULL_RUN_ID, cfg.ABLATION_BENCHMARKS, metric_names=('psnr', 'ssim'))
        for run in plan:
            st = self.train(run)
            if st == 'paused':
                self.stop = True
                return
            if st == 'complete':
                self.evaluate_many(run['run_id'], cfg.ABLATION_BENCHMARKS, metric_names=('psnr', 'ssim'))

    def stage_analyses_ablation(self):
        cfg = self.cfg
        sev_benches = sorted({b for _, _, b, _ in SEVERITY_LEVELS})
        self.bench_roots(sev_benches)
        run_ids = [cfg.FULL_RUN_ID] + [r for r in cfg.SEVERITY_RUNS]
        for rid in run_ids:
            if self.complete(rid):
                self.evaluate_many(rid, sev_benches, metric_names=('psnr', 'ssim'))
        severity_report(cfg, [r for r in run_ids if self.complete(r)])
        ablation_tables(cfg)

    def stage_tables(self):
        main_tables(self.cfg, self.cfg.report_run_id())
        ablation_tables(self.cfg)
        section41_report(self.cfg)
        print(f'[tables] all tables in {self.cfg.tables_dir}')


def main(cfg=None):
    cfg = cfg or Config()
    setup_environment(cfg)
    exp = Experiment(cfg)
    for stage in cfg.STAGES:
        if exp.stop:
            print('[main] stopped: a training run paused at the session time limit -> run main() again to resume')
            break
        print(f'\n========== stage: {stage} ==========')
        getattr(exp, 'stage_' + stage)()
    return exp


if __name__ == '__main__':
    main()
