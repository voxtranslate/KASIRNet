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
                      manifest_is_current, overlap_benchmarks, manifest_used_files, validate_data, resolve_path)  # nb:skip
from Trainer import Trainer, read_json, write_json, try_lock, release_lock  # nb:skip
from Inferencer import Evaluator, KASIRNetInferencer  # nb:skip
from Analysis import (step0_audit, cost_report, significance_report, prompt_probe_report, severity_report,  # nb:skip
                      feature_stability_report, spectral_report, qualitative_figures, ablation_tables,  # nb:skip
                      main_tables, section41_report, SEVERITY_LEVELS)  # nb:skip


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
        self._roots, self._origin, self._manifest, self._evaluator = {}, {}, None, None
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
        return self._evaluator

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
        """Step 0: all sources (Drive cache or download), frozen training manifest with the leakage
        audit, validation of every training category and benchmark, Step-0 facts, Drive cache."""
        cfg = self.cfg
        man = self.manifest()
        b = cfg.benchmarks()
        self.roots([k for n in b for k in spec_sources(b[n], b)] + ['jrdr', 'rain100l', 'legacy_raining'])
        if not os.path.exists(os.path.join(cfg.manifest_dir, 'data_validation.json')):
            validate_data(cfg, self._roots, man)
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

    def stage_eval_full(self):
        rid = self.cfg.FULL_RUN_ID
        if not self.complete(rid):
            print('[eval_full] full model not trained yet'); return
        benches = self.cfg.MAIN_BENCHMARKS + [b for b in self.cfg.EXTRA_BENCHMARKS if not b.startswith('mix_')]
        self.evaluate_many(rid, benches, save=True)

    def stage_analyses_full(self):
        cfg, rid = self.cfg, self.cfg.FULL_RUN_ID
        if not self.complete(rid):
            print('[analyses_full] full model not trained yet'); return
        self.bench_roots(cfg.MAIN_BENCHMARKS + cfg.EXTRA_BENCHMARKS)
        ev = self.evaluator()
        steps = dict(
            cost=lambda: cost_report(cfg, cfg.full_run(), self.device),
            significance=lambda: significance_report(cfg, ev, rid),
            probe=lambda: prompt_probe_report(cfg, self._roots, ev, self.model(rid), rid),
            spectral=lambda: spectral_report(cfg, self._roots, self.model(rid), self.device),
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
        main_tables(self.cfg, self.cfg.FULL_RUN_ID)
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
