"""Build KASIRNet.ipynb from the repository modules (single source of truth).

Every code cell of the notebook is the verbatim content of one module, minus the
intra-repository import lines marked '# nb:skip' (in the notebook all modules share
one namespace).  Run:  python tools/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ['Config.py', 'metrics.py', 'Model.py', 'Loss.py', 'Datasets.py', 'Trainer.py', 'Inferencer.py', 'Analysis.py', 'main.py']

INTRO = """# KASIRNet — revision notebook (Information Sciences, INS-D-26-12219)

One notebook = full training → full inference → every analysis requested by the reviewers → all ablations → manuscript tables.
Each code cell below is **identical** to the module of the same name in the repository (`tools/build_notebook.py` generates this notebook from them).

**Hardware.** Colab *G4* runtime (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB). The exact GPU, driver, CUDA/cuDNN/PyTorch/pyiqa versions are written to every `run.json`.

**How to run.** Runtime → G4 → *Run all*. All settings are in `Config` (no argparse). Stages are idempotent and every training run checkpoints atomically every 2,000 iterations to Google Drive, so when a Colab session ends simply *Run all* again: the current run resumes from its last checkpoint (bit-exact data order and RNG state) and finished work is skipped.

| stage | what it does | main outputs (under `OUTPUT_ROOT`) |
|---|---|---|
| `audit` | restores the datasets from the Google-Drive cache (or downloads them once and caches them), builds the frozen training manifest with the train/test leakage audit, validates every category and benchmark (pairing, alignment, direction, sizes), writes the Step-0 facts (metric definition, SSIM-loss gradient, SRB gate initialisation, submitted checkpoint, data pairing) | `datasets_cache/*.tar`, `data_manifest/{train_manifest,data_validation}.json`, `analysis/step0_facts/` |
| `train_full` | full KASIRNet (batch 64, Adam 1e-4, cosine, clip 1.0, FP16, six loss weights 5/0.5/1/1/1/0.5), `EPOCHS` = 100 = 35,700 iterations | `checkpoints/KASIRNet_full/final.pth`, `runs/KASIRNet_full/{run.json, curves.png, history_*.csv}` |
| `eval_full` | 8 benchmarks + Rain100H + Snow100K-S/M + Gaussian σ15/25/50 + LIVE1/LOL/DPDD at full resolution | `runs/KASIRNet_full/eval/<bench>/per_image.csv` (PSNR, SSIM, LPIPS, DISTS), `summary.json` |
| `analyses_full` | cost (TODO-13), bootstrap CI + Wilcoxon (TODO-12C), LPIPS/DISTS + divergent % (TODO-14), prompt probe + mixed sets (TODO-16), spectral/phase analysis, feature stability, Figs. S1/S2 (TODO-06), explainability | `tables/`, `analysis/` |
| `ablations` | 45 runs with the same 100 epochs as the full model (`ABLATION_EPOCHS = EPOCHS`; the seed-1 full reference is the full model itself, trained once): TODO-12A (5 variants × 3 seeds), TODO-09B (extractors), TODO-12B (interactions), TODO-15 (loss weights ×0.5/×2), Table 6 rows, Table 7 rows, SCWAM design | `runs/<run_id>/run.json` + GoPro/Rain100L/SIDD per-image CSVs |
| `analyses_ablation` | SIM vs severity (TODO-09A) and all ablation tables | `tables/table6…`, `table7…`, `table8…`, `table9…`, `table12…` |
| `tables` | Tables 4/5, Rain100H (TODO-02), zero-shot (TODO-17), Section 4.1 reproducibility summary | `tables/*.csv`, `tables/*.tex` |

**Parallel sessions.** If more than one G4 session is available, open this notebook in each of them (same Google account / Drive) and *Run all*: a run being trained holds a lock with a heartbeat, so the sessions share the ablation queue without training the same run twice (`Config.LOCK_STALE_MINUTES`). The measured s/iteration and the GPU-hours left are printed at the start of the `ablations` stage.

Competitor outputs that the authors hold (HIDE Restormer, TextOCR CNN-ViT, Köhler Rahman et al., RESIDE-6K ADSM, SIDD Defusion; Rain100L / Snow100K-L methods for Figs. S1/S2) are read from `Config.COMPETITOR_ROOT` and scored with the same metric function.
"""

SETUP = """# ---- environment (Colab G4).  pyiqa is pinned to the version that produced the submitted figures.
import importlib, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyiqa==0.1.16', '--no-deps'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lmdb', 'gdown', 'kagglehub'], check=False)
PIP_NAME = {'cv2': 'opencv-python-headless', 'yaml': 'pyyaml', 'skimage': 'scikit-image', 'sklearn': 'scikit-learn',
            'clip': 'openai-clip', 'PIL': 'pillow'}
for _ in range(10):  # install any import-time dependency of pyiqa missing from the runtime
    try:
        import pyiqa
        break
    except ModuleNotFoundError as e:
        mod = e.name.split('.')[0]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', PIP_NAME.get(mod, mod)], check=False)
import torch
print('torch', torch.__version__, '| pyiqa', pyiqa.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""

RUN = """# ---- run everything (re-run this cell in a new session to resume)
cfg = Config()
# Optional overrides, e.g. to split the work across sessions:
# cfg.STAGES = ['audit', 'train_full', 'eval_full', 'analyses_full']
# cfg.RUN_ABLATION_GROUPS = ['TODO12A_seeds', 'TODO09B_extractor', 'TODO12B_interactions', 'TODO15_loss_sensitivity']
exp = main(cfg)
"""


def module_source(name):
    lines = open(os.path.join(ROOT, name)).read().rstrip('\n').split('\n')
    kept = [l for l in lines if '# nb:skip' not in l]
    header = f'# ===== {name} (identical to {name} in the repository, intra-repository imports removed) ====='
    return header + '\n' + '\n'.join(kept)


def cell(kind, src):
    c = {'cell_type': kind, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def build(path=os.path.join(ROOT, 'KASIRNet.ipynb')):
    cells = [cell('markdown', INTRO), cell('code', SETUP)]
    for m in MODULES:
        cells.append(cell('code', module_source(m).replace("\nif __name__ == '__main__':\n    main()", '')))
    cells.append(cell('code', RUN))
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'G4', 'machine_shape': 'hm', 'provenance': []},
                                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 0}
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print('written', path, len(cells), 'cells')


if __name__ == '__main__':
    build()
