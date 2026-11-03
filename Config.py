# ==========================================================================
# KASIRNet - configuration of the revision (INS-D-26-12219)
# --------------------------------------------------------------------------
# Every setting of every run lives in this single Config class (no argparse,
# no sys.argv).  A run is "Config + one line of overrides" (see ABLATION_GROUPS).
# Section 4.1 of the manuscript is generated from the values below and from the
# run.json that each training run writes (GPU, iterations, wall-clock, seeds).
# ==========================================================================
import os
import copy
import json


class Config:
    # ------------------------------------------------------------------ stages
    # main() executes these stages in order; each one is idempotent (a finished run
    # or analysis is detected from its JSON and skipped), so the notebook can be
    # re-run in as many Colab sessions as needed.
    STAGES = ['audit', 'train_full', 'eval_full', 'analyses_full', 'ablations', 'analyses_ablation', 'tables']

    # ---------------------------------------------------------------- hardware
    # Colab "G4" runtime = NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
    # The actual device name, driver, CUDA/cuDNN/PyTorch versions are recorded in run.json.
    DEVICE = 'cuda'
    NUM_WORKERS = min(16, os.cpu_count() or 4)
    PIN_MEMORY = True
    AMP = True                    # mixed precision (paper: FP16)
    AMP_DTYPE = 'float16'         # 'float16' (paper) or 'bfloat16'
    EVAL_AMP = False              # all reported metrics are computed with an FP32 forward pass
    CUDNN_BENCHMARK = True
    DETERMINISTIC = False         # True -> cudnn.deterministic (slower); seeds are always fixed
    MAX_SESSION_HOURS = 23.5      # save + stop cleanly before a Colab session is recycled
    # Several Colab sessions (each with its own G4) can run main() at the same time on the same
    # Drive folder: a run being trained is locked (heartbeat at every checkpoint) and skipped by the
    # other sessions; a lock older than LOCK_STALE_MINUTES is considered abandoned.
    LOCK_STALE_MINUTES = 45

    # ------------------------------------------------------------------- paths
    DRIVE_MOUNT = '/content/drive'
    OUTPUT_ROOT = '/content/drive/MyDrive/KASIRNet_revision'   # checkpoints, results, tables (persistent)
    DATA_ROOT = '/content/datasets'                            # local disk: downloads and restored caches
    # ---- dataset cache on Google Drive.  First session: every source is downloaded (kagglehub,
    # Hugging Face, Google Drive) and then packed into ONE uncompressed tar per source in
    # DATA_CACHE_ROOT; later sessions extract these tars to local disk instead of downloading.
    #   'used' : only the files the experiments read (frozen training manifest + every benchmark);
    #            if the training manifest has to be rebuilt the full source is downloaded again
    #   'full' : the complete sources (about 55 GB of Drive space)   |   'off': no cache
    DATA_CACHE_ROOT = '/content/drive/MyDrive/KASIRNet_revision/datasets_cache'
    DATA_CACHE_MODE = 'used'
    DATA_CACHE_REFRESH = False    # True: rebuild the tars of this session's sources
    # Google-Drive archives often fail with "quota exceeded".  A copy of the archive placed by hand in
    # DATA_CACHE_ROOT/archives/<file name> (Drive web page -> "Make a copy" / "Add shortcut") is used first.
    LEGACY_CHECKPOINT = '/content/drive/MyDrive/Checkpoints/KASIRNet/best.pth'        # submitted model (audit only)
    LEGACY_NOSIM_CHECKPOINT = ''                                                       # submitted "w/o SIM" model, if kept
    # Competitor outputs held by the authors (restored images named like the test inputs).
    COMPETITOR_ROOT = '/content/drive/MyDrive/KASIRNet_revision/competitors'
    COMPETITOR_OUTPUTS = {          # benchmark -> {method name: folder}   (TODO-12C, TODO-14)
        'hide':     {'Restormer': f'{COMPETITOR_ROOT}/hide/Restormer'},
        'textocr':  {'CNN-ViT': f'{COMPETITOR_ROOT}/textocr/CNN-ViT'},
        'kohler':   {'Rahman et al.': f'{COMPETITOR_ROOT}/kohler/Rahman'},
        'reside6k': {'ADSM': f'{COMPETITOR_ROOT}/reside6k/ADSM'},
        'sidd':     {'Defusion': f'{COMPETITOR_ROOT}/sidd/Defusion'},
    }
    FIGURE_COMPETITORS = {          # supplementary Figs. S1/S2 (TODO-06): up to 3 methods each
        'rain100l':   {'PromptIR': f'{COMPETITOR_ROOT}/rain100l/PromptIR', 'AdaIR': f'{COMPETITOR_ROOT}/rain100l/AdaIR',
                       'Restormer': f'{COMPETITOR_ROOT}/rain100l/Restormer'},
        'snow100k_l': {'TransWeather': f'{COMPETITOR_ROOT}/snow100k_l/TransWeather', 'WGWS-Net': f'{COMPETITOR_ROOT}/snow100k_l/WGWS',
                       'Histoformer': f'{COMPETITOR_ROOT}/snow100k_l/Histoformer'},
    }

    # --------------------------------------------------------- data sources
    # kagglehub.dataset_download(slug) returns the local root of each source.
    KAGGLE_SOURCES = {
        'dblur':    'jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets',  # GoPro train/test, HIDE test only, TextOCR, CelebA, Helen
        'kohler':   'eshapahwa1302/kohlerdeblur',              # 4 scenes x 12 kernels = 48 pairs
        'realblur_j': 'adwythdarsanr/realblurj-image-deblurring-dataset',  # RealBlur-J official split: train 3758 (182 scenes) / test 980 (50 scenes)
        'jrdr':     'shivakanthsujit/jrdr-deraining-dataset',  # RainTrainH / RainTrainL (1800 each)
        'raindrop': 'chenchienying/deraindataset',             # DeRaindrop: train 861 / test_a 58 / test_b 249
        'rain100l': 'bshaurya/rain-dataset',                   # Rain100L test, original 481x321 (rain-/norain-xxx.png)
        'rain100h': 'alicewen820/rain100h',                    # Rain100H test (100)
        'reside6k': 'kmljts/reside-6k',                        # RESIDE-6K train 6000 / test 1000
        'sidd':     'soumikrakshit/smartphone-image-denoising-dataset',  # SIDD-Medium sRGB (320) + SIDD val crops (lmdb)
        'sp_noise': 'rajneesh231/salt-and-pepper-noise-images',
        'docs':     'mohamedkhaled201841/image-denoising',     # noisy documents (train/train_cleaned)
        'alphaisp': 'tenxengineers/denoising-dataset-multiple-iso-levels',
        'cbsd68':   'tarekmebrouk/cbsd68',                     # clean images for Gaussian sigma 15/25/50
        'lol':      'soumikrakshit/lol-dataset',               # LOL eval15
        'live1':    'nikshko/live-1',                          # LIVE1 references (29 bmp)
        'legacy_raining': 'adeshpal/image-deraining',          # audit only: the "raining" category of the submitted pool
    }
    # Google-Drive archives (official links of the dataset authors).  Layouts checked on the archives:
    # training tar -> all/{gt,mask,synthetic}; test tar -> media/jdway/GameSSD/overlapping/test/Snow100K-{S,M,L}/{gt,mask,synthetic}
    GDOWN_SOURCES = {
        'snow100k_train': dict(id='1SGS2w2RBhfwAMn9dMQZUnoQcqCcz1baU', file='Snow100K-training.tar.gz'),
        'snow100k_test':  dict(id='15z-A3i3Z2ybdKuyxwYRA8YKxIA8_kTDm', file='Snow100K-testset.tar.gz'),
    }
    # Hugging Face parquet mirrors.  DPDD test (76 pairs, 1680x1120, original file names, combined image
    # 'source' = inputC, 'target'); the official Google-Drive file (DPDD_GDOWN_ID) is the fallback.
    HF_PARQUET_SOURCES = {
        'dpdd_test': dict(repo='JacobLinCool/DPDD', files=[f'combined/test-0000{i}-of-00003.parquet' for i in range(3)],
                          columns={'source': 'test/inputC', 'target': 'test/target'}, key_column='source'),
    }
    DPDD_GDOWN_ID = '1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw'

    # -------------------------------------------------------- training pool
    # family is used by the prompt probe and by the per-family reports.
    # match: 'stem' (same file stem) or ('regex', input_regex, gt_regex) -> key = joined groups.
    # exclude_overlap_with: training pairs whose clean image is a near-duplicate of a clean
    # image of these benchmarks are removed (leakage audit, see Datasets.LeakageAuditor).
    _JRDR_MATCH = ('regex', r'norain-(\d+)x2', r'norain-(\d+)')
    TRAIN_CATEGORIES = {
        # (DBlur has no HIDE training split: HIDE is a test set only, the standard GoPro -> HIDE protocol)
        'gopro':      dict(family='blur',  input=('dblur', 'DBlur/Gopro/train/blur'),   gt=('dblur', 'DBlur/Gopro/train/sharp')),
        'textocr':    dict(family='blur',  input=('dblur', 'DBlur/TextOCR/train/blur'), gt=('dblur', 'DBlur/TextOCR/train/sharp')),
        'celeba':     dict(family='blur',  input=('dblur', 'DBlur/CelebA/train/blur'),  gt=('dblur', 'DBlur/CelebA/train/sharp')),
        'helen':      dict(family='blur',  input=('dblur', 'DBlur/Helen/train/blur'),   gt=('dblur', 'DBlur/Helen/train/sharp')),
        # Real recorded camera-shake blur: replaces the Koehler category of the submitted pool (Koehler has no
        # training split; its 48 images are now a test set only).  Official RealBlur-J training split, scenes
        # disjoint from the RealBlur test scenes; checked against the Koehler ground truths by the leakage audit.
        'realblur_j': dict(family='blur',  input=('realblur_j', 'RealBlur/train/blur'), gt=('realblur_j', 'RealBlur/train/gt')),
        # RainTrainH: rainy images are rain/X2/norain-NNNx2.png, clean are norain/norain-NNN.png
        'rain_heavy': dict(family='rain',  input=('jrdr', 'JRDR/rain_data_train_Heavy/rain/X2'), gt=('jrdr', 'JRDR/rain_data_train_Heavy/norain'),
                           match=_JRDR_MATCH, exclude_overlap_with=['rain100h', 'rain100l']),
        'rain_light': dict(family='rain',  input=('jrdr', 'JRDR/rain_data_train_Light/rain'), gt=('jrdr', 'JRDR/rain_data_train_Light/norain'),
                           match=_JRDR_MATCH, exclude_overlap_with=['rain100l', 'rain100h']),
        # raindrops on the glass blur and veil the scene: the clean image has the higher contrast
        'raindrop':   dict(family='rain',  input=('raindrop', 'train/data'), gt=('raindrop', 'train/gt'),
                           match=('regex', r'(\d+)_rain', r'(\d+)_clean'), direction='contrast_gt'),
        'haze':       dict(family='haze',  input=('reside6k', 'RESIDE-6K/train/hazy'), gt=('reside6k', 'RESIDE-6K/train/GT'),
                           exclude_overlap_with=['reside6k']),
        'snow':       dict(family='snow',  input=('snow100k_train', '**/synthetic'), gt=('snow100k_train', '**/gt')),
        'sidd':       dict(family='noise', root=('sidd', 'train/train'), input_glob='**/*NOISY_SRGB_*.PNG', gt_replace=('NOISY', 'GT'),
                           repeat=8),   # 320 very large images: 8 random crops of each per epoch
        'sp_noise':   dict(family='noise', input=('sp_noise', 'Noisy_folder'), gt=('sp_noise', 'Ground_truth'),
                           match=('regex', r'noisy_(.+)', r'(.+)')),
        # stained documents: the cleaned page (black text on white) has the higher contrast
        'docs':       dict(family='noise', input=('docs', 'train'), gt=('docs', 'train_cleaned'), direction='contrast_gt'),
        'alphaisp':   dict(family='noise', input=('alphaisp', 'AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Noised'),
                           gt=('alphaisp', 'AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Ground Truth')),
    }
    # Koehler has no official training split.  'test_only' (default, standard protocol):
    # all 48 images are a test set and none is trained on.  'scene_split': scenes 1-3 are a
    # training category, scene 4 (12 images) is the test set.
    KOHLER_PROTOCOL = 'test_only'
    MAX_PER_CATEGORY = 2500       # seeded random subset when a category is larger
    VAL_PER_CATEGORY = 10         # held out from training, used only to select the checkpoint
    DATA_SUBSET_SEED = 2026
    # Leakage audit: every training candidate is compared with the clean images of every benchmark of
    # the same degradation family (and of `exclude_overlap_with`); near-duplicates are removed before
    # the per-category cap is applied.  Screening: correlation of 24x24 grey thumbnails > OVERLAP_THRESHOLD,
    # confirmation: 64x64 thumbnails > OVERLAP_CONFIRM and aspect ratio within 5 %.
    EXCLUDE_TEST_OVERLAP = True
    OVERLAP_THRESHOLD = 0.97
    OVERLAP_CONFIRM = 0.95
    # Pair validation (audit stage): per category / benchmark, VALIDATE_SAMPLES pairs are checked for
    # readability, size agreement, alignment (64x64 grey correlation >= MIN_PAIR_CORR) and degradation
    # direction (input blurrier / noisier / hazier than the ground truth).  A category whose median
    # correlation or direction rate is below the limits stops the run (MISPAIR / SWAP).
    VALIDATE_SAMPLES = 40
    MIN_PAIR_CORR = 0.5
    MIN_DIRECTION_RATE = 0.6
    VALIDATION_STRICT = True          # stop before training if a training category is flagged
    ALLOW_MISSING_CATEGORIES = False  # a training category without data is an error (not silently skipped)

    # ------------------------------------------------------------- benchmarks
    _KOHLER_MATCH = ('regex', r'Blurry(\d+_\d+)', r'Sharp(\d+_\d+)')
    BENCHMARKS = {
        # ---- the eight benchmarks of Tables 4-5
        'gopro':      dict(family='blur',  input=('dblur', 'DBlur/Gopro/test/blur'),   gt=('dblur', 'DBlur/Gopro/test/sharp')),
        'hide':       dict(family='blur',  input=('dblur', 'DBlur/HIDE/test/blur'),    gt=('dblur', 'DBlur/HIDE/test/sharp')),
        'textocr':    dict(family='blur',  input=('dblur', 'DBlur/TextOCR/test/blur'), gt=('dblur', 'DBlur/TextOCR/test/sharp')),
        'kohler':     dict(family='blur',  input=('kohler', 'kohler/blur'), gt=('kohler', 'kohler/sharp'), match=_KOHLER_MATCH),
        'rain100l':   dict(family='rain',  input=('rain100l', '.'), gt=('rain100l', '.'), match=('regex', r'^rain-(\d+)$', r'^norain-(\d+)$')),
        'reside6k':   dict(family='haze',  input=('reside6k', 'RESIDE-6K/test/hazy'), gt=('reside6k', 'RESIDE-6K/test/GT')),
        'snow100k_l': dict(family='snow',  input=('snow100k_test', '**/Snow100K-L/synthetic'), gt=('snow100k_test', '**/Snow100K-L/gt')),
        'sidd':       dict(family='noise', lmdb=('sidd', 'SIDD-val-lmdb/SIDD/val/input_crops.lmdb'), lmdb_gt=('sidd', 'SIDD-val-lmdb/SIDD/val/gt_crops.lmdb')),
        # ---- TODO-02
        'rain100h':   dict(family='rain',  input=('rain100h', 'Rain100H-test/rain'), gt=('rain100h', 'Rain100H-test/norain')),
        # ---- TODO-09 Part A severity levels
        'snow100k_s': dict(family='snow',  input=('snow100k_test', '**/Snow100K-S/synthetic'), gt=('snow100k_test', '**/Snow100K-S/gt')),
        'snow100k_m': dict(family='snow',  input=('snow100k_test', '**/Snow100K-M/synthetic'), gt=('snow100k_test', '**/Snow100K-M/gt')),
        'cbsd68_g15': dict(family='noise', clean=('cbsd68', 'dataset/train'), degrade=[('gaussian', 15)]),
        'cbsd68_g25': dict(family='noise', clean=('cbsd68', 'dataset/train'), degrade=[('gaussian', 25)]),
        'cbsd68_g50': dict(family='noise', clean=('cbsd68', 'dataset/train'), degrade=[('gaussian', 50)]),
        # ---- TODO-17 zero-shot (degradations never trained on)
        'live1_q10':  dict(family='jpeg',     clean=('live1', 'live1'), degrade=[('jpeg', 10)]),
        'lol':        dict(family='lowlight', input=('lol', 'lol_dataset/eval15/low'), gt=('lol', 'lol_dataset/eval15/high')),
        'dpdd':       dict(family='defocus',  input=('dpdd_test', 'test/inputC'), gt=('dpdd_test', 'test/target')),
        # ---- TODO-16 mixed degradations (second degradation added to the first one)
        'mix_gopro_g25':      dict(family='mixed', base='gopro', degrade=[('gaussian', 25)], families=('blur', 'noise')),
        'mix_rain100l_haze':  dict(family='mixed', base='rain100l', degrade=[('haze', 1.0, 0.8)], families=('rain', 'haze')),
        'mix_snow100kl_g25':  dict(family='mixed', base='snow100k_l', degrade=[('gaussian', 25)], families=('snow', 'noise'), max_images=2000),
    }
    MAIN_BENCHMARKS = ['gopro', 'hide', 'textocr', 'kohler', 'rain100l', 'reside6k', 'snow100k_l', 'sidd']
    EXTRA_BENCHMARKS = ['rain100h', 'snow100k_s', 'snow100k_m', 'cbsd68_g15', 'cbsd68_g25', 'cbsd68_g50',
                        'live1_q10', 'lol', 'dpdd', 'mix_gopro_g25', 'mix_rain100l_haze', 'mix_snow100kl_g25']
    ABLATION_BENCHMARKS = ['gopro', 'rain100l', 'sidd']      # reference = mean over these three (Table 6)
    SYNTHETIC_SEED = 1234         # per-image seeds of synthetic noise = SYNTHETIC_SEED + crc32(file name)
    HAZE_DEPTH = (0.5, 1.0)       # pseudo-depth ramp (top -> bottom = far -> near) of the ASM haze

    # ----------------------------------------------- training (Section 4.1)
    PATCH_SIZE = 256
    BATCH_SIZE = 6
    LR = 1e-4
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 0.0
    ETA_MIN = 1e-7                # cosine annealing per iteration from LR to ETA_MIN over the whole budget
    GRAD_CLIP = 1.0
    # Budget.  One epoch = one pass over the (capped, repeated) training pool, i.e.
    # floor(N_train / BATCH_SIZE) iterations (about 22,900 samples -> 3,800 iterations for this pool).
    # The full model gets 100 epochs (~0.38 M iterations, ~2.3 M patches).  Every ablation run gets
    # the identical reduced budget B = 10 epochs (one tenth): 46 runs x B = 4.6 full-run equivalents.
    # main() prints the measured s/iteration and the GPU-hours left for the whole plan; raise
    # ABLATION_EPOCHS if the deadline allows (all ablation runs must share the same value).
    EPOCHS = 100
    ABLATION_EPOCHS = 10
    CKPT_EVERY_ITERS = 2000       # resumable checkpoint (atomic write) every N iterations
    VAL_EVERY_ITERS = 5000
    LOG_EVERY_ITERS = 100
    MODEL_SELECTION = 'best_val'  # 'best_val' (held-out split of the training pool, never a test set) or 'last'
    LOSS_WEIGHTS = dict(pix=5.0, fft=0.5, perc=1.0, ssim=1.0, id=1.0, edge=0.5)
    SEED = 1

    # ------------------------------------------------------------------ model
    MODEL = dict(base_channels=64, prompt_dim=256, use_sim=True, extractor='vgg19', extractor_pretrained=True,
                 use_srb=True, srb_replacement='plain_conv', use_scwam=True, scwam_replacement='conv',
                 use_meshgrid=True, window_size=8, shift_size=4, masked_shift=False, rel_pos_bias=False,
                 use_prompt=True, use_gcb=True, num_scales=4)

    # ----------------------------------------------------------- run plan
    FULL_RUN_ID = 'KASIRNet_full'
    ABLATION_SEEDS = (1, 2, 3)
    RUN_ABLATION_GROUPS = ['TODO12A_seeds', 'TODO09B_extractor', 'TODO12B_interactions', 'TODO15_loss_sensitivity',
                           'T6_components', 'T7_loss_terms', 'SCWAM_design']

    # --------------------------------------------------------------- analyses
    BOOTSTRAP_RESAMPLES = 10000
    BOOTSTRAP_SEED = 7
    PROBE_IMAGES_PER_FAMILY = 100
    PROBE_FAMILIES = {'blur': 'gopro', 'rain': 'rain100l', 'haze': 'reside6k', 'snow': 'snow100k_l', 'noise': 'sidd'}
    SEVERITY_RUNS = ('full_s1', 'noSIM_s1')   # TODO-09A: same reduced budget, with / without SIM
    PROFILE_SIZES = [(256, 256), (720, 1280)]
    PROFILE_WARMUP, PROFILE_REPEATS = 10, 50
    EXPLAIN_IMAGES_PER_BENCHMARK = 3
    EXPLAIN_OCCLUSION = False
    SAVE_RESTORED_IMAGES = True

    # ======================================================================
    # helpers
    # ======================================================================
    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(type(self), k):
                raise AttributeError(f'Unknown Config field {k}')
            setattr(self, k, copy.deepcopy(v))
        self.MODEL = copy.deepcopy(self.MODEL)
        self.LOSS_WEIGHTS = copy.deepcopy(self.LOSS_WEIGHTS)

    # paths
    def ckpt_dir(self, run_id):   return os.path.join(self.OUTPUT_ROOT, 'checkpoints', run_id)
    def run_dir(self, run_id):    return os.path.join(self.OUTPUT_ROOT, 'runs', run_id)
    def analysis_dir(self, name): return os.path.join(self.OUTPUT_ROOT, 'analysis', name)
    @property
    def tables_dir(self):         return os.path.join(self.OUTPUT_ROOT, 'tables')
    @property
    def manifest_dir(self):       return os.path.join(self.OUTPUT_ROOT, 'data_manifest')

    def train_categories(self):
        cats = copy.deepcopy(self.TRAIN_CATEGORIES)
        if self.KOHLER_PROTOCOL == 'scene_split':
            cats['kohler'] = dict(family='blur', input=('kohler', 'kohler/blur'), gt=('kohler', 'kohler/sharp'),
                                  match=self._KOHLER_MATCH, include=r'Blurry[123]_')
        return cats

    def benchmarks(self):
        b = copy.deepcopy(self.BENCHMARKS)
        if self.KOHLER_PROTOCOL == 'scene_split':
            b['kohler']['include'] = r'Blurry4_'
        return b

    # run plan ------------------------------------------------------------
    def full_run(self):
        return dict(run_id=self.FULL_RUN_ID, group='full', seed=self.SEED, epochs=self.EPOCHS, model={}, loss={})

    def ablation_groups(self):
        S = self.ABLATION_SEEDS
        E = self.ABLATION_EPOCHS
        w = self.LOSS_WEIGHTS

        def run(rid, group, seed=1, model=None, loss=None):
            return dict(run_id=rid, group=group, seed=seed, epochs=E, model=model or {}, loss=loss or {})

        variants = [('full', {}), ('noSIM', dict(use_sim=False)), ('noSRB', dict(use_srb=False)),
                    ('noSCWAM', dict(use_scwam=False)), ('noPrompt', dict(use_prompt=False))]
        return {
            # TODO-12 Part A: 5 variants x 3 seeds (also the VGG-19 reference of TODO-09B and default of TODO-15)
            'TODO12A_seeds': [run(f'{n}_s{s}', 'TODO12A_seeds', s, m) for n, m in variants for s in S],
            # TODO-09 Part B: alternative frozen extractors in the SIM
            'TODO09B_extractor': [run('ext_resnet50_s1', 'TODO09B_extractor', model=dict(extractor='resnet50')),
                                  run('ext_convnextT_s1', 'TODO09B_extractor', model=dict(extractor='convnext_t'))],
            # TODO-12 Part B: pairwise interactions
            'TODO12B_interactions': [run('noSIM+noSRB_s1', 'TODO12B_interactions', model=dict(use_sim=False, use_srb=False)),
                                     run('noSIM+noSCWAM_s1', 'TODO12B_interactions', model=dict(use_sim=False, use_scwam=False)),
                                     run('noSRB+noSCWAM_s1', 'TODO12B_interactions', model=dict(use_srb=False, use_scwam=False)),
                                     run('noSRB+noFFT_s1', 'TODO12B_interactions', model=dict(use_srb=False), loss=dict(fft=0.0))],
            # TODO-15: each loss weight halved and doubled
            'TODO15_loss_sensitivity': [run(f'w_{k}_x{f:g}_s1', 'TODO15_loss_sensitivity', loss={k: w[k] * f})
                                        for k in w for f in (0.5, 2.0)],
            # Table 6 remaining rows under the same budget
            'T6_components': [run('noMeshgrid_s1', 'T6_components', model=dict(use_meshgrid=False)),
                              run('noGCB_s1', 'T6_components', model=dict(use_gcb=False)),
                              run('depth3_s1', 'T6_components', model=dict(num_scales=3)),
                              run('width32_s1', 'T6_components', model=dict(base_channels=32))],
            # Table 7: each loss term removed
            'T7_loss_terms': [run(f'no_{k}_s1', 'T7_loss_terms', loss={k: 0.0}) for k in w],
            # SCWAM vs Swin design choices (Reviewer 3 comment 1, Reviewer 5 comment 1)
            'SCWAM_design': [run('scwam_noShift_s1', 'SCWAM_design', model=dict(shift_size=0)),
                             run('scwam_swinMask_s1', 'SCWAM_design', model=dict(masked_shift=True)),
                             run('scwam_RPB_s1', 'SCWAM_design', model=dict(use_meshgrid=False, rel_pos_bias=True))],
        }

    def ablation_plan(self):
        groups = self.ablation_groups()
        plan, seen = [], set()
        for g in self.RUN_ABLATION_GROUPS:
            for r in groups[g]:
                if r['run_id'] not in seen:
                    seen.add(r['run_id']); plan.append(r)
        return plan

    def all_runs(self):
        runs = {self.FULL_RUN_ID: self.full_run()}
        for g in self.ablation_groups().values():
            for r in g:
                runs.setdefault(r['run_id'], r)
        return runs

    def model_kwargs(self, run):
        m = dict(self.MODEL)
        m.update(run.get('model', {}))
        return m

    def loss_weights(self, run):
        w = dict(self.LOSS_WEIGHTS)
        w.update(run.get('loss', {}))
        return w

    def to_dict(self):
        d = {}
        for k in dir(type(self)):
            if k.isupper():
                v = getattr(self, k)
                try:
                    json.dumps(v)
                    d[k] = v
                except TypeError:
                    d[k] = str(v)
        return d
