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
    # Which sources go to the Drive cache.  'fragile' (default): the Google-Drive and Hugging Face downloads
    # (Snow100K test, Snow100K train, DPDD; about 7 GB) - Kaggle sources are fast and mirrored by Colab, so
    # caching them would only fill the Drive quota (about 40 GB).  'all': every source.
    DATA_CACHE_SOURCES = 'fragile'
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
        'snow_test_l': 'suzukimio/allweather-testing',         # Snow100K-L official test set (16,801 pairs, AllWeather_Testing/Snow100K-L);
                                                               # names and byte sizes identical to the official archive
    }
    # Google-Drive archives (official links of the dataset authors).  Layouts checked on the archives:
    # training tar -> all/{gt,mask,synthetic}; test tar -> media/jdway/GameSSD/overlapping/test/Snow100K-{S,M,L}/{gt,mask,synthetic}
    # (The official Snow100K Google-Drive archives - training 1SGS2w2RBhfwAMn9dMQZUnoQcqCcz1baU, test
    # 15z-A3i3Z2ybdKuyxwYRA8YKxIA8_kTDm - fail repeatedly with "quota exceeded"; they are no longer used.)
    GDOWN_SOURCES = {}
    # Snow100K from Hugging Face (the archive released with ConvIR, no quota): train2500 = 2,500 pairs of the
    # official training set; test2000 = 713 Snow100K-S + 662 Snow100K-M + 625 Snow100K-L official test pairs.
    # Checked against the official archive listings: every file has the official name and byte size.
    HF_ZIP_SOURCES = {
        'snow100k_hf': dict(repo='fancongyi/snow100k', file='Snow100K_ConvIR.zip'),
    }
    # names of the Snow100K-S / -M official test images contained in test2000 (zlib + base64)
    SNOW_LEVEL_NAMES = {
        'S': (
        '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'),
        'M': (
        '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'
        '4CntbCTTBfnW8CneJ+L7i6+3/CAaVS'),
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
        'snow':       dict(family='snow',  input=('snow100k_hf', 'Snow100K/train2500/Snow'), gt=('snow100k_hf', 'Snow100K/train2500/Gt')),
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
        'snow100k_l': dict(family='snow',  input=('snow_test_l', 'AllWeather_Testing/Snow100K-L/input'), gt=('snow_test_l', 'AllWeather_Testing/Snow100K-L/gt')),
        'sidd':       dict(family='noise', lmdb=('sidd', 'SIDD-val-lmdb/SIDD/val/input_crops.lmdb'), lmdb_gt=('sidd', 'SIDD-val-lmdb/SIDD/val/gt_crops.lmdb')),
        # ---- TODO-02
        'rain100h':   dict(family='rain',  input=('rain100h', 'Rain100H-test/rain'), gt=('rain100h', 'Rain100H-test/norain')),
        # ---- TODO-09 Part A severity levels
        # severity levels: the official S / M test images contained in the Hugging Face test2000 archive
        'snow100k_s': dict(family='snow',  input=('snow100k_hf', 'Snow100K/test2000/Snow'), gt=('snow100k_hf', 'Snow100K/test2000/Gt'), names='S'),
        'snow100k_m': dict(family='snow',  input=('snow100k_hf', 'Snow100K/test2000/Snow'), gt=('snow100k_hf', 'Snow100K/test2000/Gt'), names='M'),
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
        'mix_snow100kl_g25':  dict(family='mixed', base='snow100k_l', degrade=[('gaussian', 25)], families=('snow', 'noise')),
    }
    MAIN_BENCHMARKS = ['gopro', 'hide', 'textocr', 'kohler', 'rain100l', 'reside6k', 'snow100k_l', 'sidd']
    EXTRA_BENCHMARKS = ['rain100h', 'snow100k_s', 'snow100k_m', 'cbsd68_g15', 'cbsd68_g25', 'cbsd68_g50',
                        'live1_q10', 'lol', 'dpdd', 'mix_gopro_g25', 'mix_rain100l_haze', 'mix_snow100kl_g25']
    ABLATION_BENCHMARKS = ['gopro', 'rain100l', 'sidd']      # reference = mean over these three (Table 6)
    SYNTHETIC_SEED = 1234         # per-image seeds of synthetic noise = SYNTHETIC_SEED + crc32(file name)
    HAZE_DEPTH = (0.5, 1.0)       # pseudo-depth ramp (top -> bottom = far -> near) of the ASM haze

    # ----------------------------------------------- training (Section 4.1)
    PATCH_SIZE = 256
    BATCH_SIZE = 64               # G4 (96 GB); Section 4.1 must state batch 64 (the submitted paper said 6)
    LR = 1e-4
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 0.0
    ETA_MIN = 1e-7                # cosine annealing per iteration from LR to ETA_MIN over the whole budget
    GRAD_CLIP = 1.0
    # Budget.  One epoch = one pass over the (capped, repeated) training pool, i.e.
    # floor(N_train / BATCH_SIZE) iterations (22,871 samples -> 357 iterations at batch 64).
    # The full model gets 100 epochs (35,700 iterations, ~2.3 M patches) and every ablation run gets
    # exactly the same budget (ABLATION_EPOCHS = EPOCHS).  With equal budgets the ablation reference
    # "full, seed SEED" is the full model itself: it is trained once (run KASIRNet_full) and reused,
    # never trained a second time.  main() prints the measured s/iteration and the GPU-hours left.
    EPOCHS = 100
    ABLATION_EPOCHS = 100
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
    def full_alias(self):
        """Ablation run id that is identical to the full model (same seed, budget, model and loss), or None."""
        if self.ABLATION_EPOCHS == self.EPOCHS and self.SEED in self.ABLATION_SEEDS:
            return f'full_s{self.SEED}'
        return None

    def resolve_run(self, run_id):
        return self.FULL_RUN_ID if run_id == self.full_alias() else run_id

    def ckpt_dir(self, run_id):   return os.path.join(self.OUTPUT_ROOT, 'checkpoints', self.resolve_run(run_id))
    def run_dir(self, run_id):    return os.path.join(self.OUTPUT_ROOT, 'runs', self.resolve_run(run_id))
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
        seen.add(self.full_alias())   # trained once as the full model, never again
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
