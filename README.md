# KASIRNet

**Knowledge-Augmented Single Image Revamp Network for unified all-in-one degradation removal** (Information Sciences, manuscript INS-D-26-12219, revision).

KASIRNet is a four-scale U-Net trained once on a pool of degradation categories and evaluated with a single set of weights on deblurring, deraining, dehazing, desnowing and denoising benchmarks, without degradation labels at inference. Its components:

* **SRB** (Spectral Resonance Block): rFFT2 → channel gate computed on the Fourier amplitude, `A' = A(1+G)` (this step keeps the phase exactly) → learned complex 1×1 mixing of `[A'cosΦ; A'sinΦ]` (this step can change the phase) → irFFT2 → depthwise/pointwise spatial branch, residual.
* **SIM** (Semantic Injection Module): frozen ImageNet VGG-19 features (blocks ending at pool1–pool4: 64/128/256/512 channels at 1/2–1/16 resolution) gate the restoration features through channel and spatial attention.
* **SCWAM** (Shifted Cyclic Window Attention with Meshgrid): 8×8 windows, cyclic shift of 4 in every layer, **no** Swin attention mask (tokens on opposite borders attend to each other), **no** relative-position-bias table; absolute position comes from a CoordConv-like Meshgrid embedding added before partitioning.
* **Prompt Learner**: a 0.5 M-parameter CNN computes a 256-D *visual* descriptor `p` from the degraded image (no text, no label); `p` modulates the four decoder stages through channel-wise SFT `y = x(1+γ(p)) + β(p)`.

## Repository

| file | content |
|---|---|
| `KASIRNet.ipynb` | the notebook to run on Colab (generated from the modules below by `tools/build_notebook.py`; each code cell is identical to one module) |
| `Config.py` | every setting: data sources, training pool, benchmarks, hyper-parameters, model flags, the run plan of all ablations |
| `Model.py` | KASIRNet. With default flags it is exactly the submitted architecture (same 438 tensors, a submitted checkpoint loads with `strict=True`); flags implement the ablations |
| `Loss.py` | six-term Omni-Restoration loss |
| `metrics.py` | the single metric function (PSNR/SSIM/LPIPS/DISTS) |
| `Datasets.py` | data download, key-based pairing, train/test leakage audit, training manifest, synthetic degradations, test sets |
| `Trainer.py` | resumable FP16 trainer, curves, `run.json` |
| `Inferencer.py` | full-resolution benchmark evaluation with per-image CSV, competitor scoring, explainability figures |
| `Analysis.py` | analyses requested by the reviewers and the manuscript tables |
| `main.py` | `main()`: audit → full training → full inference → analyses → ablations → tables |
| `REVISION_FACTS.md` | facts about the submitted code and data established for the revision |

## Training configuration (Section 4.1)

| item | value |
|---|---|
| hardware | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (Colab G4 runtime); exact device/driver/CUDA/cuDNN/PyTorch versions are recorded in each `run.json` |
| patches | 256 × 256 random crops, random horizontal and vertical flips; images whose shorter side is below 256 (CelebA, 178 × 218) are first up-scaled with bicubic interpolation keeping the aspect ratio |
| batch / optimiser | 64 / Adam (β = 0.9, 0.999), no weight decay |
| schedule | reference model and ablations: learning rate 1e-4, cosine annealing per iteration to 1e-7 over the whole budget; final model: warm restart from the reference model, linear warm-up over 1,500 iterations to 3e-4 (square-root scaling of 1e-4 from batch 6 to batch 64), cosine to 1e-7; gradient-norm clipping 1.0 |
| precision | FP16 autocast + GradScaler (the FFT path of the SRB is computed in FP32) |
| loss | `5.0 L_pix + 0.5 L_fft + 1.0 L_perc + 1.0 L_ssim + 1.0 L_id + 0.5 L_edge` (definitions in `Loss.py`) |
| budget | one epoch = ⌊22,871 / 64⌋ = 357 iterations. Reference full model `KASIRNet_full` and every ablation run: 100 epochs (35,700 iterations); the ablation reference “full, seed 1” is `KASIRNet_full` itself (trained once), so 45 ablation runs are trained. Final model `KASIRNet_main` (Tables 4–5): `KASIRNet_full` + 200 epochs (71,400 iterations), 300 epochs / 107,100 updates in total. Iterations and wall-clock time are written to `run.json` and `tables/section41_runs.csv` |
| stopping / selection | fixed budget, no early stopping; the reported checkpoint is the one with the best mean PSNR on a validation split held out from the training pool (10 pairs per category, centre 256 × 256 crops, never a test image) |
| seeds | full model seed 1; ablations seeds 1, 2, 3 (TODO-12A) or 1; data subsets seeded (`DATA_SUBSET_SEED` = 2026), synthetic test degradations seeded per image |
| frozen networks | VGG-19 and ResNet-50 / ConvNeXt-T with torchvision `IMAGENET1K_V1` weights, always in eval mode; the loss networks (VGG-19 perceptual, ResNet-50 identity) exist only during training |

### Training pool (14 categories, written to `data_manifest/train_manifest.json`)

| family | categories (source) |
|---|---|
| blur | GoPro, TextOCR, CelebA, Helen training splits (DBlur collection; the collection has no HIDE training split, so HIDE is evaluated with the standard GoPro → HIDE protocol); RealBlur-J training split (real recorded camera shake, 3,758 pairs, replaces the Köhler category of the submitted pool) |
| rain | RainTrainH (heavy) and RainTrainL-1800 (light) from JRDR, **direction rainy → clean, pairs matched by image id, backgrounds duplicated in Rain100H/Rain100L test removed**; DeRaindrop training split (raindrops) |
| haze | RESIDE-6K training split |
| snow | Snow100K: 2,500 pairs of the official training set (the subset released with ConvIR, Hugging Face mirror `fancongyi/snow100k`; all names checked against the official training set, none in the test sets) |
| noise | SIDD-Medium sRGB (320 pairs, 8 random crops of each per epoch), salt-and-pepper, noisy documents, AlphaISP 2DNR |

Each category is capped at 2,500 pairs (seeded random subset). Before the cap, every training candidate is compared with the clean images of every benchmark of the same degradation family and near-duplicates are removed (`Datasets.LeakageAuditor`). A sample of every category and benchmark is then checked for readability, size agreement, alignment and direction of the degradation (`data_manifest/data_validation.json`); a category that looks mis-paired or swapped stops the run. Köhler has no training split and is a test set only (`KOHLER_PROTOCOL = 'test_only'`; `'scene_split'` trains on scenes 1–3 and tests on scene 4).

### Evaluation protocol (TODO-01)

* Metric function (`metrics.py`, pyiqa 0.1.16 defaults): **PSNR** on RGB, data range 1, no border crop; **SSIM** on the Y channel of YIQ scaled to [0, 255] and rounded, Gaussian window 11 × 11 (σ = 1.5), valid filtering, no down-sampling, no border crop; **LPIPS** AlexNet v0.1; **DISTS**.
* Batch 1, FP32 forward pass, output clamped to [0, 1] and not quantised; every image of every test set; mean of per-image values. A ground truth whose size differs from the input is resized to the input size (counted in `summary.json`).
* **Inference protocol** (`Config.INFERENCE_PROTOCOL`, weights untouched). Several operators aggregate over the whole input: the SRB resonance gate (mean of the un-normalised FFT amplitude, 2.3–3.8× larger on test images than on the 256 × 256 training patches), the anti-linear part of the SRB complex mixing (point reflection of the whole map, `spectral_srb_operator_decomposition`), SIM channel attention, Prompt-Learner pooling, GCB and Meshgrid. `tiledP` restores each image from overlapping P × P tiles (hop P/2, sin² window = exact partition of unity, reflect-padded borders) so every operator sees the training regime; images not larger than P are processed directly. `'auto'` selects among `full`, `tiled256`, `tiled384` once, on the held-out validation pairs of the training pool at full resolution (never a test set; `analysis/inference_protocol/`), and uses the choice for every run.
* **Köhler** is scored after registration to its reference (RealBlur protocol: least-squares intensity scaling + ECC homography, metrics on the valid region), for the input, KASIRNet and every competitor alike: the 12 blurred images of a scene share one sharp reference and are shifted from it by up to 10 px.
* Each stored result carries a signature of its data sources, protocol and metric version; results computed otherwise are recomputed. Every real benchmark is checked against a fingerprint (number of pairs, mean input PSNR) during the audit.
* Test sets: GoPro (1,111), HIDE (2,025), TextOCR (500, DBlur test split), Köhler (48), Rain100L (100, 480 × 320, Kaggle `adeshpal/rain-100l`, the MPRNet/Restormer test layout; input 25.5 dB), Rain100H (100), RESIDE-6K (1,000), Snow100K-L (16,801, complete official test set); severity levels Snow100K-S / -M (the 713 / 662 official S / M test images of the Hugging Face test2000 archive), SIDD validation (1,280 crops of 256 × 256); severity and zero-shot sets: CBSD68 + Gaussian σ = 15/25/50, LIVE1 JPEG q = 10 (29), LOL eval (15), DPDD test (76, combined image); mixed sets: GoPro + Gaussian 25, Rain100L + haze (β = 1.0, A = 0.8), Snow100K-L + Gaussian 25.

## Running

Open `KASIRNet.ipynb` in Colab with the G4 runtime and *Run all*. All pretrained networks (VGG-19, ResNet-50, ConvNeXt-T, LPIPS/AlexNet, DISTS/VGG-16, competitor checkpoints) are downloaded at the start of the audit and copied to `Config.WEIGHTS_CACHE` on Drive. No dataset depends on a Google-Drive download any more (those fail with “quota exceeded”): Kaggle sources via `kagglehub` (fast, mirrored by Colab; Snow100K-L from `suzukimio/allweather-testing`) and Hugging Face mirrors (Snow100K training and S/M test images from `fancongyi/snow100k`, DPDD test from `JacobLinCool/DPDD`). Every mirror was checked against the official archive listings: same file names and byte sizes. The Hugging Face downloads (about 2 GB) are packed into `Config.DATA_CACHE_ROOT` on Google Drive and restored from there in later sessions (`DATA_CACHE_SOURCES = 'all'` caches every source; the audit prints what the cache holds and the free Drive space). If a Google-Drive download fails with “quota exceeded”, put a copy of the archive in `DATA_CACHE_ROOT/archives/`. Outputs go to `Config.OUTPUT_ROOT` on Google Drive. Stages are idempotent and every run checkpoints every 2,000 iterations, so a session that ends is resumed by running the notebook again (resumption is bit-exact). Several G4 sessions can run the notebook at the same time on the same Drive folder: a run in training is locked, so the sessions split the ablation queue. Competitor outputs held by the authors are read from `Config.COMPETITOR_ROOT/<benchmark>/<method>/` (files named like the test inputs).

Outputs used by the manuscript (all under `OUTPUT_ROOT/tables`, CSV + LaTeX rows, PSNR with two decimals, SSIM with three):

| reviewer item | table / file |
|---|---|
| Tables 4–5, ablation reference (mean of GoPro, Rain100L, SIDD) | `table4_5_main_results` |
| TODO-02 Rain100H | `table4_5_main_results` (row `rain100h`) |
| TODO-06 Figs. S1/S2 | `analysis/figures/FigS1_rain100l.*`, `FigS2_snow100k_l.*` (images at the 25th/50th/75th percentile of PSNR gain) |
| TODO-09 A / B | `table8_sim_severity`, `table8_feature_stability`, `table8_extractors` |
| TODO-12 A / B / C | `table9_seed_variance`, `table9_interactions`, `table9_bootstrap_ci`, `table9_wilcoxon_vs_competitors` |
| TODO-13 | `analysis/cost/cost.json`, `table10_latency_memory` |
| TODO-14 | `table11_perceptual` (+ per-image CSVs) |
| TODO-15 | `table12_loss_sensitivity` |
| TODO-16 | `table13_prompt_probe`, `table13_mixed_degradations` |
| TODO-17 | `table14_zero_shot` |
| Tables 6, 7, SCWAM vs Swin design | `table6_component_ablation`, `table7_loss_ablation`, `table_scwam_design` |
| phase / amplitude (R3-2, R5-2) | `spectral_amplitude_phase_attribution`, `spectral_srb_phase_and_gates` |
| Section 4.1 | `section41_runs`, `section41_reproducibility.json` |
| inference protocol | `analysis/inference_protocol/selection.json`, `validation_per_category.csv` |
| SRB train/test consistency | `spectral_srb_operator_decomposition`, `spectral_srb_train_test_gap` |

## Abstract
All-in-one image restoration, where a single unified model addresses multiple degradation types simultaneously, remains one of the most challenging open problems in low-level computer vision. Existing methods either specialize in a single degradation (e.g., deblurring, deraining, dehazing) and suffer catastrophic failure on unseen types, or attempt multi-task learning but lack the representational flexibility to handle qualitatively different corruptions in a shared parameter space. We present KASIRNet, which introduces four tightly integrated technical contributions to address these shortcomings. First, the Spectral Resonance Block (SRB) performs amplitude modulation in the Fourier domain via a learned squeeze-and-excitation resonance gate, enabling degradation-adaptive frequency recovery. Second, the Semantic Injection Module (SIM) extracts frozen VGG19 features at four scales and fuses them into the restoration backbone through parallel channel and spatial attention. Third, the Shifted Cyclic Window Attention with Meshgrid (SCWAM) augments shifted-window self-attention with explicit 2-D coordinate embeddings for geometric-aware global context. Fourth, a Prompt Learner encodes the degradation into a compact visual embedding that conditions every decoder stage via Spatial Feature Transform (SFT) affine modulation. These components are unified within a four-scale U-Net encoder–decoder trained end-to-end with a six-term Omni-Restoration Loss comprising pixel, FFT, perceptual, SSIM, identity, and edge objectives.

Keywords: All-in-one image restoration, Degradation-adaptive network, Frequency-domain processing, Multi-degradation restoration, Prompt learning, Semantic injection, Transformer attention
