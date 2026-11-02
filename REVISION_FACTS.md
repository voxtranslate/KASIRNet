# Facts established for the revision (Step 0)

Sources: the submitted notebook and files of commit `1d4282b` (`KASIRNet.ipynb`, `Config.py`, `Loss.py`, `Model.py`), the pyiqa 0.1.16 source code, and the Kaggle datasets that the submitted configuration downloads (checked file by file on 2 Nov 2026). The notebook's `audit` stage regenerates the checkable items in `analysis/step0_facts/facts.json`, including the items that need the trained checkpoint on Google Drive.

## TODO-01: how the reported metrics were computed
* Library: pyiqa 0.1.16 (`pip install pyiqa --no-deps` in the notebook), `create_metric('psnr' | 'ssim' | 'lpips')` with default options.
* PSNR: RGB, all three channels, data range 1, `10 log10(1/(MSE + 1e-8))`, no border crop.
* SSIM: Y channel of YIQ (0.299 R + 0.587 G + 0.114 B) scaled to [0, 255] and rounded, float64, Gaussian window 11 × 11 with σ = 1.5, valid filtering, no down-sampling, no border crop, C1 = (0.01·255)², C2 = (0.03·255)².
* LPIPS: AlexNet, v0.1.
* Inference (`KASIRNetInferencer.run_dataset_inference`): full-resolution forward pass, output clamped to [0, 1] and not quantised, ground truth resized to the input size when the sizes differ. **At most the first 256 sorted images of each test folder were evaluated** (`max_images=256`).
* Checkpoint selection during training (`OMSTrainer._validate_epoch`): mean PSNR on centre 256 × 256 crops of the GoPro test set, the DeRaindrop `test_b` set and the RESIDE-6K test set, i.e. on test images.

The revised code keeps the same metric function (`metrics.py`), evaluates every test image and selects the checkpoint on a split held out from the training pool.

## TODO-03: frozen networks in the released code
* `VGG19SemanticExtractor`, the perceptual-loss VGG-19 and the identity-loss ResNet-50 were all built with `weights=None`: **no ImageNet weights were loaded** by the released code. The identity-loss ResNet-50 also stayed in train mode (BatchNorm with batch statistics).
* `Loss.py` contained `ELRICLoss` from another project; the loss actually used is `OmniRestorationLoss` in the notebook.
* `Config.py` had batch size 64 and 3,000 epochs; the notebook trained in FP32 (no autocast), with `CosineAnnealingLR(T_max=3000)` stepped once per epoch.

Revised code: ImageNet `IMAGENET1K_V1` weights everywhere, frozen networks always in eval mode, paper hyper-parameters (batch 6, FP16, per-iteration cosine). The `audit` stage measures the distance between the VGG-19 weights stored in the submitted `best.pth` and the ImageNet weights (`vgg_is_imagenet`).

## TODO-04: overlap between training and test data in the submitted pool
* **Köhler**: training category `blurred` = `kohlerdeblur/kohler/blur|sharp`, i.e. all 48 Köhler pairs; the test entry of `Config.py` points to the same folders (the notebook test entry is empty). A Köhler figure obtained with this configuration is measured on training images.
* **Rain100L**: training category `raining` (`adeshpal/image-deraining`, `Dataset/input` = `rain-001 … rain-080`) contains **80 of the 100 Rain100L test images** (80/80 ground-truth images are near-duplicates, correlation of 24 × 24 thumbnails > 0.97, of the Rain100L test ground truths; checked against two independent copies of Rain100L).
* **Rain100H**: training category `rainy` (JRDR `rain_data_train_Heavy` = RainTrainH) has input and ground truth swapped (`input = norain/`, mean gradient energy 0.067, `gt = rain/X2/`, 0.138) and the pairs were formed by zipping two independently sorted lists, which mismatches the scene id in **1,800 of 1,800 pairs** (`norain-10.png` ↔ `norain-1001x2.png`). In addition, 546 RainTrainH backgrounds duplicate Rain100H test images (all 100 test images are covered), the overlap documented by the PReNet authors; the light counterpart (`rain_data_train_Light`) has the same 546-image overlap with Rain100L.
* **HIDE**: the training category `hide` points to `DBlur/HIDE/train`, which does not exist in the DBlur collection (it contains HIDE `test` only, 2,025 pairs; full archive listing checked). The loader silently skipped empty folders, so **no HIDE image was trained on**, contrary to the manuscript statement that the HIDE training split is in the pool. Together with the empty `snow` category, the submitted pool had 12 non-empty categories, not 13.
* **SIDD**: no SIDD path in any submitted configuration (training or test); the SIDD figure cannot be traced to the released code.
* **Snow100K**: the training category `snow` has empty paths (no snow training data); the snow test images came from a private archive (`gdown 1tfeBnjZ…`, `/content/input`, `/content/gt`).
* The training category `rain` (`chenchienying/deraindataset`) is DeRaindrop (raindrops on glass, 861 training pairs), not rain streaks.
* **RESIDE-6K**: 7 of the 6,000 training ground truths are the same scenes as test ground truths (e.g. `train/GT/1439.jpg` = `test/GT/1933_0.8_0.2.jpg`, checked visually); 6 of them were in the submitted training subset (first 2,500 sorted names). Small (0.6 % of the test set) but removed by the revised audit.
* CelebA images of the DBlur collection are 178 × 218; the submitted loader stretched every image smaller than 256 to a 256 × 256 square (aspect ratio and blur kernels distorted). The revised loader up-scales with bicubic interpolation keeping the aspect ratio (shorter side 256).

Revised code: pairs matched by image id, JRDR direction fixed, near-duplicates of test images removed automatically against every benchmark of the same family (`Datasets.LeakageAuditor`, which finds exactly the 546 RainTrainH/Rain100H duplicates), Rain100L test-image category removed, Köhler test-only, HIDE test-only (standard GoPro → HIDE protocol), Snow100K training set and SIDD-Medium added for the snow and SIDD columns, SIDD evaluated on the standard validation crops; missing or empty folders now stop the run instead of being skipped; every category and benchmark is validated for pairing, alignment and direction before training. Revised pool: 13 categories (GoPro, TextOCR, CelebA, Helen, RainTrainH, RainTrainL, DeRaindrop, RESIDE-6K, Snow100K, SIDD-Medium, salt-and-pepper, noisy documents, AlphaISP).

## TODO-05: SRB gate
`resonance_gate` ends with a Sigmoid and is not zero-initialised: on random features its output is 0.495 ± 0.45 at initialisation, so `A' = A(1 + G)` starts around 1.5 A. The gate multiplies the amplitude only and keeps the phase exactly (measured phase change of the gated spectrum: about 1e-8 rad); the complex 1 × 1 mixing that follows does change the phase (`spectral_srb_phase_and_gates` reports both per SRB on the trained model).

## TODO-13: architecture of the code
The refactored `Model.py` with default flags has the same 438 tensors with the same shapes as the submitted model (outputs identical within 2.4e-7 with the same weights). Parameter count: trainable 37.60 M (SCWAM 12.18, SIM 9.82, SRB 5.47, Prompt Learner 0.50, SFT 0.43, GCB 0.26, other 8.94), frozen VGG-19 10.59 M, loaded at inference 48.19 M; SRB blocks 5.47 M vs 2.00 M for the plain-conv replacement (difference 3.46 M). MACs (`torch.utils.flop_counter`, FFTs not counted): 99.29 G at 256 × 256 (76.22 restoration path + 23.06 VGG-19) and 24.85 G at 128 × 128, as in Table 10.
Networks instantiated at training: 48.19 M (model) + 10.59 M (separate VGG-19 of the perceptual loss) + 23.51 M (ResNet-50 of the identity loss) = **82.28 M**. The 71.70 M of Table 10 leaves out the perceptual-loss VGG-19, which is a second instance in both the submitted and the revised code.

## TODO-15: loss weights actually used
The submitted notebook used `fft_weight = 1.0` (HYPERPARAMS) and `edge_weight = 1.0` (constructor default), not 0.5 / 0.5, and its SSIM term (pyiqa metric created without `as_loss=True`, evaluated under `torch.set_grad_enabled(False)`) contributed **no gradient**. The revised code uses the paper weights (5.0, 0.5, 1.0, 1.0, 1.0, 0.5) with a differentiable SSIM of the same definition (|difference| to the pyiqa value ≈ 1e-4, due to the uint8 rounding).

## Verification of the revised data pipeline on the real datasets (2–3 Nov 2026)

The audit stage of the revised notebook (`Datasets.build_training_manifest` + `Datasets.validate_data`) was run on the real data: every Kaggle source downloaded, the DBlur collection read member by member from its archive (GoPro train/test, HIDE test, TextOCR, Helen, CelebA test and 4,000 CelebA training pairs), SIDD validation crops and 6 SIDD-Medium scenes, the DPDD test split from the Hugging Face mirror. Snow100K (2 × 7.8 GB) was verified from the complete archive listings.

| check | result |
|---|---|
| archive layouts | DBlur: GoPro train 2,103 / test 1,111; HIDE test 2,025 (no train); TextOCR train 5,000 / val 500 / test 500; CelebA train 100,000 / test 1,300; Helen train 2,000 / test 165. Snow100K: train `all/{gt,mask,synthetic}` 50,000; test S/M/L 16,611 / 16,588 / 16,801 in `media/jdway/GameSSD/overlapping/test/Snow100K-*/`; every synthetic image has a ground truth with the same name; no name shared between train, S, M, L. SIDD: 160 scenes × 2 pairs (SIDD-Medium sRGB) + validation lmdb with 1,280 aligned 256 × 256 crops (`ValidationBlocksSrgb_*`). DPDD test: 76 pairs, 1680 × 1120. |
| pairing | 100 % of the input and ground-truth files of every training category and benchmark are paired (key-based) |
| leakage audit (every training candidate vs every benchmark of the same family) | RainTrainH: 546 duplicates of Rain100H and of Rain100L test backgrounds (the two test sets share their 100 clean images); RainTrainL: the same 546; RESIDE-6K: 4 duplicates among the 3,044 candidates checked (7 in the whole training set); GoPro / TextOCR / CelebA / Helen vs GoPro, HIDE, TextOCR, Köhler: none; DeRaindrop vs Rain100L/H: none; noise categories vs CBSD68: none. All duplicates are removed. |
| pair validation (60 pairs per set: readability, sizes, 64 × 64 correlation, direction of the degradation) | no flag on any of the 12 local training categories or 10 benchmarks; median correlation 0.88–1.00. On the submitted RainTrainH pairing the same validator reports MISPAIR (median correlation 0.11), SWAP and SIZE. |
| image sizes | no training image below 256 px except CelebA (178 × 218); salt-and-pepper and document images are greyscale (converted to 3 identical channels) |
| revised training pool | GoPro 2,093 + 10 val; TextOCR 2,500 + 10; CelebA 2,500 + 10; Helen 1,990 + 10; RainTrainH 1,244 + 10; RainTrainL 1,244 + 10; DeRaindrop 851 + 10; RESIDE-6K 2,500 + 10; Snow100K 2,500 + 10; SIDD-Medium 310 + 10 (× 8 crops per epoch); salt-and-pepper 290 + 10; documents 134 + 10; AlphaISP 45 + 5 |
