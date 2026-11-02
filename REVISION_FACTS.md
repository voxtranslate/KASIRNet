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
* **SIDD**: no SIDD path in any submitted configuration (training or test); the SIDD figure cannot be traced to the released code.
* **Snow100K**: the training category `snow` has empty paths (no snow training data); the snow test images came from a private archive (`gdown 1tfeBnjZ…`, `/content/input`, `/content/gt`).
* The training category `rain` (`chenchienying/deraindataset`) is DeRaindrop (raindrops on glass, 861 training pairs), not rain streaks.

Revised code: pairs matched by image id, JRDR direction fixed, near-duplicates of test images removed automatically (`Datasets.LeakageAuditor`, which finds exactly the 546 RainTrainH/Rain100H duplicates), Rain100L test-image category removed, Köhler test-only, Snow100K training set and SIDD-Medium added for the snow and SIDD columns, SIDD evaluated on the standard validation crops.

## TODO-05: SRB gate
`resonance_gate` ends with a Sigmoid and is not zero-initialised: on random features its output is 0.495 ± 0.45 at initialisation, so `A' = A(1 + G)` starts around 1.5 A. The gate multiplies the amplitude only and keeps the phase exactly (measured phase change of the gated spectrum: about 1e-8 rad); the complex 1 × 1 mixing that follows does change the phase (`spectral_srb_phase_and_gates` reports both per SRB on the trained model).

## TODO-13: architecture of the code
The refactored `Model.py` with default flags has the same 438 tensors with the same shapes as the submitted model (outputs identical within 2.4e-7 with the same weights). Parameter count: trainable 37.60 M (SCWAM 12.18, SIM 9.82, SRB 5.47, Prompt Learner 0.50, SFT 0.43, GCB 0.26, other 8.94), frozen VGG-19 10.59 M, loaded at inference 48.19 M; SRB blocks 5.47 M vs 2.00 M for the plain-conv replacement (difference 3.46 M).

## TODO-15: loss weights actually used
The submitted notebook used `fft_weight = 1.0` (HYPERPARAMS) and `edge_weight = 1.0` (constructor default), not 0.5 / 0.5, and its SSIM term (pyiqa metric created without `as_loss=True`, evaluated under `torch.set_grad_enabled(False)`) contributed **no gradient**. The revised code uses the paper weights (5.0, 0.5, 1.0, 1.0, 1.0, 0.5) with a differentiable SSIM of the same definition (|difference| to the pyiqa value ≈ 1e-4, due to the uint8 rounding).
