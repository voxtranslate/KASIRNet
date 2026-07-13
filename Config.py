from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

def setup_pyiqa(device):
    import pyiqa
    return {
        'psnr': pyiqa.create_metric('psnr', device=device),
        'ssim': pyiqa.create_metric('ssim', device=device),
        'lpips': pyiqa.create_metric('lpips', device=device)
    }



CHECKPOINT_DIR = '/content/drive/MyDrive/Checkpoints/KASIRNet'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best.pth')
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_oms_net.pth')

DATA_CONFIG = {
    'noise_sp': {
        'input': '/kaggle/input/salt-and-pepper-noise-images/Noisy_folder',
        'gt':    '/kaggle/input/salt-and-pepper-noise-images/Ground_truth',
    },
    'blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/TextOCR/train/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/TextOCR/train/sharp',
    },
    'rain': {
        'input': '/kaggle/input/deraindataset/train/data',
        'gt':    '/kaggle/input/deraindataset/train/gt',
    },
    'rainy': {
        'input': '/kaggle/input/jrdr-deraining-dataset/JRDR/rain_data_train_Heavy/norain',
        'gt':    '/kaggle/input/jrdr-deraining-dataset/JRDR/rain_data_train_Heavy/rain/X2',
    },
    'blurred': {
        'input': '/kaggle/input/kohlerdeblur/kohler/blur',
        'gt':    '/kaggle/input/kohlerdeblur/kohler/sharp',
    },
    'raining': {
        'input': '/kaggle/input/image-deraining/Dataset/input',
        'gt':    '/kaggle/input/image-deraining/Dataset/target',
    },
    'haze': {
        'input': '/kaggle/input/reside-6k/RESIDE-6K/train/hazy',
        'gt':    '/kaggle/input/reside-6k/RESIDE-6K/train/GT',
    },
    'gopro': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/train/sharp',
    },
    'hide': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/HIDE/train/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/HIDE/train/sharp',
    },
    'celeb': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/CelebA/train/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/CelebA/train/sharp',
    },
    'helen': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Helen/train/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Helen/train/sharp',
    },
    'snow': {
        'input': '',
        'gt':    '',
    },
    'noise': {
        'input': '/kaggle/input/image-denoising/train',
        'gt':    '/kaggle/input/image-denoising/train_cleaned',
    },
    'noisy': {
        'input': '/kaggle/input/denoising-dataset-multiple-iso-levels/AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Noised',
        'gt':    '/kaggle/input/denoising-dataset-multiple-iso-levels/AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Ground Truth',
    },
}

VAL_DATA_CONFIG = {
    'blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/sharp',
    },
    'rain': {
        'input': '/kaggle/input/deraindataset/test_b/data',
        'gt':    '/kaggle/input/deraindataset/test_b/gt',
    },
    'haze': {
        'input': '/kaggle/input/reside-6k/RESIDE-6K/test/hazy',
        'gt':    '/kaggle/input/reside-6k/RESIDE-6K/test/GT',
    },
}

TEST_DATA_CONFIG = {
    'kohler_blur': {
        'input': '/kaggle/input/kohlerdeblur/kohler/blur',
        'gt':    '/kaggle/input/kohlerdeblur/kohler/sharp',
    },
    'gopro_blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Gopro/test/sharp',
    },
    'helen_blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Helen/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/Helen/test/sharp',
    },
    'hide_blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/HIDE/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/HIDE/test/sharp',
    },
    'text_ocr_blur': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/TextOCR/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/TextOCR/test/sharp',
    },
    'celeba': {
        'input': '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/CelebA/test/blur',
        'gt':    '/kaggle/input/a-curated-list-of-image-deblurring-datasets/DBlur/CelebA/test/sharp',
    },
    'rain_test_a': {
        'input': '/kaggle/input/deraindataset/test_a/data',
        'gt':    '/kaggle/input/deraindataset/test_a/gt',
    },
    'rain_test_b': {
        'input': '/kaggle/input/deraindataset/test_b/data',
        'gt':    '/kaggle/input/deraindataset/test_b/gt',
    },
    'reside_haze': {
        'input': '/kaggle/input/reside-6k/RESIDE-6K/test/hazy',
        'gt':    '/kaggle/input/reside-6k/RESIDE-6K/test/GT',
    },
    'snow': {
        'input': '/content/input',
        'gt':    '/content/gt',
    },
    'noise': {
        'input': '/kaggle/input/image-denoising/train',
        'gt':    '/kaggle/input/image-denoising/train_cleaned',
    },
    'noisy': {
        'input': '/kaggle/input/denoising-dataset-multiple-iso-levels/AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Noised',
        'gt':    '/kaggle/input/denoising-dataset-multiple-iso-levels/AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset/PNG Data/2DNR Denoising/Ground Truth',
    },
    'noise_sp': {
        'input': '/kaggle/input/salt-and-pepper-noise-images/Noisy_folder',
        'gt':    '/kaggle/input/salt-and-pepper-noise-images/Ground_truth',
    },
}

HYPERPARAMS = {
    'batch_size': 64,
    'patch_size': 256,
    'lr': 1e-4,
    'epochs': 3000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fft_weight': 1.0,
    'perceptual_weight': 1.0,
    'identity_weight': 1.0,
}