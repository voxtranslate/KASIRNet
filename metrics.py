# ==========================================================================
# KASIRNet - THE metric function (TODO-01)
# --------------------------------------------------------------------------
# One implementation, used everywhere (validation, every benchmark, every
# ablation, every competitor output).  It is the pyiqa 0.1.16 implementation
# that produced the figures of the submitted manuscript, called with its
# default options:
#
#   PSNR  pyiqa 'psnr' : RGB, all three channels, data range 1, MSE over C x H x W,
#                        10 log10(1 / (MSE + 1e-8)), no border crop.
#   SSIM  pyiqa 'ssim' : luminance Y of YIQ (0.299 R + 0.587 G + 0.114 B) scaled to
#                        [0, 255] and rounded (MATLAB uint8 convention), float64,
#                        Gaussian window 11 x 11, sigma 1.5, 'valid' filtering,
#                        no down-sampling, C1 = (0.01*255)^2, C2 = (0.03*255)^2,
#                        no border crop.
#   LPIPS pyiqa 'lpips': AlexNet, v0.1.         DISTS pyiqa 'dists'.
#
# Inputs: restored output clamped to [0, 1] and NOT quantised to 8 bit; ground
# truth read from the 8-bit file; full resolution (no patches, no resizing,
# except that a ground truth whose size differs from the input is resized to
# the input size, the rule of the submitted code; such images are counted).
# ==========================================================================
import torch
import torch.nn.functional as F

PROTOCOL = {
    'library': 'pyiqa==0.1.16 (create_metric defaults)',
    'psnr': 'RGB, data range 1, no border crop, eps 1e-8, full resolution, float output clamped to [0,1] (not quantised)',
    'ssim': 'Y of YIQ scaled to [0,255] and rounded, Gaussian 11x11 sigma 1.5, valid filtering, no downsampling, no border crop',
    'lpips': 'AlexNet v0.1 (pyiqa lpips)',
    'dists': 'pyiqa dists',
    'psnr_y / ssim_y': 'additional columns: same pyiqa functions on the Y channel of YCbCr (BT.601, MATLAB rgb2ycbcr), '
                       'the convention of most deraining papers; reported only for like-with-like comparisons',
    'resolution': 'full image, batch 1, no tiling',
    'averaging': 'arithmetic mean of per-image values over all test images',
}


CUSTOM_METRICS = {'psnr_y': ('psnr', dict(test_y_channel=True, color_space='ycbcr')),
                  'ssim_y': ('ssim', dict(test_y_channel=True, color_space='ycbcr'))}


class MetricSuite:
    """metrics(restored, gt) -> {'psnr': [...], 'ssim': [...], ...} (one value per image)."""

    def __init__(self, device, names=('psnr', 'ssim')):
        import pyiqa
        self.names = tuple(names)
        self.device = device
        self.m = {}
        for n in self.names:
            base, opts = CUSTOM_METRICS.get(n, (n, {}))
            self.m[n] = pyiqa.create_metric(base, device=device, **opts)

    @torch.no_grad()
    def __call__(self, restored, gt):
        restored = restored.to(self.device, torch.float32).clamp(0, 1)
        gt = gt.to(self.device, torch.float32).clamp(0, 1)
        if restored.dim() == 3:
            restored, gt = restored.unsqueeze(0), gt.unsqueeze(0)
        out = {}
        for n, fn in self.m.items():
            v = fn(restored, gt)
            out[n] = [float(t) for t in v.reshape(-1).cpu()]
        return out


def psnr_torch(x, y):
    """Same formula as pyiqa 'psnr' (used only for the cheap training-batch log)."""
    mse = ((x.float() - y.float()) ** 2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / (mse + 1e-8))


# ------------------------------------------------------------------
# Differentiable SSIM for the loss: identical definition (Y of YIQ, Gaussian
# 11x11, sigma 1.5, valid filtering) but float32 and without the uint8 rounding.
# ------------------------------------------------------------------
_YIQ = torch.tensor([0.299, 0.587, 0.114])


def _gauss_win(size=11, sigma=1.5, device='cpu', dtype=torch.float32):
    ax = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, size, size)


def ssim_y(x, y, size=11, sigma=1.5):
    x = x.float()
    y = y.float()
    w = _YIQ.to(x.device).view(1, 3, 1, 1)
    x = (x * w).sum(1, keepdim=True)
    y = (y * w).sum(1, keepdim=True)
    win = _gauss_win(size, sigma, x.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = F.conv2d(x, win), F.conv2d(y, win)
    s11 = F.conv2d(x * x, win) - mu1 ** 2
    s22 = F.conv2d(y * y, win) - mu2 ** 2
    s12 = F.conv2d(x * y, win) - mu1 * mu2
    cs = F.relu((2 * s12 + C2) / (s11 + s22 + C2))
    ssim_map = ((2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)) * cs
    return ssim_map.mean(dim=[1, 2, 3])
