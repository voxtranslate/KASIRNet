"""Competitor outputs generated from public code + official pretrained weights.

The analysis code only *scores* competitor restorations stored in
    COMPETITOR_ROOT/<benchmark>/<method>/<input stem>.png
This module fills those folders automatically: it downloads the architectures (pip
package `deepinv` for Restormer / PromptIR / MPRNet; NAFNet and Histoformer are
vendored below with their licence headers) and the authors' pretrained weights from
GitHub releases or the Hugging Face Hub (never Google Drive), runs every test image at
full resolution with the method's own inference convention (RGB in [0, 1], FP32,
batch 1, the method's padding rule, output clamped to [0, 1]) and writes 8-bit PNGs.

    for method, bench in competitor_plan(cfg):
        generate_competitor_outputs(cfg, roots, method, bench)

Weights are verified against their SHA-256 when downloaded and cached in the torch hub
cache (torch.hub.get_dir()/checkpoints, i.e. $TORCH_HOME/hub), which main.py syncs to
Google Drive.  Output folders are resumable: existing PNGs are skipped (written
atomically), and a pair whose done.json reports the same number of images is skipped.
"""
import os
import json
import time
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from Datasets import BenchmarkDataset  # nb:skip


# ==========================================
# 1. Weights: official files, pinned by SHA-256 (first URL = primary, others = mirrors)
# ==========================================

_RESTORMER_REL = 'https://github.com/swz30/Restormer/releases/download/v1.0/'   # official GitHub release v1.0
_RESTORMER_HF = 'https://huggingface.co/deepinv/Restormer/resolve/main/'        # byte-identical mirror (deepinv)

COMPETITOR_WEIGHTS = {
    # Restormer (Zamir et al., CVPR 2022), MIT licence
    'restormer_motion': dict(file='restormer_motion_deblurring.pth',
                             urls=[_RESTORMER_REL + 'motion_deblurring.pth'],
                             sha256='194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31',
                             note='GoPro-trained motion deblurring'),
    'restormer_derain': dict(file='restormer_deraining.pth',
                             urls=[_RESTORMER_REL + 'deraining.pth', _RESTORMER_HF + 'deraining.pth'],
                             sha256='225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681',
                             note='Rain13K-trained deraining'),
    'restormer_real_noise': dict(file='restormer_real_denoising.pth',
                                 urls=[_RESTORMER_REL + 'real_denoising.pth', _RESTORMER_HF + 'real_denoising.pth'],
                                 sha256='4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387',
                                 note='SIDD-trained real denoising'),
    'restormer_gauss_color': dict(file='restormer_gaussian_color_denoising_blind.pth',
                                  urls=[_RESTORMER_REL + 'gaussian_color_denoising_blind.pth',
                                        _RESTORMER_HF + 'gaussian_color_denoising_blind.pth'],
                                  sha256='4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd',
                                  note='blind colour Gaussian denoising (sigma in [0, 50])'),
    'restormer_defocus': dict(file='restormer_single_image_defocus_deblurring.pth',
                              urls=[_RESTORMER_REL + 'single_image_defocus_deblurring.pth',
                                    _RESTORMER_HF + 'single_image_defocus_deblurring.pth'],
                              sha256='7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e',
                              note='DPDD single-image defocus deblurring'),
    # PromptIR (Potlapalli et al., NeurIPS 2023), all-in-one noise / rain / haze model, MIT licence
    'promptir_3task': dict(file='promptir_promptir.ckpt',
                           urls=['https://huggingface.co/deepinv/promptir/resolve/main/promptir.ckpt'],
                           sha256='d80e000e3c1b3f0b9d16e34530ff00d573eeb6e38de787195cd60f2d884e5075',
                           note='official all-in-one (noise+rain+haze) checkpoint, mirrored by deepinv'),
    # MPRNet (Zamir et al., CVPR 2021), Academic Public Licence; deepinv mirror of the authors' Google-Drive files
    'mprnet_deblur': dict(file='mprnet_model_deblurring.pth',
                          urls=['https://huggingface.co/deepinv/MPRNet/resolve/main/model_deblurring.pth'],
                          sha256='70a45be8899bc168bea8dae14be56f90b2343d7f674ee8c52d53ac51d6cb144b',
                          note='GoPro-trained deblurring (n_feat=96)'),
    'mprnet_derain': dict(file='mprnet_model_deraining.pth',
                          urls=['https://huggingface.co/deepinv/MPRNet/resolve/main/model_deraining.pth'],
                          sha256='d73d057a097e1cc97fbbea8481e98873f61ba4ee416834d9ef8201908e46dee5',
                          note='Rain13K-trained deraining (n_feat=40)'),
    'mprnet_denoise': dict(file='mprnet_model_denoising.pth',
                           urls=['https://huggingface.co/deepinv/MPRNet/resolve/main/model_denoising.pth'],
                           sha256='8b438e140767b951694ff2dcd9a95a77f8dde5bc41657c4315093483c3e8403e',
                           note='SIDD-trained real denoising (n_feat=80)'),
    # NAFNet (Chen et al., ECCV 2022), MIT licence.  SIDD file: the authors' own HF Space; GoPro file: HF mirror
    # of the authors' release (same set of files, the SIDD one byte-identical to the authors' copy)
    'nafnet_gopro': dict(file='nafnet_NAFNet-GoPro-width64.pth',
                         urls=['https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth',
                               'https://huggingface.co/mikestealth/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth'],
                         sha256='329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f',
                         note='GoPro-trained NAFNet-width64, run with TLC (NAFNetLocal) as in the official test'),
    'nafnet_sidd': dict(file='nafnet_NAFNet-SIDD-width64.pth',
                        urls=['https://huggingface.co/spaces/chuxiaojie/NAFNet/resolve/main/NAFNet-SIDD-width64.pth',
                              'https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width64.pth'],
                        sha256='cd685efaae01f7c4e9951f2deab05780079c8eb1e49ed664b72f6db04dabb445',
                        note='SIDD-trained NAFNet-width64'),
    # Histoformer (Sun et al., ECCV 2024), MIT licence.  net_g_best.pth = the All-weather model (Snow100K,
    # Outdoor-Rain, RainDrop); the authors' HF repo only holds net_g_real.pth (RealSnow), so the All-weather
    # file comes from a mirror whose net_g_real.pth is byte-identical to the authors' one
    'histoformer_allweather': dict(file='histoformer_allweather_net_g_best.pth',
                                   urls=['https://huggingface.co/dronefreak/Histoformer/resolve/main/net_g_best.pth'],
                                   sha256='5c7423bbc18edd3adde14f0bb48b5968da95a134cb7ab4bdb06dd7d544fea480',
                                   note='All-weather (Snow100K / Outdoor-Rain / RainDrop) model'),
}


def competitor_checkpoint(key, verbose=True):
    """Local path of a pinned checkpoint, downloaded once into the torch hub cache."""
    w = COMPETITOR_WEIGHTS[key]
    d = os.path.join(torch.hub.get_dir(), 'checkpoints')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, w['file'])
    if os.path.isfile(path):
        return path
    errors = []
    for url in w['urls']:
        try:
            if verbose:
                print(f'[competitors] downloading {url}')
            torch.hub.download_url_to_file(url, path, hash_prefix=w['sha256'], progress=verbose)
            return path
        except Exception as e:  # try the next mirror
            errors.append(f'{url}: {e}')
            if os.path.exists(path):
                os.remove(path)
    raise RuntimeError(f'could not download {key}: ' + ' | '.join(errors))


def _cmp_state_dict(key):
    path = competitor_checkpoint(key)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:  # Lightning checkpoints hold non-tensor objects; the file is SHA-256 pinned
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for k in ('params', 'params_ema', 'state_dict'):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            ckpt = ckpt[k]
            break
    out = {}
    for k, v in ckpt.items():
        for p in ('module.', 'net.'):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _cmp_load(model, key):
    model.load_state_dict(_cmp_state_dict(key), strict=True)
    return model.eval()


# ==========================================
# 2. Vendored architectures (prefixed names: the notebook shares one namespace with Model.py)
# ==========================================

# ------------------------------------------------------------------------
# NAFNet: Copyright (c) 2022 megvii-model. All Rights Reserved.  MIT License.
# Verbatim from https://github.com/megvii-research/NAFNet  basicsr/models/archs/{NAFNet_arch,local_arch,arch_util}.py
# (only renamed; training-only backward of LayerNorm2d dropped).  TLC: Chu et al., "Revisiting Global Statistics
# Aggregation for Improving Image Restoration", ECCV 2022.
# ------------------------------------------------------------------------

class NAF_LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)


class NAF_SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAF_Block(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = NAF_SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = NAF_LayerNorm2d(c)
        self.norm2 = NAF_LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAF_Net(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAF_Block(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAF_Block(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAF_Block(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAF_AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        return out


def _naf_replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            _naf_replace_layers(m, base_size, train_size, fast_imp, **kwargs)
        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = NAF_AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class NAF_NetLocal(NAF_Net):
    """NAFNetLocal (Local_Base.convert): global average pooling replaced by local pooling of the training size."""

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        NAF_Net.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            _naf_replace_layers(self, base_size, train_size=train_size, fast_imp=fast_imp)
            self.forward(torch.rand(train_size))


# ------------------------------------------------------------------------
# Histoformer: Sun et al., "Restoring Images in Adverse Weather Conditions via Histogram Transformer", ECCV 2024.
# MIT License.  Verbatim from https://huggingface.co/sunsean/Histoformer  basicsr/models/archs/histoformer_arch.py
# (= github.com/sunshangquan/Histoformer); only renamed, PyTorchModelHubMixin / pdb imports removed.
# ------------------------------------------------------------------------

def _histo_to_3d(x):
    from einops import rearrange
    return rearrange(x, 'b c h w -> b (h w) c')


def _histo_to_4d(x, h, w):
    from einops import rearrange
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Histo_BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)


class Histo_WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class Histo_LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = Histo_BiasFree_LayerNorm(dim)
        else:
            self.body = Histo_WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _histo_to_4d(self.body(_histo_to_3d(x)), h, w)


class Histo_FeedForward(nn.Module):
    """Dual-scale Gated Feed-Forward Network (DGFF)."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
                                  groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
                                           groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


class Histo_Attention_histogram(nn.Module):
    """Dynamic-range Histogram Self-Attention (DHSA)."""

    def __init__(self, dim, num_heads, bias, ifBox=True):
        super().__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

    def reshape_attn(self, q, k, v, ifBox):
        from einops import rearrange
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x
        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)
        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


class Histo_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.attn_g = Histo_Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = Histo_LayerNorm(dim, LayerNorm_type)
        self.ffn = Histo_FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = Histo_LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))
        return x_out


class Histo_OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Histo_SkipPatchEmbed(nn.Module):
    def __init__(self, in_c=3, dim=48, bias=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None),
            nn.Conv2d(in_c, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))

    def forward(self, x):
        return self.proj(x)


class Histo_Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Histo_Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Histoformer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=36, num_blocks=[4, 4, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.667, bias=False, LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        super().__init__()
        TB = Histo_TransformerBlock
        kw = dict(ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.patch_embed = Histo_OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TB(dim=dim, num_heads=heads[0], **kw) for i in range(num_blocks[0])])
        self.down1_2 = Histo_Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TB(dim=int(dim * 2 ** 1), num_heads=heads[1], **kw) for i in range(num_blocks[1])])
        self.down2_3 = Histo_Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[TB(dim=int(dim * 2 ** 2), num_heads=heads[2], **kw) for i in range(num_blocks[2])])
        self.down3_4 = Histo_Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[TB(dim=int(dim * 2 ** 3), num_heads=heads[3], **kw) for i in range(num_blocks[3])])
        self.up4_3 = Histo_Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TB(dim=int(dim * 2 ** 2), num_heads=heads[2], **kw) for i in range(num_blocks[2])])
        self.up3_2 = Histo_Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TB(dim=int(dim * 2 ** 1), num_heads=heads[1], **kw) for i in range(num_blocks[1])])
        self.up2_1 = Histo_Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[TB(dim=int(dim * 2 ** 1), num_heads=heads[0], **kw) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TB(dim=int(dim * 2 ** 1), num_heads=heads[0], **kw) for i in range(num_refinement_blocks)])
        self.skip_patch_embed1 = Histo_SkipPatchEmbed(3, 3)
        self.skip_patch_embed2 = Histo_SkipPatchEmbed(3, 3)
        self.skip_patch_embed3 = Histo_SkipPatchEmbed(3, 3)
        self.reduce_chan_level_1 = nn.Conv2d(int(dim * 2 ** 1) + 3, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level_2 = nn.Conv2d(int(dim * 2 ** 2) + 3, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level_3 = nn.Conv2d(int(dim * 2 ** 3) + 3, int(dim * 2 ** 3), kernel_size=1, bias=bias)
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        skip_enc_level1 = self.skip_patch_embed1(inp_img)
        inp_enc_level2 = self.reduce_chan_level_1(torch.cat([inp_enc_level2, skip_enc_level1], 1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        skip_enc_level2 = self.skip_patch_embed2(skip_enc_level1)
        inp_enc_level3 = self.reduce_chan_level_2(torch.cat([inp_enc_level3, skip_enc_level2], 1))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        skip_enc_level3 = self.skip_patch_embed3(skip_enc_level2)
        inp_enc_level4 = self.reduce_chan_level_3(torch.cat([inp_enc_level4, skip_enc_level3], 1))
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1 + inp_img


# ==========================================
# 3. Builders: build(device, bench) -> eval-mode nn.Module with the benchmark's official weights
# ==========================================

_BLUR = ['gopro', 'hide', 'textocr', 'kohler']
_RAIN = ['rain100l', 'rain100h']
_SNOW = ['snow100k_l', 'snow100k_s', 'snow100k_m']
_GAUSS = ['cbsd68_g15', 'cbsd68_g25', 'cbsd68_g50']


def _cmp_task(bench):
    """Benchmark key -> degradation task used to pick a method's checkpoint."""
    if bench in _BLUR or bench == 'mix_gopro_g25':
        return 'blur'
    if bench in _RAIN:
        return 'rain'
    if bench == 'reside6k':
        return 'haze'
    if bench in _SNOW:
        return 'snow'
    if bench == 'sidd':
        return 'real_noise'
    if bench in _GAUSS:
        return 'gauss_noise'
    if bench == 'dpdd':
        return 'defocus'
    if bench == 'mix_rain100l_haze':
        return 'rain+haze'
    raise KeyError(f'no competitor task for benchmark {bench!r}')


_RESTORMER_TASKS = {'blur': ('restormer_motion', 'WithBias'), 'rain': ('restormer_derain', 'WithBias'),
                    'real_noise': ('restormer_real_noise', 'BiasFree'), 'gauss_noise': ('restormer_gauss_color', 'BiasFree'),
                    'defocus': ('restormer_defocus', 'WithBias')}
_MPRNET_TASKS = {'blur': ('mprnet_deblur', dict(n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32)),
                 'rain': ('mprnet_derain', dict(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16)),
                 'real_noise': ('mprnet_denoise', dict(n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32))}
_NAFNET_TASKS = {'blur': ('nafnet_gopro', dict(enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]), True),
                 'real_noise': ('nafnet_sidd', dict(enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]), False)}


def _cmp_weights_key(method, bench):
    task = _cmp_task(bench)
    table = {'Restormer': _RESTORMER_TASKS, 'MPRNet': _MPRNET_TASKS, 'NAFNet': _NAFNET_TASKS,
             'PromptIR': {'rain': ('promptir_3task',), 'haze': ('promptir_3task',), 'gauss_noise': ('promptir_3task',),
                          'rain+haze': ('promptir_3task',)},
             'Histoformer': {'snow': ('histoformer_allweather',)}}[method]
    if task not in table:
        raise KeyError(f'{method} has no official weights for {bench!r} (task {task})')
    return table[task][0]


def build_restormer(device='cpu', bench='rain100l'):
    from deepinv.models import Restormer
    key, ln = _RESTORMER_TASKS[_cmp_task(bench)]
    net = Restormer(in_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                    heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False, LayerNorm_type=ln,
                    dual_pixel_task=False, pretrained=None)
    return _cmp_load(net, key).to(device)


def build_mprnet(device='cpu', bench='rain100l'):
    from deepinv.models import MPRNet
    key, arch = _MPRNET_TASKS[_cmp_task(bench)]
    net = MPRNet(in_channels=3, out_channels=3, num_cab=8, kernel_size=3, reduction=4, bias=False,
                 pretrained=None, device='cpu', **arch)
    return _cmp_load(net, key).to(device)


def build_promptir(device='cpu', bench='rain100l'):
    from deepinv.models import PromptIR
    net = PromptIR(in_channels=3, out_channels=3, dim=48, num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                   heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                   decoder=True, pretrained=None)
    return _cmp_load(net, 'promptir_3task').to(device)


def build_nafnet(device='cpu', bench='gopro'):
    key, arch, tlc = _NAFNET_TASKS[_cmp_task(bench)]
    if tlc:   # options/test/GoPro/NAFNet-width64.yml: type NAFNetLocal (TLC, train_size 256)
        net = NAF_NetLocal(img_channel=3, width=64, train_size=(1, 3, 256, 256), fast_imp=False, **arch)
    else:     # options/test/SIDD/NAFNet-width64.yml: type NAFNet
        net = NAF_Net(img_channel=3, width=64, **arch)
    return _cmp_load(net, key).to(device)


def build_histoformer(device='cpu', bench='snow100k_l'):
    # Allweather/Options/Allweather_Histoformer.yml
    net = Histoformer(inp_channels=3, out_channels=3, dim=36, num_blocks=[4, 4, 6, 8], num_refinement_blocks=4,
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.667, bias=False, LayerNorm_type='WithBias',
                      dual_pixel_task=False)
    return _cmp_load(net, 'histoformer_allweather').to(device)


# benchmarks: the benchmark keys a method has official weights for.  pad: inputs are reflect-padded at the
# bottom / right to a multiple of `pad` and the output cropped back (1 = none; NAFNet zero-pads internally).
COMPETITOR_MODELS = {
    'Restormer': dict(benchmarks=_BLUR + _RAIN + ['sidd'] + _GAUSS + ['dpdd'], build=build_restormer, pad=8,
                      source='github.com/swz30/Restormer releases v1.0 (motion_deblurring / deraining / real_denoising / '
                             'gaussian_color_denoising_blind / single_image_defocus_deblurring .pth); arch: deepinv.models.Restormer',
                      paper='Zamir et al., CVPR 2022'),
    'MPRNet': dict(benchmarks=_BLUR + _RAIN + ['sidd'], build=build_mprnet, pad=8,
                   source='huggingface.co/deepinv/MPRNet (model_deblurring / model_deraining / model_denoising .pth, '
                          'authors\' release); arch: deepinv.models.MPRNet',
                   paper='Zamir et al., CVPR 2021'),
    'NAFNet': dict(benchmarks=_BLUR + ['sidd'], build=build_nafnet, pad=1,
                   source='NAFNet-GoPro-width64.pth (huggingface.co/nyanko7/nafnet-models), NAFNet-SIDD-width64.pth '
                          '(huggingface.co/spaces/chuxiaojie/NAFNet); arch: megvii-research/NAFNet (vendored)',
                   paper='Chen et al., ECCV 2022'),
    'PromptIR': dict(benchmarks=_RAIN + ['reside6k'] + _GAUSS + ['mix_rain100l_haze'], build=build_promptir, pad=16,
                     source='huggingface.co/deepinv/promptir promptir.ckpt (official all-in-one noise/rain/haze model); '
                            'arch: deepinv.models.PromptIR',
                     paper='Potlapalli et al., NeurIPS 2023'),
    'Histoformer': dict(benchmarks=_SNOW, build=build_histoformer, pad=8,
                        source='net_g_best.pth All-weather model (huggingface.co/dronefreak/Histoformer mirror); '
                               'arch: huggingface.co/sunsean/Histoformer (vendored)',
                        paper='Sun et al., ECCV 2024'),
}


# ==========================================
# 4. Inference
# ==========================================

def _cmp_forward(model, x, pad):
    """Full-image forward with the method's padding rule; x: (1,3,H,W) in [0,1]."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % pad, (-w) % pad
    if ph or pw:
        mode = 'reflect' if (ph < h and pw < w) else 'replicate'
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y[..., :h, :w]


def _cmp_forward_tiled(model, x, pad, tile=512, overlap=32):
    """Overlapping tiles averaged uniformly (the Restormer demo's tiling) - used only after a CUDA OOM."""
    b, c, h, w = x.shape
    tile = min(tile, max(h, w))
    stride = max(tile - overlap, 1)
    hs = list(range(0, max(h - tile, 0), stride)) + [max(h - tile, 0)]
    ws = list(range(0, max(w - tile, 0), stride)) + [max(w - tile, 0)]
    E = torch.zeros(b, c, h, w, device=x.device)
    W = torch.zeros(b, 1, h, w, device=x.device)
    for i in hs:
        for j in ws:
            patch = x[..., i:i + tile, j:j + tile]
            out = _cmp_forward(model, patch, pad)
            E[..., i:i + patch.shape[-2], j:j + patch.shape[-1]] += out
            W[..., i:i + patch.shape[-2], j:j + patch.shape[-1]] += 1
    return E / W


def _cmp_is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()


@torch.no_grad()
def competitor_restore(model, x, pad, tile=None):
    """x: (3,H,W) or (1,3,H,W) float in [0,1] on the model's device -> (3,H,W) in [0,1].
    Full image first; on CUDA OOM falls back to 512 px tiles, then 256 px.  Returns (output, tile used)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    tiles = ([tile] if tile else [None]) + [t for t in (512, 256) if not tile or t < tile]
    for i, t in enumerate(tiles):
        try:
            y = _cmp_forward(model, x, pad) if t is None else _cmp_forward_tiled(model, x, pad, tile=t)
            return y.clamp(0, 1)[0], t
        except RuntimeError as e:
            if not _cmp_is_oom(e) or i == len(tiles) - 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


_CMP_LOADED = {'key': None, 'model': None}


def competitor_model(method, bench, device):
    """Build (or reuse the last built) model; only one competitor network is kept in memory."""
    key = (method, _cmp_weights_key(method, bench), str(device))
    if _CMP_LOADED['key'] != key:
        _CMP_LOADED['key'], _CMP_LOADED['model'] = None, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _CMP_LOADED['model'] = COMPETITOR_MODELS[method]['build'](device, bench)
        _CMP_LOADED['key'] = key
    return _CMP_LOADED['model']


def competitor_plan(cfg):
    """All (method, benchmark) pairs whose benchmark is in MAIN_BENCHMARKS + EXTRA_BENCHMARKS, method-major
    (consecutive pairs share the same checkpoint)."""
    benches = list(cfg.MAIN_BENCHMARKS) + [b for b in cfg.EXTRA_BENCHMARKS if b not in cfg.MAIN_BENCHMARKS]
    return [(m, b) for m, spec in COMPETITOR_MODELS.items() for b in benches if b in spec['benchmarks']]


def competitor_folder(cfg, bench, method):
    return os.path.join(cfg.COMPETITOR_ROOT, bench, method)


def competitor_folders(cfg):
    """{bench: {method: folder}} of every planned pair (to merge into cfg.COMPETITOR_OUTPUTS / FIGURE_COMPETITORS)."""
    out = {}
    for m, b in competitor_plan(cfg):
        out.setdefault(b, {})[m] = competitor_folder(cfg, b, m)
    return out


def _cmp_save_png(t, path):
    arr = (t.detach().clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    tmp = path + '.tmp.png'
    Image.fromarray(arr).save(tmp)
    os.replace(tmp, path)   # atomic: an interrupted session never leaves a truncated PNG that would be skipped


@torch.no_grad()
def generate_competitor_outputs(cfg, roots, method, bench, device=None, max_images=None):
    """Restore every image of `bench` with `method` and write COMPETITOR_ROOT/<bench>/<method>/<input stem>.png.
    Resumable; returns the output folder."""
    spec = COMPETITOR_MODELS[method]
    if bench not in spec['benchmarks']:
        raise ValueError(f'{method} has no official weights for {bench!r}')
    out_dir = competitor_folder(cfg, bench, method)
    os.makedirs(out_dir, exist_ok=True)
    ds = BenchmarkDataset(cfg, roots, bench, max_images=max_images)
    n = len(ds)
    done_path = os.path.join(out_dir, 'done.json')
    if os.path.isfile(done_path):
        try:
            with open(done_path) as f:
                if json.load(f).get('n') == n:
                    print(f'[competitors] {method} / {bench}: done ({n} images)')
                    return out_dir
        except (OSError, ValueError):
            pass
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    wkey = _cmp_weights_key(method, bench)
    t0 = time.time()
    model = None
    written = skipped = 0
    tiled = {}
    for i in range(n):
        item = ds[i]
        path = os.path.join(out_dir, os.path.splitext(item['name'])[0] + '.png')
        if os.path.isfile(path):
            skipped += 1
            continue
        if model is None:
            model = competitor_model(method, bench, device)
        x = item['input'].to(device=device, dtype=torch.float32)
        y, tile = competitor_restore(model, x, spec['pad'], tile=getattr(cfg, 'COMPETITOR_TILE', None))
        if tile:
            tiled[item['name']] = tile
        _cmp_save_png(y, path)
        written += 1
        if written % 50 == 0:
            print(f'[competitors] {method} / {bench}: {written + skipped}/{n}  ({time.time() - t0:.0f} s)')
    w = COMPETITOR_WEIGHTS[wkey]
    info = dict(n=n, method=method, benchmark=bench, written=written, skipped_existing=skipped,
                seconds=round(time.time() - t0, 1), device=str(device), weights=wkey, weights_file=w['file'],
                weights_url=w['urls'][0], weights_sha256=w['sha256'], weights_note=w['note'], source=spec['source'],
                paper=spec['paper'], protocol=dict(input_range='[0,1] RGB float32, batch 1', pad_multiple=spec['pad'],
                                                   pad_mode='reflect (bottom/right), output cropped' if spec['pad'] > 1 else 'none',
                                                   output='clamp [0,1], round(x*255) uint8 PNG'),
                tiled=tiled, max_images=max_images, torch=torch.__version__)
    with open(done_path, 'w') as f:
        json.dump(info, f, indent=1)
    print(f'[competitors] {method} / {bench}: {written} written, {skipped} existing, {info["seconds"]} s')
    return out_dir
