# ==========================================================================
# KASIRNet (Knowledge-Augmented Single Image Revamp Network) - model
# --------------------------------------------------------------------------
# With every flag at its default value the network built here is exactly the
# architecture of the submitted paper (same modules, same parameter names, same
# shapes, same forward computation): a submitted checkpoint loads with
# strict=True.  The flags only exist so that every ablation of the revision is
# "one line of Config" instead of a copy of this file:
#
#   use_sim / extractor            : Semantic Injection Module + frozen extractor
#                                    ('vgg19' = paper, 'resnet50', 'convnext_t')
#   use_srb / srb_replacement      : Spectral Resonance Block or its replacement
#                                    ('plain_conv' = same spatial branch, no FFT)
#   use_scwam / scwam_replacement  : shifted cyclic window attention or a local
#                                    residual conv block ('conv') / 'identity'
#   use_meshgrid                   : CoordConv-like Meshgrid coordinate embedding
#   shift_size / masked_shift /    : SCWAM design variants used to document how
#   rel_pos_bias                     SCWAM differs from Swin (mask, RPB)
#   use_prompt                     : Prompt Learner + the four SFT layers
#   use_gcb                        : Global Context Block in the bottleneck
#   num_scales / base_channels     : depth / width ablations
#
# Numerical notes (they do not change the function that is computed):
#   * the FFT path of the SRB always runs in float32, also under FP16 autocast
#     (cuFFT half precision only supports power-of-two sizes and the
#     un-normalised amplitude overflows the FP16 range);
#   * window attention uses torch's fused scaled_dot_product_attention, which
#     computes softmax(QK^T / sqrt(d)) V exactly like the original code.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==========================================
# 1. Core Components (SCWAM & Meshgrid)
# ==========================================

class MesgridCoordinateEmbedder(nn.Module):
    """Normalised (x, y) meshgrid in [-1, 1] embedded by a 1x1-conv MLP (CoordConv-like).
    The embedding is added to the features before window partitioning, so every
    token carries its absolute image position (no relative position bias table)."""

    def __init__(self, embed_dim):
        super(MesgridCoordinateEmbedder, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(2, embed_dim // 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        coords = torch.cat([x_coords, y_coords], dim=1)
        return self.embed(coords)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one window.  `rel_pos_bias=True` adds a
    Swin relative-position-bias table (ablation only, not used by KASIRNet)."""
    use_fused = True  # fused SDPA kernel; mathematically identical to the explicit form

    def __init__(self, dim, window_size, num_heads, rel_pos_bias=False):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.rel_pos_bias = rel_pos_bias
        if rel_pos_bias:
            ws = window_size
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
            coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')).flatten(1)
            rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
            rel[:, :, 0] += ws - 1
            rel[:, :, 1] += ws - 1
            rel[:, :, 0] *= 2 * ws - 1
            self.register_buffer('relative_position_index', rel.sum(-1), persistent=False)

    def _bias(self, N):
        if not self.rel_pos_bias:
            return None
        b = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        return b.view(N, N, -1).permute(2, 0, 1).contiguous()  # heads, N, N

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        bias = self._bias(N)
        if mask is not None:  # Swin-style shifted-window mask (ablation only)
            nW = mask.shape[0]
            m = mask.to(q.dtype).unsqueeze(1).repeat(B_ // nW, 1, 1, 1)  # B_, 1, N, N
            bias = m if bias is None else bias.unsqueeze(0).to(q.dtype) + m
        elif bias is not None:
            bias = bias.unsqueeze(0).to(q.dtype)

        if self.use_fused:
            # CUDA fused kernels launch one block per (window, head): their grid is limited to 65,535 in that
            # dimension (batch 64 x 1,024 windows x 2 heads exceeds it -> 'CUDA error: invalid argument').
            # The windows are therefore processed in chunks; the result is identical.
            step = max(1, 65535 // (2 * self.num_heads))
            if B_ <= step:
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            else:
                x = torch.cat([F.scaled_dot_product_attention(
                    q[s:s + step], k[s:s + step], v[s:s + step],
                    attn_mask=None if bias is None else (bias if bias.shape[0] == 1 else bias[s:s + step]))
                    for s in range(0, B_, step)], dim=0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if bias is not None:
                attn = attn + bias
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SCWAMLayer(nn.Module):
    """Shifted Cyclic Window Attention with Meshgrid.
    Differences with a Swin block (all are switches so that they can be ablated):
      * the window grid is cyclically rolled by `shift_size` in EVERY layer and no
        attention mask is applied, so tokens on opposite borders attend to each
        other (wrap-around context);  `masked_shift=True` restores the Swin mask;
      * absolute position comes from the Meshgrid embedding instead of a
        relative-position-bias table (`rel_pos_bias=True` adds the Swin table)."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=4, use_meshgrid=True,
                 masked_shift=False, rel_pos_bias=False):
        super(SCWAMLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_meshgrid = use_meshgrid
        self.masked_shift = masked_shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, rel_pos_bias=rel_pos_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        if use_meshgrid:
            self.grid_embedder = MesgridCoordinateEmbedder(dim)
        self._mask_cache = {}

    def _swin_mask(self, Hp, Wp, device):
        key = (Hp, Wp, str(device))
        if key not in self._mask_cache:
            ws, s = self.window_size, self.shift_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
            cnt = 0
            for h in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for w in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = self.window_partition(img_mask, ws).squeeze(-1)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            self._mask_cache[key] = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
        return self._mask_cache[key]

    def forward(self, x):
        B, C, H, W = x.shape

        if self.use_meshgrid:
            grid_emb = self.grid_embedder(x)
            x = x + grid_emb
        shortcut = x

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x).view(B, Hp, Wp, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        mask = self._swin_mask(Hp, Wp, x.device) if (self.masked_shift and self.shift_size > 0) else None
        x_windows = self.window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=mask)

        shifted_x = self.window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut.flatten(2).transpose(1, 2) + x
        x = x + self.mlp(self.norm2(x))

        return x.transpose(1, 2).view(B, C, H, W)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class LocalConvBlock(nn.Module):
    """Replacement used by the 'w/o SCWAM' ablation: residual depthwise 3x3 +
    pointwise MLP (local mixing only, no attention, no coordinates)."""

    def __init__(self, dim):
        super(LocalConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        return x + self.body(x)


# ==========================================
# 2. Spectral Resonance Block
# ==========================================

class SpectralResonanceBlock(nn.Module):
    """F = rFFT2(X);  A = |F|, Phi = angle(F)
       G = sigmoid(W2 ReLU(W1 GAP(A)))                  (channel gate computed on the amplitude)
       A' = A (1 + G)                                    (phase-preserving amplitude resonance)
       [R; I] = Conv1x1([A' cos Phi ; A' sin Phi])       (complex 1x1 mixing: MAY change the phase)
       Y = X + DWConv-PWConv(irFFT2(R + iI))
    Only the amplitude gate is phase-preserving; the learned complex mixing is not.
    The gate is not zero-initialised (G ~ 0.5 at initialisation)."""

    def __init__(self, dim):
        super(SpectralResonanceBlock, self).__init__()
        self.freq_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.resonance_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_restore = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

    def _spectral(self, x, trace=False):
        B, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            x_fft = torch.fft.rfft2(x32, norm='backward')
            mag = torch.abs(x_fft)
            pha = torch.angle(x_fft)

            gate = self.resonance_gate(mag)
            mag_resonant = mag * (1.0 + gate)

            real_res = mag_resonant * torch.cos(pha)
            imag_res = mag_resonant * torch.sin(pha)

            f_stack = torch.cat([real_res, imag_res], dim=1)
            f_stack = self.freq_conv(f_stack)

            real_final, imag_final = torch.chunk(f_stack, 2, dim=1)
            x_fft_final = torch.complex(real_final, imag_final)

            x_restored = torch.fft.irfft2(x_fft_final, s=(H, W), norm='backward')
        if trace:
            return dict(F_in=x_fft, F_gated=torch.polar(mag_resonant, pha), F_out=x_fft_final, gate=gate)
        return x_restored.to(x.dtype)

    def forward(self, x):
        identity = x
        x_restored = self._spectral(x)
        return identity + self.spatial_restore(x_restored)

    @torch.no_grad()
    def trace(self, x):
        """Intermediate spectra used by the phase-fidelity analysis (no side effect)."""
        return self._spectral(x, trace=True)


class PlainConvBlock(nn.Module):
    """Replacement used by the 'w/o SRB' ablation: the FFT/gate/complex-mixing path
    is replaced by a spatial 1x1 conv, the spatial branch is kept as in the SRB."""

    def __init__(self, dim):
        super(PlainConvBlock, self).__init__()
        self.mix = nn.Conv2d(dim, dim, 1)
        self.spatial_restore = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.spatial_restore(self.mix(x))


# ==========================================
# 3. Frozen semantic feature extractors
# ==========================================

class _FrozenExtractor(nn.Module):
    """Frozen ImageNet backbone returning four feature maps; always in eval mode."""

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):  # BatchNorm of ResNet must never switch to batch statistics
        return super().train(False)


class VGG19SemanticExtractor(_FrozenExtractor):
    """VGG-19 (ImageNet IMAGENET1K_V1), blocks conv1_1..pool1 / conv2_1..pool2 /
    conv3_1..pool3 / conv4_1..pool4 (feature index 0..27) -> 64/128/256/512 channels at
    1/2, 1/4, 1/8, 1/16 of the input resolution."""
    dims = [64, 128, 256, 512]

    def __init__(self, pretrained=True):
        super(VGG19SemanticExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.scale1 = nn.Sequential(*[vgg[i] for i in range(5)])
        self.scale2 = nn.Sequential(*[vgg[i] for i in range(5, 10)])
        self.scale3 = nn.Sequential(*[vgg[i] for i in range(10, 19)])
        self.scale4 = nn.Sequential(*[vgg[i] for i in range(19, 28)])
        self._freeze()

    def forward(self, x):
        x = (x - self.mean) / self.std
        f1 = self.scale1(x)
        f2 = self.scale2(f1)
        f3 = self.scale3(f2)
        f4 = self.scale4(f3)
        return [f1, f2, f3, f4]


class ResNet50SemanticExtractor(_FrozenExtractor):
    """ResNet-50 (ImageNet IMAGENET1K_V1) layer1..layer4 -> 256/512/1024/2048 channels."""
    dims = [256, 512, 1024, 2048]

    def __init__(self, pretrained=True):
        super(ResNet50SemanticExtractor, self).__init__()
        r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self._freeze()

    def forward(self, x):
        x = self.stem((x - self.mean) / self.std)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class ConvNeXtTSemanticExtractor(_FrozenExtractor):
    """ConvNeXt-T (ImageNet IMAGENET1K_V1) four stages -> 96/192/384/768 channels."""
    dims = [96, 192, 384, 768]

    def __init__(self, pretrained=True):
        super(ConvNeXtTSemanticExtractor, self).__init__()
        c = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = c.features
        self._freeze()

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in (1, 3, 5, 7):
                feats.append(x)
        return feats


SEMANTIC_EXTRACTORS = {
    'vgg19': VGG19SemanticExtractor,
    'resnet50': ResNet50SemanticExtractor,
    'convnext_t': ConvNeXtTSemanticExtractor,
}


# ==========================================
# 4. Semantic Injection Module
# ==========================================

class SemanticInjection(nn.Module):
    def __init__(self, restore_dim, semantic_dim):
        super(SemanticInjection, self).__init__()
        self.semantic_proj = nn.Conv2d(semantic_dim, restore_dim, 1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(restore_dim, restore_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(restore_dim // 4, restore_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(restore_dim, 1, 7, 1, 3),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(restore_dim * 2 + semantic_dim, restore_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(restore_dim, restore_dim, 3, 1, 1)
        )

    def forward(self, restore_feat, semantic_feat):
        B, C, H, W = restore_feat.shape
        identity = restore_feat

        sem_proj = self.semantic_proj(semantic_feat)
        if sem_proj.shape[2:] != restore_feat.shape[2:]:
            sem_proj = torch.nn.functional.interpolate(sem_proj, size=(H, W), mode='bilinear', align_corners=False)
            semantic_feat = torch.nn.functional.interpolate(semantic_feat, size=(H, W), mode='bilinear', align_corners=False)

        ch_attn = self.channel_attn(sem_proj)
        restore_ch = restore_feat * ch_attn

        sp_attn = self.spatial_attn(sem_proj)
        restore_sp = restore_feat * sp_attn

        fused = self.fusion(torch.cat([semantic_feat, restore_ch, restore_sp], dim=1))
        return identity + fused


# ==========================================
# 5. Encoder / Decoder blocks
# ==========================================

def _make_srb(dim, use_srb, replacement):
    if use_srb:
        return SpectralResonanceBlock(dim)
    if replacement == 'plain_conv':
        return PlainConvBlock(dim)
    return nn.Identity()


def _make_scwam(dim, num_heads, opts):
    if opts.get('use_scwam', True):
        return SCWAMLayer(dim, num_heads=num_heads, window_size=opts.get('window_size', 8),
                          shift_size=opts.get('shift_size', 4), use_meshgrid=opts.get('use_meshgrid', True),
                          masked_shift=opts.get('masked_shift', False), rel_pos_bias=opts.get('rel_pos_bias', False))
    if opts.get('scwam_replacement', 'conv') == 'conv':
        return LocalConvBlock(dim)
    return nn.Identity()


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, semantic_dim, num_heads=8, use_semantic=True, opts=None):
        super(EncoderBlock, self).__init__()
        opts = opts or {}
        self.use_semantic = use_semantic
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.srb = _make_srb(out_channels, opts.get('use_srb', True), opts.get('srb_replacement', 'plain_conv'))
        self.scwam = _make_scwam(out_channels, num_heads, opts)
        if use_semantic:
            self.sim = SemanticInjection(out_channels, semantic_dim)

    def forward(self, x, semantic_feat=None):
        x = self.downsample(x)
        x = self.srb(x)
        if self.use_semantic and semantic_feat is not None:
            x = self.sim(x, semantic_feat)
        x = self.scwam(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, semantic_dim, prompt_dim=256, num_heads=8,
                 use_semantic=True, use_prompt=True, opts=None):
        super(DecoderBlock, self).__init__()
        opts = opts or {}
        self.use_semantic = use_semantic
        self.use_prompt = use_prompt
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.srb = _make_srb(out_channels, opts.get('use_srb', True), opts.get('srb_replacement', 'plain_conv'))
        self.scwam = _make_scwam(out_channels, num_heads, opts)
        if use_prompt:
            self.sft = SFTLayer(out_channels, prompt_dim)
        if use_semantic:
            self.sim = SemanticInjection(out_channels, semantic_dim)

    def forward(self, x, skip, prompt_emb, semantic_feat=None):
        x = self.upsample(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = torch.nn.functional.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.srb(x)
        if self.use_semantic and semantic_feat is not None:
            x = self.sim(x, semantic_feat)
        x = self.scwam(x)
        if self.use_prompt and prompt_emb is not None:
            x = self.sft(x, prompt_emb)
        return x


class SFTLayer(nn.Module):
    """Channel-wise affine modulation by the 256-D visual descriptor p:
       y = x * (1 + gamma(p)) + beta(p).  gamma, beta are 1x1 convs applied to p broadcast
       over H x W, i.e. spatially constant (channel-wise SFT)."""

    def __init__(self, in_channels, prompt_dim=256):
        super(SFTLayer, self).__init__()
        self.sft_scale = nn.Conv2d(prompt_dim, in_channels, 1)
        self.sft_shift = nn.Conv2d(prompt_dim, in_channels, 1)

    def forward(self, x, prompt_emb):
        prompt = prompt_emb.view(prompt_emb.size(0), prompt_emb.size(1), 1, 1)
        prompt = prompt.expand(-1, -1, x.size(2), x.size(3))
        scale = self.sft_scale(prompt)
        shift = self.sft_shift(prompt)
        return x * (scale + 1) + shift


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContextBlock, self).__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LayerNorm([in_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        context = torch.matmul(input_x, mask)
        context = context.view(b, c, 1, 1)
        return x + self.channel_add_conv(context)


class PromptLearner(nn.Module):
    """Visual degradation descriptor p (256-D) computed from the degraded image by a
    small CNN (no text, no degradation label at inference)."""

    def __init__(self, in_channels=3, prompt_dim=256):
        super(PromptLearner, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, prompt_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(prompt_dim, prompt_dim)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


# ==========================================
# 6. KASIRNet Definition
# ==========================================

MODEL_DEFAULTS = dict(
    in_channels=3, out_channels=3, base_channels=64, prompt_dim=256,
    use_sim=True, extractor='vgg19', extractor_pretrained=True,
    use_srb=True, srb_replacement='plain_conv',
    use_scwam=True, scwam_replacement='conv', use_meshgrid=True,
    window_size=8, shift_size=4, masked_shift=False, rel_pos_bias=False,
    use_prompt=True, use_gcb=True, num_scales=4,
)


class KASIRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, prompt_dim=256, **flags):
        super(KASIRNet, self).__init__()
        unknown = set(flags) - set(MODEL_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown KASIRNet flags: {sorted(unknown)}')
        o = dict(MODEL_DEFAULTS, in_channels=in_channels, out_channels=out_channels,
                 base_channels=base_channels, prompt_dim=prompt_dim, **flags)
        self.flags = o
        self.use_sim, self.use_prompt, self.num_scales = o['use_sim'], o['use_prompt'], o['num_scales']
        assert self.num_scales in (3, 4), 'num_scales must be 3 or 4'
        self.prompt_dim = prompt_dim

        if self.use_sim:
            ext = SEMANTIC_EXTRACTORS[o['extractor']](pretrained=o['extractor_pretrained'])
            sem_dims = ext.dims
            if o['extractor'] == 'vgg19':
                self.vgg_extractor = ext          # parameter names of the submitted model
            else:
                self.semantic_extractor = ext
        else:
            sem_dims = [0, 0, 0, 0]
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*6, base_channels*8]
        if self.use_prompt:
            self.prompt_learner = PromptLearner(in_channels, prompt_dim)
        us, up = self.use_sim, self.use_prompt

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1)
        )
        self.head_attn = _make_scwam(ch[0], 2, o)

        self.encoder1 = EncoderBlock(ch[0], ch[1], sem_dims[0], num_heads=4, use_semantic=us, opts=o)
        self.encoder2 = EncoderBlock(ch[1], ch[2], sem_dims[1], num_heads=4, use_semantic=us, opts=o)
        self.encoder3 = EncoderBlock(ch[2], ch[3], sem_dims[2], num_heads=8, use_semantic=us, opts=o)
        if self.num_scales == 4:
            self.encoder4 = EncoderBlock(ch[3], ch[4], sem_dims[3], num_heads=8, use_semantic=us, opts=o)
        cb = ch[4] if self.num_scales == 4 else ch[3]

        self.bottleneck = nn.Sequential(
            GlobalContextBlock(cb) if o['use_gcb'] else nn.Identity(),
            _make_srb(cb, o['use_srb'], o['srb_replacement']),
            _make_scwam(cb, 8, o)
        )

        if self.num_scales == 4:
            self.decoder4 = DecoderBlock(ch[4], ch[3], ch[3], sem_dims[3], prompt_dim, num_heads=8, use_semantic=us, use_prompt=up, opts=o)
        self.decoder3 = DecoderBlock(ch[3], ch[2], ch[2], sem_dims[2], prompt_dim, num_heads=8, use_semantic=us, use_prompt=up, opts=o)
        self.decoder2 = DecoderBlock(ch[2], ch[1], ch[1], sem_dims[1], prompt_dim, num_heads=4, use_semantic=us, use_prompt=up, opts=o)
        self.decoder1 = DecoderBlock(ch[1], ch[0], ch[0], sem_dims[0], prompt_dim, num_heads=4, use_semantic=us, use_prompt=up, opts=o)

        self.tail = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch[0], out_channels, 3, 1, 1)
        )

    @property
    def extractor_module(self):
        return getattr(self, 'vgg_extractor', None) or getattr(self, 'semantic_extractor', None)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (16 - h % 16) % 16
        mod_pad_w = (16 - w % 16) % 16
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x, h, w

    def forward(self, x, prompt_override=None):
        """Returns (restored image, 256-D descriptor p).  `prompt_override` replaces p
        (used only by the mixed-degradation control experiment); p is None when the
        Prompt Learner is ablated."""
        x_padded, H_orig, W_orig = self.check_image_size(x)
        if self.use_sim:
            with torch.no_grad():
                sem = self.extractor_module(x_padded)
        else:
            sem = [None, None, None, None]

        if prompt_override is not None:
            prompt_emb = prompt_override
        elif self.use_prompt:
            prompt_emb = self.prompt_learner(x_padded)
        else:
            prompt_emb = None

        x0 = self.head_conv(x_padded)
        x0 = self.head_attn(x0)

        x1 = self.encoder1(x0, sem[0])
        x2 = self.encoder2(x1, sem[1])
        x3 = self.encoder3(x2, sem[2])
        if self.num_scales == 4:
            x4 = self.encoder4(x3, sem[3])
            x_mid = self.bottleneck(x4)
            d4 = self.decoder4(x_mid, x3, prompt_emb, sem[3])
        else:
            d4 = self.bottleneck(x3)
        d3 = self.decoder3(d4, x2, prompt_emb, sem[2])
        d2 = self.decoder2(d3, x1, prompt_emb, sem[1])
        d1 = self.decoder1(d2, x0, prompt_emb, sem[0])

        out = self.tail(d1)
        out = out + x_padded
        out = out[:, :, :H_orig, :W_orig].clamp(0, 1)

        return out, prompt_emb


# ==========================================
# 7. Parameter accounting (Table 10)
# ==========================================

def component_parameter_table(model):
    """Trainable parameters per component, frozen extractor parameters (millions)."""
    groups = {'SCWAM (incl. Meshgrid)': (SCWAMLayer, LocalConvBlock), 'SIM': (SemanticInjection,),
              'SRB': (SpectralResonanceBlock, PlainConvBlock), 'Prompt Learner': (PromptLearner,),
              'SFT': (SFTLayer,), 'GCB': (GlobalContextBlock,)}
    seen, out = set(), {}
    for gname, types in groups.items():
        n = 0
        for m in model.modules():
            if isinstance(m, types):
                for p in m.parameters():
                    if id(p) not in seen and p.requires_grad:
                        seen.add(id(p)); n += p.numel()
        out[gname] = n / 1e6
    other = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in seen)
    out['head/tail/sampling/fusion'] = other / 1e6
    out['trainable total'] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    out['frozen extractor'] = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    out['loaded at inference'] = out['trainable total'] + out['frozen extractor']
    return out
