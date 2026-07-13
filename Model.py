import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# ==========================================================================
# KASIRNet (Omni-Restoration Network)
# Full Production Implementation
# ==========================================================================

# ==========================================
# 1. Core Components (SCWAM & Mesgrid)
# ==========================================

class MesgridCoordinateEmbedder(nn.Module):
    """
    Generates a spatial meshgrid (x, y) and embeds it into feature space.
    Gives the attention mechanism explicit geometric awareness.
    """
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
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SCWAMLayer(nn.Module):
    """
    Shifted Cyclic Window Attention with Mesgrid.
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=4):
        super(SCWAMLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.grid_embedder = MesgridCoordinateEmbedder(dim)

    def forward(self, x):
        B, C, H, W = x.shape

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

        x_windows = self.window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows)

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


# ==========================================
# 2. Spectral Resonance Block
# ==========================================

class SpectralResonanceBlock(nn.Module):
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

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        x_fft = torch.fft.rfft2(x, norm='backward')
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
        return identity + self.spatial_restore(x_restored)


# ==========================================
# 3. VGG19 Semantic Feature Extractor
# ==========================================

class VGG19SemanticExtractor(nn.Module):
    def __init__(self):
        super(VGG19SemanticExtractor, self).__init__()
        vgg = models.vgg19(weights=None).features
        self.scale1 = nn.Sequential(*[vgg[i] for i in range(5)])
        self.scale2 = nn.Sequential(*[vgg[i] for i in range(5, 10)])
        self.scale3 = nn.Sequential(*[vgg[i] for i in range(10, 19)])
        self.scale4 = nn.Sequential(*[vgg[i] for i in range(19, 28)])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        f1 = self.scale1(x)
        f2 = self.scale2(f1)
        f3 = self.scale3(f2)
        f4 = self.scale4(f3)
        return [f1, f2, f3, f4]


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
# 5. Enhanced Encoder/Decoder Blocks
# ==========================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, semantic_dim, num_heads=8, use_semantic=True):
        super(EncoderBlock, self).__init__()
        self.use_semantic = use_semantic
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.srb = SpectralResonanceBlock(out_channels)
        self.scwam = SCWAMLayer(out_channels, num_heads=num_heads)
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
    def __init__(self, in_channels, skip_channels, out_channels, semantic_dim, prompt_dim=256, num_heads=8, use_semantic=True):
        super(DecoderBlock, self).__init__()
        self.use_semantic = use_semantic
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.srb = SpectralResonanceBlock(out_channels)
        self.scwam = SCWAMLayer(out_channels, num_heads=num_heads)
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
        x = self.sft(x, prompt_emb)
        return x


class SFTLayer(nn.Module):
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

class KASIRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, prompt_dim=256):
        super(KASIRNet, self).__init__()
        self.vgg_extractor = VGG19SemanticExtractor()
        vgg_dims = [64, 128, 256, 512]
        ch = [base_channels, base_channels*2, base_channels*4, base_channels*6, base_channels*8]
        self.prompt_learner = PromptLearner(in_channels, prompt_dim)

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1)
        )
        self.head_attn = SCWAMLayer(ch[0], num_heads=2)

        self.encoder1 = EncoderBlock(ch[0], ch[1], vgg_dims[0], num_heads=4)
        self.encoder2 = EncoderBlock(ch[1], ch[2], vgg_dims[1], num_heads=4)
        self.encoder3 = EncoderBlock(ch[2], ch[3], vgg_dims[2], num_heads=8)
        self.encoder4 = EncoderBlock(ch[3], ch[4], vgg_dims[3], num_heads=8)

        self.bottleneck = nn.Sequential(
            GlobalContextBlock(ch[4]),
            SpectralResonanceBlock(ch[4]),
            SCWAMLayer(ch[4], num_heads=8)
        )

        self.decoder4 = DecoderBlock(ch[4], ch[3], ch[3], vgg_dims[3], prompt_dim, num_heads=8)
        self.decoder3 = DecoderBlock(ch[3], ch[2], ch[2], vgg_dims[2], prompt_dim, num_heads=8)
        self.decoder2 = DecoderBlock(ch[2], ch[1], ch[1], vgg_dims[1], prompt_dim, num_heads=4)
        self.decoder1 = DecoderBlock(ch[1], ch[0], ch[0], vgg_dims[0], prompt_dim, num_heads=4)

        self.tail = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch[0], out_channels, 3, 1, 1)
        )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (16 - h % 16) % 16
        mod_pad_w = (16 - w % 16) % 16
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x, h, w

    def forward(self, x):
        x_padded, H_orig, W_orig = self.check_image_size(x)
        with torch.no_grad():
            vgg_feats = self.vgg_extractor(x_padded)

        prompt_emb = self.prompt_learner(x_padded)

        x0 = self.head_conv(x_padded)
        x0 = self.head_attn(x0)

        x1 = self.encoder1(x0, vgg_feats[0])
        x2 = self.encoder2(x1, vgg_feats[1])
        x3 = self.encoder3(x2, vgg_feats[2])
        x4 = self.encoder4(x3, vgg_feats[3])

        x_mid = self.bottleneck(x4)

        d4 = self.decoder4(x_mid, x3, prompt_emb, vgg_feats[3])
        d3 = self.decoder3(d4, x2, prompt_emb, vgg_feats[2])
        d2 = self.decoder2(d3, x1, prompt_emb, vgg_feats[1])
        d1 = self.decoder1(d2, x0, prompt_emb, vgg_feats[0])

        out = self.tail(d1)
        out = out + x_padded
        out = out[:, :, :H_orig, :W_orig].clamp(0, 1)

        return out, prompt_emb