# ==========================================================================
# KASIRNet - six-term Omni-Restoration loss (Section 3.2)
#   L = w_pix L_pix + w_fft L_fft + w_perc L_perc + w_ssim L_ssim + w_id L_id + w_edge L_edge
#   defaults (paper): 5.0, 0.5, 1.0, 1.0, 1.0, 0.5
#   L_pix  : L1(pred, gt)
#   L_fft  : L1(|rFFT2(pred)| + 1e-6, |rFFT2(gt)| + 1e-6)
#   L_perc : sum_k 2^-(k-1) L1(phi_k(pred), phi_k(gt)), phi_k = frozen ImageNet VGG-19 blocks
#            ending at pool1..pool4 (same cut points as the SIM extractor), k = 1..4
#   L_ssim : 1 - SSIM_Y(pred, gt) (Gaussian 11x11, sigma 1.5; metrics.ssim_y, differentiable)
#   L_id   : L1 between the 2048-D global features of a frozen ImageNet ResNet-50 (no classifier)
#   L_edge : L1 between Sobel gradient magnitudes sqrt(gx^2 + gy^2 + 1e-6)
# Frozen networks are ImageNet-normalised, kept in eval mode, used only at training
# time (never loaded for inference).  A term whose weight is 0 is not computed.
# ==========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from metrics import ssim_y  # nb:skip

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _Frozen(nn.Module):
    def _finish(self):
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        return super().train(False)


class VGGFeatureExtractor(_Frozen):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(5)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(5, 10)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(10, 19)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(19, 28)])
        self._finish()

    def forward(self, x):
        h1 = self.slice1((x - self.mean) / self.std)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


class IdentityFeature(_Frozen):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self._finish()

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std).flatten(1)


class OmniRestorationLoss(nn.Module):
    TERMS = ('pix', 'fft', 'perc', 'ssim', 'id', 'edge')

    def __init__(self, weights, amp=False, amp_dtype=torch.float16):
        super().__init__()
        self.w = {k: float(weights.get(k, 0.0)) for k in self.TERMS}
        self.amp, self.amp_dtype = amp, amp_dtype
        self.vgg = VGGFeatureExtractor() if self.w['perc'] > 0 else None
        self.idnet = IdentityFeature() if self.w['id'] > 0 else None
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def compute_edge(self, x):
        ex = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        ey = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    def forward(self, preds, targets):
        preds, targets = preds.float(), targets.float()
        terms = {}
        if self.w['pix'] > 0:
            terms['pix'] = F.l1_loss(preds, targets)
        if self.w['fft'] > 0:
            eps = 1e-6
            terms['fft'] = F.l1_loss(torch.abs(torch.fft.rfft2(preds, norm='backward')) + eps,
                                     torch.abs(torch.fft.rfft2(targets, norm='backward')) + eps)
        dev = preds.device.type
        with torch.autocast(device_type=dev, dtype=self.amp_dtype, enabled=self.amp):
            if self.vgg is not None:
                with torch.no_grad():
                    tf = self.vgg(targets)
                pf = self.vgg(preds)
                terms['perc'] = sum(F.l1_loss(a.float(), b.float()) * (1.0 / (2 ** i)) for i, (a, b) in enumerate(zip(pf, tf)))
            if self.idnet is not None:
                with torch.no_grad():
                    t_id = self.idnet(targets)
                terms['id'] = F.l1_loss(self.idnet(preds).float(), t_id.float())
        if self.w['ssim'] > 0:
            terms['ssim'] = 1.0 - ssim_y(preds, targets).mean()
        if self.w['edge'] > 0:
            terms['edge'] = F.l1_loss(self.compute_edge(preds), self.compute_edge(targets))
        total = sum(self.w[k] * v for k, v in terms.items())
        logs = {k: float(v.detach()) for k, v in terms.items()}
        logs['total'] = float(total.detach())
        return total, logs
