import torch
import torch.nn as nn


# =============================================
# LOSS FUNCTIONS
# =============================================

class ELRICLoss(nn.Module):
    def __init__(self, cfg, lpips_net=None):
        super().__init__()
        self.lam_preview      = cfg["lambda_preview"]
        self.lam_ds_match     = cfg["lambda_ds_match"]
        self.lam_sr           = cfg["lambda_sr"]
        self.lam_msssim       = cfg["lambda_msssim"]
        self.lam_lpips        = cfg["lambda_lpips"]
        self.downscale_factor = cfg["downscale_factor"]
        self.bpp_floor        = cfg["bpp_floor"]
        self.lam_bpp_floor    = cfg["lambda_bpp_floor"]
        self.bpp_ceiling      = cfg["bpp_ceiling"]
        self.lam_bpp_ceiling  = cfg["lambda_bpp_ceiling"]
        self.bpp_hard_div     = cfg["bpp_hard_div"]
        self.lpips_net        = lpips_net

    def forward(self, x, out, lambda_rd=0.013):
        x = x.clamp(0.0, 1.0)
        B, C, H, W = x.shape
        num_pixels  = B * H * W
        
        xhc = out["x_hat_compress"].clamp(0.0, 1.0)
        xhsr = out["x_hat_sr"].clamp(0.0, 1.0)
        xpv = out["x_preview"].clamp(0.0, 1.0)
        xds = out["x_downscaled"].clamp(0.0, 1.0)
        
        ly, lz = out["likelihoods_y"], out["likelihoods_z"]
        bpp      = compute_bpp(ly, lz, num_pixels)
        bpp_hard = bpp / self.bpp_hard_div
        
        L_floor   = self.lam_bpp_floor   * F.relu(self.bpp_floor   - bpp_hard).pow(2)
        L_ceiling = self.lam_bpp_ceiling * F.relu(bpp - self.bpp_ceiling).pow(2)
        
        d_compress = F.mse_loss(xhc, x)
        d_preview  = F.mse_loss(xpv, x)
        
        sf = self.downscale_factor
        x_aa = F.avg_pool2d(x, kernel_size=sf, stride=sf).clamp(0.0, 1.0)
        d_ds_match = F.mse_loss(xds, x_aa)
        d_sr = F.mse_loss(xhsr, x)
        
        if H >= 160 and W >= 160:
            ms_c  = compute_ms_ssim_lib(xhc, x, data_range=1., size_average=True)
            ms_sr = compute_ms_ssim_lib(xhsr, x, data_range=1., size_average=True)
        else:
            ms_c  = compute_ssim_lib(xhc, x, data_range=1., size_average=True)
            ms_sr = compute_ssim_lib(xhsr, x, data_range=1., size_average=True)
            
        L_msssim = (1. - ms_c).clamp(0,1) + (1. - ms_sr).clamp(0,1)
        lpips_c  = self.lpips_net(xhc, x).mean() if self.lpips_net else torch.tensor(0., device=x.device)
        
        total = (lambda_rd * bpp + d_compress + L_floor + L_ceiling
                 + self.lam_preview * d_preview + self.lam_ds_match * d_ds_match
                 + self.lam_sr * d_sr + self.lam_msssim * L_msssim + self.lam_lpips * lpips_c)
                 
        return {"total": total, "bpp": bpp.detach(), "bpp_hard": bpp_hard.detach(),
                "lambda_rd": torch.tensor(lambda_rd), "d_compress": d_compress.detach(),
                "d_preview": d_preview.detach(), "d_ds_match": d_ds_match.detach(),
                "d_sr": d_sr.detach(), "L_msssim": L_msssim.detach(),
                "L_lpips": lpips_c.detach(), "L_floor": L_floor.detach(), "L_ceiling": L_ceiling.detach()}

