import math
import torch
import torch.nn.functional as F


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.detach().float(), target.detach().float(), reduction="mean").item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, return_loss: bool = False):
    # Force FP32 to avoid precision underflow in local variance computation under AMP
    with torch.amp.autocast('cuda', enabled=False):
        pred, target = pred.float(), target.float()
        channels = pred.size(1)
        window = _gaussian_window(window_size, 1.5, channels, pred.device)
        padding = window_size // 2

        mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
        mu2 = F.conv2d(target, window, padding=padding, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_val = (numerator / (denominator + 1e-6)).mean()
        return (1.0 - ssim_val) if return_loss else ssim_val.item()