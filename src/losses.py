import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from src.metrics import calculate_ssim


class CharbonnierLoss(nn.Module):
    """A smooth L1 loss variant that is robust to outliers and stable near zero."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class VGGPerceptualLoss(nn.Module):
    """Compares deep feature representations from a pre-trained VGG16 network
    to preserve textures and structural details from a human perceptual view."""

    def __init__(self):
        super().__init__()
        # Extract the first 9 layers of VGG16 (up to the 3rd conv block)
        vgg_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:9].eval()
        for param in vgg_features.parameters():
            param.requires_grad = False
        self.vgg = vgg_features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_norm), self.vgg(target_norm))


class ColorConstancyLoss(nn.Module):
    """Penalizes hue shifts by comparing the cosine similarity of RGB vectors,
    preventing the model from altering the original color balance during enhancement."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(pred + 1e-2, target + 1e-2, dim=1, eps=1e-2)
        return torch.mean(1.0 - cos_sim)


class CompositeLowLightLoss(nn.Module):
    """Weighted combination of Charbonnier, SSIM, VGG perceptual, and Color Constancy losses."""

    def __init__(self, w_ssim: float = 0.2, w_vgg: float = 0.02, w_color: float = 0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.perceptual = VGGPerceptualLoss()
        self.color = ColorConstancyLoss()
        self.w_ssim = w_ssim
        self.w_vgg = w_vgg
        self.w_color = w_color

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force FP32 for loss computation to avoid precision underflow during AMP
        with torch.amp.autocast('cuda', enabled=False):
            pred_f32 = torch.clamp(pred.float(), 0.0, 1.0)
            target_f32 = target.float()

            l_char = self.charbonnier(pred_f32, target_f32)
            l_ssim = calculate_ssim(pred_f32, target_f32, return_loss=True)
            l_vgg = self.perceptual(pred_f32, target_f32)
            l_color = self.color(pred_f32, target_f32)

            return l_char + (self.w_ssim * l_ssim) + (self.w_vgg * l_vgg) + (self.w_color * l_color)