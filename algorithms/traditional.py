"""
传统图像去噪算法 —— 无需训练、无需预训练权重
通过 nn.Module 包装以符合平台接口
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from denoising_platform.core.models import register_model


@register_model('gaussian_filter', category='traditional', display_name='高斯滤波去噪')
class GaussianFilterDenoiser(nn.Module):
    """
    高斯滤波去噪

    使用二维高斯核对图像做空间域平滑，
    有效抑制高斯白噪声，但会模糊边缘细节。
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('kernel', self._make_gaussian_kernel(kernel_size, sigma))

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        kernel = self.kernel.to(x.device).expand(C, 1, -1, -1)
        return F.conv2d(x, kernel, padding=pad, groups=C)


@register_model('poisson_nlm', category='traditional', display_name='泊松-NLM去噪')
class PoissonNLMDenoiser(nn.Module):
    """
    泊松噪声去噪（Anscombe 变换 + 非局部均值）

    处理流程：
      1. Anscombe 变换：将泊松噪声近似转化为高斯噪声
      2. 非局部均值 (NLM) 滤波：利用图像块相似度加权去噪
      3. 逆 Anscombe 变换：还原值域

    适用于低光照、医学成像等泊松噪声主导的场景。
    """

    def __init__(self, search_window: int = 11, patch_size: int = 3, h: float = 0.1):
        super().__init__()
        self.search_window = search_window
        self.patch_size = patch_size
        self.h = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        results = []
        for b in range(B):
            img_np = x[b].detach().cpu().numpy()  # [C, H, W]
            denoised_channels = []
            for c in range(C):
                ch = img_np[c]
                # Anscombe 变换
                ch_anscombe = 2.0 * np.sqrt(np.maximum(ch, 0) + 3.0 / 8.0)
                # NLM 去噪
                ch_denoised = self._nlm_filter(ch_anscombe)
                # 逆 Anscombe 变换
                ch_result = (ch_denoised / 2.0) ** 2 - 3.0 / 8.0
                denoised_channels.append(np.clip(ch_result, 0, 1))
            results.append(np.stack(denoised_channels, axis=0))
        out = torch.from_numpy(np.stack(results, axis=0)).float()
        return out.to(x.device)

    def _nlm_filter(self, img: np.ndarray) -> np.ndarray:
        """简化版非局部均值滤波"""
        H, W = img.shape
        pad_s = self.search_window // 2
        pad_p = self.patch_size // 2
        pad_total = pad_s + pad_p

        padded = np.pad(img, pad_total, mode='reflect')
        result = np.zeros_like(img)
        weight_sum = np.zeros_like(img)

        h2 = self.h ** 2
        patch_area = self.patch_size ** 2

        for di in range(-pad_s, pad_s + 1):
            for dj in range(-pad_s, pad_s + 1):
                shifted = padded[
                    pad_total + di: pad_total + di + H,
                    pad_total + dj: pad_total + dj + W,
                ]
                # 计算 patch 距离（用均匀滤波近似）
                diff_sq = (img - shifted) ** 2
                # 简化：直接用像素级距离
                weight = np.exp(-diff_sq / (h2 * patch_area + 1e-10))
                result += weight * shifted
                weight_sum += weight

        return result / (weight_sum + 1e-10)


@register_model('median_filter', category='traditional', display_name='中值滤波去噪')
class MedianFilterDenoiser(nn.Module):
    """
    中值滤波去噪

    对每个像素取邻域中值进行替换，
    对椒盐噪声（脉冲噪声）效果极佳，同时较好保留边缘。
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        x_padded = F.pad(x, [pad] * 4, mode='reflect')
        # 展开为滑动窗口
        unfolded = x_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # unfolded: [B, C, H, W, k, k]
        median_val = unfolded.contiguous().view(B, C, H, W, -1).median(dim=-1).values
        return median_val


@register_model('bilateral_filter', category='traditional', display_name='双边滤波去噪')
class BilateralFilterDenoiser(nn.Module):
    """
    双边滤波去噪

    同时考虑空间距离和像素值差异的加权平均，
    能在平滑噪声的同时较好保留图像边缘。
    """

    def __init__(self, kernel_size: int = 5, sigma_s: float = 2.0, sigma_r: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.register_buffer('spatial_kernel', self._make_spatial_kernel(kernel_size, sigma_s))

    @staticmethod
    def _make_spatial_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        spatial = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        return spatial  # [k, k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        pad = k // 2
        x_padded = F.pad(x, [pad] * 4, mode='reflect')
        unfolded = x_padded.unfold(2, k, 1).unfold(3, k, 1)  # [B, C, H, W, k, k]

        center = x.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
        range_weight = torch.exp(
            -(unfolded - center) ** 2 / (2 * self.sigma_r ** 2)
        )

        spatial = self.spatial_kernel.to(x.device)  # [k, k]
        combined_weight = range_weight * spatial.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        weight_sum = combined_weight.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
        result = (unfolded * combined_weight).sum(dim=(-2, -1), keepdim=True) / weight_sum
        return result.squeeze(-1).squeeze(-1)
