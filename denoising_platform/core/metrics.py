"""
图像质量评估指标：PSNR 和 SSIM

支持 numpy 数组和 torch 张量两种输入格式
"""
import numpy as np
from typing import Union
import torch


def _to_numpy(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """统一转换为 float64 numpy 数组，值域 [0, 1]"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float64)
    if img.ndim == 4:      # [B,C,H,W] → [C,H,W]
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW → HWC
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def compute_psnr(img1: Union[np.ndarray, torch.Tensor],
                 img2: Union[np.ndarray, torch.Tensor],
                 data_range: float = 1.0) -> float:
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)

    PSNR = 10 * log10(MAX² / MSE)
    值越高表示两张图越接近，一般去噪后 > 25dB 即有明显改善
    """
    a, b = _to_numpy(img1), _to_numpy(img2)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(img1: Union[np.ndarray, torch.Tensor],
                 img2: Union[np.ndarray, torch.Tensor],
                 data_range: float = 1.0,
                 win_size: int = 7) -> float:
    """
    计算结构相似性指数 (Structural Similarity Index Measure)

    使用滑动窗口比较亮度、对比度和结构三个分量
    值域 [-1, 1]，越接近 1 表示两张图结构越相似
    """
    try:
        from skimage.metrics import structural_similarity
        a, b = _to_numpy(img1), _to_numpy(img2)
        multichannel = a.ndim == 3 and a.shape[2] > 1
        return float(structural_similarity(
            a, b,
            data_range=data_range,
            win_size=win_size,
            channel_axis=2 if multichannel else None,
        ))
    except ImportError:
        return _ssim_numpy(img1, img2, data_range)


def _ssim_numpy(img1, img2, data_range: float = 1.0) -> float:
    """纯 numpy 实现的 SSIM（skimage 不可用时的后备方案）"""
    a, b = _to_numpy(img1), _to_numpy(img2)
    if a.ndim == 3:
        # 对各通道分别计算后取均值
        return float(np.mean([
            _ssim_single_channel(a[:, :, c], b[:, :, c], data_range)
            for c in range(a.shape[2])
        ]))
    return _ssim_single_channel(a, b, data_range)


def _ssim_single_channel(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a_sq = a.var()
    sigma_b_sq = b.var()
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
    return float(num / den)
