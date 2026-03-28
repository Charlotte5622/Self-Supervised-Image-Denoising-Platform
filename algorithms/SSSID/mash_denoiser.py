#!/usr/bin/env python3
"""Standalone MASH single-image denoiser following the official training logic."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


CONFIG = {
    "input_channels": 3,
    "std_kernel_size": 2,
    "shuffling_tile_size": 4,
    "lr": 4e-4,
    "masking_threshold": 0.5,
    "num_predictions": 10,
    "num_iterations": 800,
    "shuffling_iteration": 50,
    "mask_high": 0.8,
    "mask_low": 0.2,
    "mask_medium": 0.5,
    "epsilon_low": 1.0,
    "epsilon_high": 2.0,
    "probe_mask_ratios": (0.8, 0.2),
}


class UNetN2NUn(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
        )
        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
        )
        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
        )
        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
        )
        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.de_block3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.de_block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(32, out_channels, 3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self.de_block5(concat1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="RGB")
    img.save(path)


def compute_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - ref.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def compute_ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([_ssim_single_channel(pred[:, :, i], ref[:, :, i]) for i in range(3)]))


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)


def smooth(noisy: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    channels = noisy.shape[1]
    kernel = torch.full(
        (channels, 1, kernel_size, kernel_size),
        1.0 / float(kernel_size * kernel_size),
        dtype=noisy.dtype,
        device=noisy.device,
    )
    return F.conv2d(noisy, kernel, stride=stride, padding=0, groups=channels)


def calculate_sliding_std(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    upsampler = nn.Upsample(scale_factor=kernel_size, mode="nearest")
    slided_mean = smooth(img, kernel_size, stride=kernel_size)
    mean_upsampled = upsampler(slided_mean)
    variance = smooth((img - mean_upsampled) ** 2, kernel_size, stride=kernel_size)
    upsampled_variance = upsampler(variance)
    return upsampled_variance.sqrt()


def get_shuffling_mask(std_map_torch: torch.Tensor, threshold: float) -> np.ndarray:
    std_map = std_map_torch.detach().cpu().numpy().squeeze()
    max_value = float(std_map.max())
    normalized = std_map / max_value if max_value > 0 else np.zeros_like(std_map)
    thresholded = np.zeros_like(normalized, dtype=np.float32)
    thresholded[normalized >= threshold] = 1.0
    return thresholded


def generate_random_permutation(height: int, width: int, channels: int, tile: int) -> torch.Tensor:
    num_tiles = (height // tile) * (width // tile)
    permutation = torch.argsort(torch.rand(1, num_tiles, tile * tile), dim=-1)
    return permutation.repeat(channels, 1, 1)


def shuffle_input(
    img: np.ndarray,
    indices: torch.Tensor,
    mask: np.ndarray,
    channels: int,
    height: int,
    width: int,
    tile: int,
) -> np.ndarray:
    img_torch = torch.from_numpy(img).float()
    mask_torch = torch.from_numpy(mask).float().unsqueeze(0).repeat(channels, 1, 1)
    h_tiles = height // tile
    w_tiles = width // tile
    img_tiles = img_torch.view(channels, h_tiles, tile, w_tiles, tile)
    img_tiles = img_tiles.permute(0, 1, 3, 2, 4).reshape(channels, h_tiles * w_tiles, tile * tile)
    mask_tiles = mask_torch.view(channels, h_tiles, tile, w_tiles, tile)
    mask_tiles = mask_tiles.permute(0, 1, 3, 2, 4).reshape(channels, h_tiles * w_tiles, tile * tile)
    mask_tiles, _ = torch.max(mask_tiles, dim=2, keepdim=True)
    shuffled = torch.gather(img_tiles.clone(), dim=-1, index=indices)
    mixed = mask_tiles * img_tiles + (1.0 - mask_tiles) * shuffled
    mixed = mixed.reshape(channels, h_tiles, w_tiles, tile, tile)
    mixed = mixed.permute(0, 1, 3, 2, 4).reshape(channels, height, width)
    return mixed.numpy()


def build_model(device: torch.device) -> UNetN2NUn:
    model = UNetN2NUn().to(device)
    model.train()
    return model


def random_mask(shape: torch.Size, threshold: float, device: torch.device) -> torch.Tensor:
    return (torch.rand(shape, device=device) < threshold).float()


def run_prediction_average(
    model: nn.Module,
    noisy: torch.Tensor,
    mask_threshold: float,
    num_predictions: int,
) -> torch.Tensor:
    preds = []
    with torch.no_grad():
        for _ in range(num_predictions):
            mask = random_mask(noisy.shape, mask_threshold, noisy.device)
            preds.append(model(mask * noisy).detach())
    return torch.stack(preds, dim=0).mean(dim=0)


def train_probe_model(noisy: torch.Tensor, mask_ratio: float, device: torch.device) -> nn.Module:
    model = build_model(device)
    criterion = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG["num_iterations"])

    for _ in range(CONFIG["num_iterations"]):
        mask = random_mask(noisy.shape, mask_ratio, device)
        output = model(mask * noisy)
        loss = criterion((1.0 - mask) * output, (1.0 - mask) * noisy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model.eval()


def estimate_probe_std(model: nn.Module, noisy: torch.Tensor, mask_ratio: float) -> float:
    avg = run_prediction_average(model, noisy, mask_ratio, CONFIG["num_predictions"])
    avg_np = avg.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    noisy_np = noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return float(np.std(avg_np * 255.0 - noisy_np * 255.0))


def create_locally_shuffled_target(
    model: nn.Module,
    noisy_original: torch.Tensor,
    mask_ratio: float,
) -> torch.Tensor:
    avg = run_prediction_average(
        model,
        noisy_original,
        1.0 - mask_ratio,
        CONFIG["num_predictions"],
    )
    mean_gray = (avg * 255.0).mean(dim=1, keepdim=True)
    std_map = calculate_sliding_std(mean_gray, CONFIG["std_kernel_size"])
    shuffling_mask = get_shuffling_mask(std_map, CONFIG["masking_threshold"])
    _, channels, height, width = noisy_original.shape
    permutation_indices = generate_random_permutation(
        height,
        width,
        channels,
        CONFIG["shuffling_tile_size"],
    )
    shuffled_np = shuffle_input(
        noisy_original.squeeze(0).detach().cpu().numpy(),
        permutation_indices,
        shuffling_mask,
        channels,
        height,
        width,
        CONFIG["shuffling_tile_size"],
    )
    return torch.from_numpy(shuffled_np).unsqueeze(0).to(noisy_original.device)


def train_final_model(noisy_original: torch.Tensor, apply_local_shuffling: bool, mask_ratio: float) -> nn.Module:
    model = build_model(noisy_original.device)
    criterion = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG["num_iterations"])
    noisy_shuffled = noisy_original.detach().clone()
    mask_threshold = 1.0 - mask_ratio

    for iteration in range(CONFIG["num_iterations"]):
        mask = random_mask(noisy_original.shape, mask_threshold, noisy_original.device)
        output = model(mask * noisy_original)
        loss = criterion((1.0 - mask) * output, (1.0 - mask) * noisy_shuffled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration == CONFIG["shuffling_iteration"] and apply_local_shuffling:
            noisy_shuffled = create_locally_shuffled_target(model, noisy_original, mask_ratio)

    return model.eval()


def run_mash(noisy_np: np.ndarray, device: torch.device) -> np.ndarray:
    noisy = torch.from_numpy(noisy_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pad_multiple = 32
    noisy, original_hw = pad_to_multiple(noisy, pad_multiple)

    probe_results = []
    for mask_ratio in CONFIG["probe_mask_ratios"]:
        probe_model = train_probe_model(noisy, mask_ratio, device)
        std_value = estimate_probe_std(probe_model, noisy, mask_ratio)
        probe_results.append((mask_ratio, std_value))
        print(f"[MASH] probe mask_ratio={mask_ratio:.1f}, estimated_std={std_value:.4f}")

    diff_std = abs(probe_results[0][1] - probe_results[1][1])
    apply_local_shuffling = False
    if diff_std > CONFIG["epsilon_high"]:
        apply_local_shuffling = True
        mask_ratio = CONFIG["mask_high"]
    elif diff_std < CONFIG["epsilon_low"]:
        mask_ratio = CONFIG["mask_low"]
    else:
        mask_ratio = CONFIG["mask_medium"]

    print(
        f"[MASH] |std_high-std_low|={diff_std:.4f}, "
        f"selected_mask_ratio={mask_ratio:.1f}, local_shuffling={apply_local_shuffling}"
    )

    model = train_final_model(noisy, apply_local_shuffling, mask_ratio)
    prediction = run_prediction_average(model, noisy, 1.0 - mask_ratio, CONFIG["num_predictions"])
    prediction = prediction[:, :, : original_hw[0], : original_hw[1]]
    output = prediction.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(output, 0.0, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MASH single-image denoiser")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Path to save the denoised sRGB image")
    parser.add_argument("--reference", help="Optional clean sRGB image for PSNR/SSIM")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    noisy = read_image(Path(args.input))
    denoised = run_mash(noisy, device)
    save_image(denoised, Path(args.output))
    print(f"[MASH] saved: {args.output}")

    if args.reference:
        ref = read_image(Path(args.reference))
        denoised_u8 = np.uint8(np.clip(denoised, 0.0, 1.0) * 255.0)
        ref_u8 = np.uint8(np.clip(ref, 0.0, 1.0) * 255.0)
        psnr = compute_psnr(denoised_u8, ref_u8)
        ssim = compute_ssim(denoised_u8, ref_u8)
        print(f"[MASH] PSNR: {psnr:.4f} dB")
        print(f"[MASH] SSIM: {ssim:.6f}")


if __name__ == "__main__":
    main()
