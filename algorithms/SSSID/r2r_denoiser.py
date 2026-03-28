#!/usr/bin/env python3
"""Standalone Recorrupted-to-Recorrupted denoiser for single noisy RGB images.

This script reimplements the core DnCNN + R2R training loop from
Recorrupted-to-Recorrupted-Unsupervised-Deep-Learning-for-Image-Denoising
in a single file so it has no runtime dependency on the original repo.
Given one noisy sRGB image, it performs a brief self-supervised training
session using the recorruption strategy, then outputs the denoised result.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Model definition (DnCNN)
# -----------------------------------------------------------------------------


class DnCNN(nn.Module):
    def __init__(self, channels: int, num_layers: int = 17) -> None:
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = [
            nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = self.net(x)
        return x - residual


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.clamp(0, 1).mul(255.0).byte().cpu().numpy()
    array = array.transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def random_patch_batch(
    img: torch.Tensor, patch_size: int, batch_size: int
) -> torch.Tensor:
    """Sample random patches (with replacement) from the input image tensor."""

    c, h, w = img.shape
    crop = min(patch_size, h, w)
    if crop < patch_size:
        patch_size = crop
    if patch_size <= 0:
        raise ValueError("Patch size must be >0")
    if h == patch_size and w == patch_size:
        batch = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return batch
    top = torch.randint(0, h - patch_size + 1, (batch_size,), device=img.device)
    left = torch.randint(0, w - patch_size + 1, (batch_size,), device=img.device)
    patches = torch.stack(
        [img[:, t : t + patch_size, l : l + patch_size] for t, l in zip(top.tolist(), left.tolist())],
        dim=0,
    )
    return patches


def train_r2r(
    noisy_image: torch.Tensor,
    *,
    noise_level: float,
    alpha: float,
    patch_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: torch.device,
    log_every: int,
) -> DnCNN:
    model = DnCNN(channels=noisy_image.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    eps = noise_level / 255.0
    noisy_image = noisy_image.to(device)
    model.train()
    for step in range(1, steps + 1):
        batch = random_patch_batch(noisy_image, patch_size, batch_size)
        gaussian = torch.randn_like(batch)
        input_batch = torch.clamp(batch + alpha * eps * gaussian, 0.0, 1.0)
        target_batch = torch.clamp(batch - (eps / alpha) * gaussian, 0.0, 1.0)
        pred = model(input_batch)
        loss = criterion(pred, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1 or step == steps:
            print(f"[R2R] step {step:05d}/{steps} - loss {loss.item():.6f}")
    return model


def run_denoising(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    noisy = read_image(input_path)
    if args.normalize_max:
        max_val = noisy.max().item()
        if max_val > 0:
            noisy = noisy / max_val

    model = train_r2r(
        noisy,
        noise_level=args.noise,
        alpha=args.alpha,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        device=device,
        log_every=args.log_interval,
    )

    model.eval()
    with torch.no_grad():
        preds = torch.clamp(model(noisy.unsqueeze(0).to(device)), 0.0, 1.0).squeeze(0).cpu()

    save_image(preds, output_path)
    if reference_path is not None:
        reference = read_image(reference_path)
        reference = reference.to(preds.dtype)
        psnr = compute_psnr(preds, reference)
        print(f"PSNR: {psnr:.4f} dB")
    else:
        print("Reference not provided; PSNR skipped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorrupted-to-Recorrupted single-image denoiser")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Path to save the denoised image")
    parser.add_argument("--reference", help="Optional clean reference image for PSNR")
    parser.add_argument("--device", default=None, help="Device spec, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--noise", type=float, default=25.0, help="Assumed AWGN sigma (0-255 scale)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Recorruption strength")
    parser.add_argument("--patch-size", type=int, default=40, help="Training patch size")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--steps", type=int, default=2000, help="Number of R2R training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between log lines")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--normalize-max",
        action="store_true",
        help="Scale image by its max value before training (helps HDR inputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_denoising(args)


if __name__ == "__main__":
    main()
