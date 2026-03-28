#!/usr/bin/env python3
"""Self-contained implementation of the P2N+ self-supervised denoiser.

This script distills the key ideas of the official P2N+ project into one file:
 - Symmetric-Permutation Invariant (SPI) UNet backbone
 - Recirculated Dual Consistency (RDC) and Double Self Consistency (DSC) losses
The model adapts itself to a *single* noisy RGB image using unsupervised
recorruption, then saves the denoised prediction. No external project files
are required.
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
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.clamp(0, 1).mul(255.0).byte().cpu().numpy().transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def random_patch_batch(img: torch.Tensor, patch: int, batch: int) -> torch.Tensor:
    _, h, w = img.shape
    size = min(patch, h, w)
    if size <= 0:
        raise ValueError("Patch size must be positive")
    if h == size and w == size:
        return img.unsqueeze(0).repeat(batch, 1, 1, 1)
    top = torch.randint(0, h - size + 1, (batch,), device=img.device)
    left = torch.randint(0, w - size + 1, (batch,), device=img.device)
    patches = []
    for t, l in zip(top.tolist(), left.tolist()):
        patches.append(img[:, t : t + size, l : l + size])
    return torch.stack(patches, dim=0)


# -----------------------------------------------------------------------------
# SPI UNet backbone (ported from the official network.py)
# -----------------------------------------------------------------------------


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * torch.sigmoid(self.beta * x)


class BaseConv(nn.Module):
    def __init__(self, inp: int, out: int, act: str = "swish") -> None:
        super().__init__()
        self.layer = nn.Conv2d(inp, out, 3, padding=1, bias=True)
        self.act = Swish() if act == "swish" else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.act(self.layer(x))


class SymLayer(nn.Module):
    def __init__(self, inp: int, out: int, act: str = "swish") -> None:
        super().__init__()
        mid = out // 2
        self.layer_en = BaseConv(inp, mid, act)
        self.layer_de = BaseConv(mid, out, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        f_pos = self.layer_en(x)
        f_neg = self.layer_en(-x)
        f_even = 0.5 * (f_pos + f_neg)
        f_odd = 0.5 * (f_pos - f_neg)
        return self.layer_de(f_even) + self.layer_de(f_odd)


class SPIUNet(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        block = SymLayer
        self.en1 = nn.Sequential(block(3, 48), block(48, 48), nn.MaxPool2d(2))
        self.en2 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en3 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en4 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en5 = nn.Sequential(block(48, 48), nn.MaxPool2d(2), block(48, 48), nn.Upsample(scale_factor=2))

        self.de1 = nn.Sequential(block(96, 96), nn.LeakyReLU(0.1, inplace=True), block(96, 96), nn.Upsample(scale_factor=2))
        self.de2 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de3 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de4 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de5 = nn.Sequential(block(96 + in_channels, 64), block(64, 32), nn.Conv2d(32, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        p1 = self.en1(x)
        p2 = self.en2(p1)
        p3 = self.en3(p2)
        p4 = self.en4(p3)
        u5 = self.en5(p4)

        c5 = torch.cat([u5, p4], dim=1)
        u4 = self.de1(c5)
        c4 = torch.cat([u4, p3], dim=1)
        u3 = self.de2(c4)
        c3 = torch.cat([u3, p2], dim=1)
        u2 = self.de3(c3)
        c2 = torch.cat([u2, p1], dim=1)
        u1 = self.de4(c2)
        c1 = torch.cat([u1, x], dim=1)
        return self.de5(c1)


# -----------------------------------------------------------------------------
# Loss components inspired by P2N+
# -----------------------------------------------------------------------------


class VariablePowerLoss(nn.Module):
    def __init__(self, mode: str = "1.5", eps: float = 1e-8) -> None:
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, step: int, total: int) -> torch.Tensor:  # type: ignore[override]
        if self.mode == "1.5":
            factor = 0.25
        elif self.mode == "0":
            factor = 1.0
        else:
            return F.l1_loss(pred, target)
        r = 2.0 * (1 - (step / max(total, 1)) * factor)
        r = max(r, 0.5)
        return torch.mean(torch.pow(torch.abs(pred - target) + self.eps, r))


# -----------------------------------------------------------------------------
# Training step (RDC+DSC style)
# -----------------------------------------------------------------------------


def p2n_step(
    model: SPIUNet,
    batch: torch.Tensor,
    sigma: float,
    step: int,
    total: int,
    loss_fn: VariablePowerLoss,
    clip: bool = True,
) -> torch.Tensor:
    out = model(batch)
    noise = batch - out
    rand = lambda: torch.randn(1, device=batch.device) * sigma + 1.0

    noise_pos = noise * rand()
    noise_neg = -noise * rand()

    def branch(n: torch.Tensor) -> torch.Tensor:
        perturbed = out + n
        if clip:
            perturbed = torch.clamp(perturbed, 0.0, 1.0)
        return model(perturbed)

    out_pos = branch(noise_pos)
    out_neg = branch(noise_neg)
    return loss_fn(out_pos, out_neg, step, total)


# -----------------------------------------------------------------------------
# High-level routine
# -----------------------------------------------------------------------------


def adapt_model(
    noisy: torch.Tensor,
    *,
    patch_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    sigma: float,
    device: torch.device,
    log_every: int,
    loss_mode: str,
) -> SPIUNet:
    model = SPIUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    noisy = noisy.to(device)
    loss_fn = VariablePowerLoss(loss_mode)

    for step in range(1, steps + 1):
        batch = random_patch_batch(noisy, patch_size, batch_size)
        loss = p2n_step(model, batch, sigma, step, steps, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1 or step == steps:
            print(f"[P2N+] step {step:05d}/{steps} - loss {loss.item():.6f}")
    return model


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    noisy = read_image(Path(args.input))
    reference = read_image(Path(args.reference)) if args.reference else None

    model = adapt_model(
        noisy,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        sigma=args.sigma,
        device=device,
        log_every=args.log_interval,
        loss_mode=args.loss_mode,
    )

    model.eval()
    with torch.no_grad():
        pred = torch.clamp(model(noisy.unsqueeze(0).to(device)), 0.0, 1.0).squeeze(0).cpu()
    save_image(pred, Path(args.output))
    if reference is not None:
        psnr = compute_psnr(pred, reference)
        print(f"PSNR: {psnr:.4f} dB")
    else:
        print("Reference not provided; PSNR skipped.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone P2N+ single-image denoiser")
    parser.add_argument("--input", required=True, help="Noisy RGB image path")
    parser.add_argument("--output", required=True, help="Where to save the denoised image")
    parser.add_argument("--reference", help="Optional clean reference for PSNR")
    parser.add_argument("--device", default=None, help="Torch device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--patch-size", type=int, default=96, help="Training patch size")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--steps", type=int, default=1500, help="Number of adaptation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Optimizer learning rate")
    parser.add_argument("--sigma", type=float, default=0.10, help="Noise scaling factor for RDC/DSC")
    parser.add_argument("--log-interval", type=int, default=100, help="How often to print training loss")
    parser.add_argument("--loss-mode", choices=["0", "1.5", "l1"], default="1.5", help="Which dynamic loss exponent to use")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
