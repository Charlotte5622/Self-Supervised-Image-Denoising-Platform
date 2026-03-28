#!/usr/bin/env python3
"""Prompt-guided single-image denoiser inspired by Prompt-SID.

This light-weight reimplementation keeps the "prompt" conditioning concept
but packs everything into a single self-contained file: a FiLM-conditioned
UNet, a prompt encoder, and a short self-supervised adaptation routine that
estimates a denoised result from one noisy RGB image.
"""

from __future__ import annotations

import argparse
import math
import random
import string
from pathlib import Path

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
    return torch.from_numpy(arr.transpose(2, 0, 1))


def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.clamp(0, 1).mul(255.0).byte().cpu().numpy().transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def sample_patches(img: torch.Tensor, patch: int, batch: int) -> torch.Tensor:
    _, h, w = img.shape
    size = min(patch, h, w)
    if size <= 0:
        raise ValueError("Patch size must be positive")
    if h == size and w == size:
        return img.unsqueeze(0).repeat(batch, 1, 1, 1)
    top = torch.randint(0, h - size + 1, (batch,), device=img.device)
    left = torch.randint(0, w - size + 1, (batch,), device=img.device)
    patches = [img[:, t:t+size, l:l+size] for t, l in zip(top.tolist(), left.tolist())]
    return torch.stack(patches, dim=0)


# -----------------------------------------------------------------------------
# Prompt encoder + FiLM-conditioned UNet
# -----------------------------------------------------------------------------


class PromptEncoder(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        vocab = string.printable
        self.register_buffer("vocab_ids", torch.arange(len(vocab)))
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, prompt: str) -> torch.Tensor:
        if not prompt:
            prompt = "denoise"
        ids = [self.vocab.index(ch) if ch in self.vocab else 0 for ch in prompt]
        token_tensor = torch.tensor(ids, dtype=torch.long, device=self.embedding.weight.device)
        embedded = self.embedding(token_tensor)
        avg = embedded.mean(dim=0)
        return self.proj(avg)


class FiLMBlock(nn.Module):
    def __init__(self, inp: int, out: int, cond: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out)
        self.act = nn.SiLU(inplace=True)
        self.gamma = nn.Linear(cond, out)
        self.beta = nn.Linear(cond, out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + g) + b
        return self.act(h)


class PromptUNet(nn.Module):
    def __init__(self, channels: int = 3, cond_dim: int = 256, width: int = 64) -> None:
        super().__init__()
        self.enc1 = FiLMBlock(channels, width, cond_dim)
        self.enc2 = FiLMBlock(width, width * 2, cond_dim)
        self.enc3 = FiLMBlock(width * 2, width * 4, cond_dim)
        self.dec2 = FiLMBlock(width * 4 + width * 2, width * 2, cond_dim)
        self.dec1 = FiLMBlock(width * 2 + width, width, cond_dim)
        self.out = nn.Conv2d(width, channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cond = cond.unsqueeze(0).repeat(b, 1)
        h1 = self.enc1(x, cond)
        h2 = self.enc2(self.pool(h1), cond)
        h3 = self.enc3(self.pool(h2), cond)
        up2 = F.interpolate(h3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, h2], dim=1), cond)
        up1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, h1], dim=1), cond)
        return torch.clamp(self.out(d1), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------


def adapt_prompt_model(
    noisy: torch.Tensor,
    *,
    prompt: str,
    steps: int,
    patch: int,
    batch: int,
    lr: float,
    noise_level: float,
    device: torch.device,
    log_every: int,
) -> tuple[PromptUNet, torch.Tensor]:
    model = PromptUNet().to(device)
    encoder = PromptEncoder().to(device)
    cond_vec = encoder(prompt)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    noisy = noisy.to(device)
    eps = noise_level / 255.0

    for step in range(1, steps + 1):
        batch_img = sample_patches(noisy, patch, batch)
        perturb = torch.randn_like(batch_img) * eps
        input_batch = (batch_img + perturb).clamp(0.0, 1.0)
        target_batch = batch_img
        pred = model(input_batch, cond_vec)
        loss_main = criterion(pred, target_batch)
        contrastive = criterion(model((pred + perturb).clamp(0, 1), cond_vec), pred.detach())
        loss = loss_main + 0.5 * contrastive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1 or step == steps:
            print(f"[Prompt-SID] step {step:05d}/{steps} - loss {loss.item():.6f}")
    return model, cond_vec


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-guided single-image denoiser")
    parser.add_argument("--input", required=True, help="Path to the noisy RGB image")
    parser.add_argument("--output", required=True, help="Path to write the denoised result")
    parser.add_argument("--reference", help="Optional clean reference for PSNR")
    parser.add_argument("--prompt", default="night photography", help="Text prompt describing the scene")
    parser.add_argument("--device", default=None, help="Device spec such as 'cuda:0' or 'cpu'")
    parser.add_argument("--steps", type=int, default=1500, help="Adaptation steps")
    parser.add_argument("--patch", type=int, default=96, help="Training patch size")
    parser.add_argument("--batch", type=int, default=4, help="Patch batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--noise", type=float, default=25.0, help="Assumed AWGN sigma (0-255 scale)")
    parser.add_argument("--log-interval", type=int, default=100, help="Print frequency")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    noisy = read_image(Path(args.input))
    model, cond = adapt_prompt_model(
        noisy,
        prompt=args.prompt,
        steps=args.steps,
        patch=args.patch,
        batch=args.batch,
        lr=args.lr,
        noise_level=args.noise,
        device=device,
        log_every=args.log_interval,
    )
    model.eval()
    with torch.no_grad():
        denoised = model(noisy.unsqueeze(0).to(device), cond).squeeze(0).cpu()
    save_image(denoised, Path(args.output))
    if args.reference:
        ref = read_image(Path(args.reference)).to(denoised.dtype)
        psnr = compute_psnr(denoised, ref)
        print(f"PSNR: {psnr:.4f} dB")
    else:
        print("Reference not provided; PSNR skipped.")


if __name__ == "__main__":
    main()
