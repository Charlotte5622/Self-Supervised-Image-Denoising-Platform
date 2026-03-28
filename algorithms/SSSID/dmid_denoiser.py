#!/usr/bin/env python3
"""Standalone DMID Gaussian denoiser for a single noisy sRGB image."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

DEFAULT_WEIGHTS = THIS_DIR / "pretrained_models" / "dmid" / "256x256_diffusion_uncond.pt"

try:
    from guided_diffusion.unet import UNetModel
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Cannot import local guided_diffusion package. "
        "Please make sure `algorithm/guided_diffusion` exists."
    ) from exc


def data_transform(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x - 1.0


def data_transform_reverse(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def imread_uint(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imsave_uint(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(str(path), img)


def uint2tensor4(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )


def tensor2uint(img: torch.Tensor) -> np.ndarray:
    array = img.detach().squeeze().float().clamp_(0.0, 1.0).cpu().numpy()
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    return np.uint8((array * 255.0).round())


def get_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> torch.Tensor:
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    return torch.from_numpy(betas).float()


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def generalized_steps(
    x: torch.Tensor,
    seq,
    model: nn.Module,
    betas: torch.Tensor,
    eta: float = 0.0,
    seq_next=None,
):
    with torch.no_grad():
        n = x.size(0)
        if seq_next is None:
            seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n, device=x.device) * i).long()
            next_t = (torch.ones(n, device=x.device) * j).long()
            at = compute_alpha(betas, t)
            at_next = compute_alpha(betas, next_t)
            xt = xs[-1].to(x.device)
            et = model(xt, t.float())
            if et.size(1) == 6:
                et = et[:, :3]
            x0_t = (xt - et * (1.0 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.cpu())
            c1 = eta * ((1.0 - at / at_next) * (1.0 - at_next) / (1.0 - at)).sqrt()
            c2 = ((1.0 - at_next) - c1 ** 2).sqrt()
            z = torch.randn_like(x0_t)
            xt_next = at_next.sqrt() * x0_t + c1 * z + c2 * et
            xs.append(xt_next.cpu())
    return xs, x0_preds


def precompute_ratio(betas: torch.Tensor) -> torch.Tensor:
    timesteps = torch.arange(len(betas), device=betas.device)
    a = compute_alpha(betas, timesteps)
    return torch.sqrt((1.0 - a) / a)


def find_N_from_sigma(sigma: float) -> int:
    betas = get_beta_schedule(0.0001, 0.02, 1000)
    ratio = precompute_ratio(betas)
    target = torch.tensor(float(sigma))
    return int(torch.argmin(torch.abs(ratio - target)).item()) + 1


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
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


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    return float(np.mean([_ssim_single_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))


class GaussianDMID:
    def __init__(self, checkpoint_path: Path, device: torch.device, sampling_timesteps: int = 1) -> None:
        self.device = device
        self.sampling_timesteps = sampling_timesteps
        self.model = UNetModel().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        if torch.cuda.device_count() > 1 and self.device.type == "cuda":
            self.model = nn.DataParallel(self.model)
        self.betas = get_beta_schedule(0.0001, 0.02, 1000).to(device)

    def _embed_to_timestep(self, x: torch.Tensor, diffusion_times: int) -> torch.Tensor:
        t = torch.full((x.shape[0],), diffusion_times - 1, device=self.device, dtype=torch.long)
        a = (1.0 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        return x * a.sqrt()

    def denoise(self, noisy: torch.Tensor, diffusion_times: int, eta: float = 0.8) -> torch.Tensor:
        noisy = noisy.to(self.device)
        _, _, h, w = noisy.shape
        factor = 64
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
        x_t = self._embed_to_timestep(data_transform(noisy), diffusion_times)
        skip = max(diffusion_times // self.sampling_timesteps, 1)
        seq = list(range(0, diffusion_times, skip))
        seq = list(seq[:-1]) + [diffusion_times - 1]
        xs, _ = generalized_steps(x_t, seq, self.model, self.betas, eta=eta)
        denoised = data_transform_reverse(xs[-1].to(self.device))
        return denoised[:, :, :h, :w]

    def denoise_average(
        self,
        noisy: torch.Tensor,
        diffusion_times: int,
        eta: float = 0.8,
        repeat_times: int = 1,
    ) -> torch.Tensor:
        outputs = [self.denoise(noisy, diffusion_times, eta=eta) for _ in range(repeat_times)]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)


def resolve_diffusion_times(sigma: int) -> int:
    return find_N_from_sigma(2.0 * float(sigma) / 255.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMID Gaussian single-image denoiser")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Path to save the denoised sRGB image")
    parser.add_argument("--reference", help="Optional clean sRGB image for PSNR/SSIM")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="DMID diffusion checkpoint")
    parser.add_argument("--sigma", type=int, default=75, help="Default assumes AWGN sigma=75")
    parser.add_argument(
        "--diffusion-times",
        type=int,
        default=33,
        help="Directly set the DMID sampling parameter N; default is 33",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--sampler-steps", type=int, default=1, help="Sampling steps inside one run")
    parser.add_argument("--repeat", type=int, default=3, help="Repeated runs to average")
    parser.add_argument("--eta", type=float, default=0.8, help="Generalized/DDIM eta")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_times = args.diffusion_times if args.diffusion_times > 0 else resolve_diffusion_times(args.sigma)
    noisy = uint2tensor4(imread_uint(Path(args.input)))
    denoiser = GaussianDMID(weights, device=device, sampling_timesteps=args.sampler_steps)

    print(
        f"[DMID] device={device} sigma={args.sigma} diffusion_times={diffusion_times} "
        f"sampler_steps={args.sampler_steps} repeat={args.repeat}"
    )

    with torch.no_grad():
        denoised = denoiser.denoise_average(
            noisy=noisy,
            diffusion_times=diffusion_times,
            eta=args.eta,
            repeat_times=args.repeat,
        )

    denoised_uint = tensor2uint(denoised)
    imsave_uint(denoised_uint, Path(args.output))
    print(f"[DMID] saved: {args.output}")

    if args.reference:
        clean = imread_uint(Path(args.reference))
        psnr = calculate_psnr(denoised_uint, clean)
        ssim = calculate_ssim(denoised_uint, clean)
        print(f"[DMID] PSNR: {psnr:.4f} dB")
        print(f"[DMID] SSIM: {ssim:.6f}")


if __name__ == "__main__":
    main()
