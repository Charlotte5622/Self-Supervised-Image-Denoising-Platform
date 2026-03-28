#!/usr/bin/env python3
"""Standalone CVF-SID inference script with embedded architecture."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT = THIS_DIR / "pretrained_models" / "CVF_SID_model_best.pth"
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
PAD = 20


class GenClean(nn.Module):
    def __init__(self, channels: int = 3, depth: int = 17, features: int = 64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, kernel_size=1))
        self.genclean = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.genclean(x)


class GenNoise(nn.Module):
    def __init__(self, layers: int = 10, features: int = 64):
        super().__init__()
        body = [nn.Conv2d(3, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(layers - 1):
            body += [nn.Conv2d(features, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)

        def head_block():
            blocks = []
            for _ in range(4):
                blocks += [nn.Conv2d(features, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            blocks.append(nn.Conv2d(features, 3, kernel_size=1))
            return nn.Sequential(*blocks)

        self.gen_noise_w = head_block()
        self.gen_noise_b = head_block()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self.body(x)
        noise_w = self.gen_noise_w(noise)
        noise_b = self.gen_noise_b(noise)
        noise_w = noise_w - noise_w.mean(dim=(-1, -2), keepdim=True)
        noise_b = noise_b - noise_b.mean(dim=(-1, -2), keepdim=True)
        return noise_w, noise_b


class CVFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_noise = GenNoise()
        self.genclean = GenClean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean = self.genclean(x)
        self.gen_noise(x - clean)  # keep noise branches for checkpoint compatibility
        return clean


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray


class CVFSIDDenoiser:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CVFModel().to(self.device)
        if "parse_config" not in sys.modules:
            stub = types.ModuleType("parse_config")
            stub.ConfigParser = type("ConfigParser", (), {})  # minimal placeholder
            sys.modules["parse_config"] = stub
        checkpoint = torch.load(weight_path or DEFAULT_WEIGHT, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            missing = ", ".join(incompatible.missing_keys)
            raise RuntimeError(f"Missing keys in CVF-SID checkpoint: {missing}")
        self.model.eval()

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def denoise_image(self, image: Image.Image) -> DenoiseResult:
        tensor = self._prepare_tensor(image)
        padded = F.pad(tensor, (PAD, PAD, PAD, PAD), mode="reflect")
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            clean = self.model(padded)
        clean = torch.clamp(clean, PIXEL_MIN, PIXEL_MAX)
        clean = clean[:, :, PAD:-PAD, PAD:-PAD]
        array = (clean.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return DenoiseResult(denoised=array)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    return Image.open(path).convert("RGB")


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _load_reference(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    if pred.shape != ref.shape:
        raise ValueError("Prediction and reference size mismatch")
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CVF-SID denoising on a single sRGB image.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Where to save the denoised image")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default=None, help="Device spec such as 'cuda:0' or 'cpu'")
    parser.add_argument("--reference", default=None, help="Optional clean reference for PSNR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weight_path = Path(args.weights).expanduser().resolve() if args.weights else DEFAULT_WEIGHT
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None

    denoiser = CVFSIDDenoiser(weight_path=str(weight_path), device=args.device)
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
