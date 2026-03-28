#!/usr/bin/env python3
"""Standalone RSCP2GAN Restormer inference script."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT_DIR = Path(os.environ.get("RSCP2GAN_WEIGHTS", THIS_DIR / "pretrained_models"))
DEFAULT_WEIGHT_NAME = "RSCP2GAN_model_250.pth"

PIXEL_MIN = 0.0
PIXEL_MAX = 1.0


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layernorm_type: str):
        super().__init__()
        if layernorm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool, layernorm_type: str):
        super().__init__()
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layernorm_type: str = 'WithBias',
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[3])
        ])
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[2])
        ])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[1])
        ])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_blocks[0])
        ])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias, layernorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        enc1 = self.patch_embed(inp_img)
        enc1_out = self.encoder_level1(enc1)
        enc2 = self.down1_2(enc1_out)
        enc2_out = self.encoder_level2(enc2)
        enc3 = self.down2_3(enc2_out)
        enc3_out = self.encoder_level3(enc3)
        enc4 = self.down3_4(enc3_out)
        latent = self.latent(enc4)
        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3_out], dim=1)
        dec3 = self.reduce_chan_level3(dec3)
        dec3 = self.decoder_level3(dec3)
        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2_out], dim=1)
        dec2 = self.reduce_chan_level2(dec2)
        dec2 = self.decoder_level2(dec2)
        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1_out], dim=1)
        dec1 = self.decoder_level1(dec1)
        dec1 = self.refinement(dec1)
        return self.output(dec1) + inp_img


def _resolve_weight_path(weight_path: Optional[str]) -> Path:
    candidates = []
    if weight_path:
        candidates.append(Path(weight_path).expanduser())
    env_path = os.environ.get("RSCP2GAN_WEIGHT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_WEIGHT_DIR / DEFAULT_WEIGHT_NAME)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find RSCP2GAN weights in expected locations")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        state_dict = checkpoint["model_state_dict"]
        if "D" in state_dict and isinstance(state_dict["D"], dict):
            return state_dict["D"]
        return state_dict
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
        if "D" in state_dict and isinstance(state_dict["D"], dict):
            return state_dict["D"]
        return state_dict
    return checkpoint


@dataclass(frozen=True)
class RSCP2GANDenoiseResult:
    denoised: np.ndarray


class RSCP2GANDenoiser:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Restormer().to(self.device)
        checkpoint_path = _resolve_weight_path(weight_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def denoise_image(self, image: Image.Image) -> RSCP2GANDenoiseResult:
        tensor = self._prepare_tensor(image)
        _, _, h_old, w_old = tensor.size()
        h_pad = (h_old // 8 + 1) * 8 - h_old
        w_pad = (w_old // 8 + 1) * 8 - w_old
        img_lq = torch.cat([tensor, torch.flip(tensor, [2])], dim=2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], dim=3)[:, :, :, :w_old + w_pad]
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            denoised = self.model(img_lq)
        denoised = denoised[:, :, :h_old, :w_old]
        denoised = torch.clamp(denoised, PIXEL_MIN, PIXEL_MAX)
        array = denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return RSCP2GANDenoiseResult(denoised=array)


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
        raise ValueError(f"Predicted image shape {pred.shape} does not match reference {ref.shape}")
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSCP2GAN Restormer denoising on a single sRGB image.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Path to save the denoised image")
    parser.add_argument("--weights", default=None, help="Optional custom checkpoint path")
    parser.add_argument("--device", default=None, help="Optional device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--reference", default=None, help="Optional clean reference image to compute PSNR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    denoiser = RSCP2GANDenoiser(weight_path=args.weights, device=args.device)
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
