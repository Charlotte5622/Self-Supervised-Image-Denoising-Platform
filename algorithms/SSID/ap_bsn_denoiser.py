#!/usr/bin/env python3
"""Standalone AP-BSN inference script with embedded network definitions.

This module provides both a CLI (`python ap_bsn_denoiser.py --input noisy.png --output clean.png`)
 and a programmatic wrapper (`APBSNDenoiser`) for denoising sRGB images using the
 AP-BSN model with fixed hyper-parameters taken from `conf/APBSN_SIDD.yaml`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


###############################################################################
# Constants & paths
###############################################################################
THIS_DIR = Path(__file__).resolve().parent
APBSN_WEIGHT_ENV = "APBSN_WEIGHTS"
DEFAULT_WEIGHT_NAME = "APBSN_SIDD.pth"
DEFAULT_WEIGHT_DIR = Path(os.environ.get(APBSN_WEIGHT_ENV, THIS_DIR / "pretrained_models"))

# Hyper-parameters from conf/APBSN_SIDD.yaml
APBSN_PARAMS = dict(
    pd_a=5,
    pd_b=2,
    pd_pad=2,
    R3=True,
    R3_T=8,
    R3_p=0.16,
    bsn="DBSNl",
    in_ch=3,
    bsn_base_ch=128,
    bsn_num_module=9,
)

PIXEL_MIN = 0.0
PIXEL_MAX = 255.0


###############################################################################
# Utility ops (pixel shuffle variants)
###############################################################################


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.0) -> torch.Tensor:
    """Pixel-shuffle down-sampling (AAAI 2019)."""
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(
            c, w + 2 * f * pad, h + 2 * f * pad
        )
    b, c, w, h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(
        b, c, w + 2 * f * pad, h + 2 * f * pad
    )


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0) -> torch.Tensor:
    """Inverse operation of `pixel_shuffle_down_sampling`."""
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    b, c, w, h = x.shape
    before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f, h // f)
    if pad != 0:
        before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)


###############################################################################
# Network definitions (DBSNl + AP-BSN wrapper)
###############################################################################


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class DCl(nn.Module):
    def __init__(self, stride: int, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DC_branchl(nn.Module):
    def __init__(self, stride: int, in_ch: int, num_module: int):
        super().__init__()
        layers = [
            CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        layers += [DCl(stride, in_ch) for _ in range(num_module)]
        layers += [nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9):
        super().__init__()
        assert base_ch % 2 == 0, "base channel must be divisible by 2"
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        x = torch.cat([br1, br2], dim=1)
        return self.tail(x)


class APBSN(nn.Module):
    def __init__(
        self,
        pd_a: int = 5,
        pd_b: int = 2,
        pd_pad: int = 2,
        R3: bool = True,
        R3_T: int = 8,
        R3_p: float = 0.16,
        bsn: str = "DBSNl",
        in_ch: int = 3,
        bsn_base_ch: int = 128,
        bsn_num_module: int = 9,
    ):
        super().__init__()
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        if bsn != "DBSNl":
            raise NotImplementedError(f"Unsupported BSN backbone: {bsn}")
        self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)

    def forward(self, img: torch.Tensor, pd: Optional[int] = None) -> torch.Tensor:
        pd = self.pd_a if pd is None else pd
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            return pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        p = self.pd_pad
        return pd_img_denoised[:, :, p:-p, p:-p]

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b))
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0))
        img_pd_bsn = self.forward(x, pd=self.pd_b)
        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]
        denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = torch.rand_like(x) < self.R3_p
            tmp_input = torch.clone(img_pd_bsn).detach()
            tmp_input[mask] = x[mask]
            p = self.pd_pad
            tmp_input = F.pad(tmp_input, (p, p, p, p), mode="reflect")
            if self.pd_pad == 0:
                denoised[..., t] = self.bsn(tmp_input)
            else:
                denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


###############################################################################
# Wrapper & helpers
###############################################################################


def _resolve_weight_path(weight_arg: Optional[str]) -> Path:
    candidates = []
    if weight_arg:
        candidates.append(Path(weight_arg).expanduser())
    env_path = os.environ.get("APBSN_WEIGHT_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_WEIGHT_DIR / DEFAULT_WEIGHT_NAME)
    for path in candidates:
        if path and path.exists():
            return path
    search_str = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find AP-BSN weights. Checked:\n{search_str}")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "model_weight" in checkpoint and isinstance(checkpoint["model_weight"], dict):
        model_weight = checkpoint["model_weight"]
        if "denoiser" in model_weight and isinstance(model_weight["denoiser"], dict):
            return model_weight["denoiser"]
        return model_weight
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


@dataclass(frozen=True)
class APBSNDenoiseResult:
    denoised: np.ndarray


class APBSNDenoiser:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = APBSN(**APBSN_PARAMS).to(self.device)
        checkpoint_path = _resolve_weight_path(weight_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_tensor(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, (tensor.shape[2], tensor.shape[3])

    def denoise_image(self, image: Image.Image) -> APBSNDenoiseResult:
        tensor, _ = self._prepare_tensor(image)
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            output = self.model.denoise(tensor)
        output = torch.clamp(output, PIXEL_MIN, PIXEL_MAX)
        array = output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return APBSNDenoiseResult(denoised=array)


###############################################################################
# CLI
###############################################################################


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
    parser = argparse.ArgumentParser(description="Run AP-BSN denoising on a single sRGB image.")
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
    denoiser = APBSNDenoiser(weight_path=args.weights, device=args.device)
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
