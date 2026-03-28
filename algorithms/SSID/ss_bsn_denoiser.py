#!/usr/bin/env python3
"""Standalone SS-BSN inference script."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT = THIS_DIR / "pretrained_models" / "SSBSN_SIDD.pth"
PIXEL_MIN = 0.0
PIXEL_MAX = 255.0

SSBSN_PARAMS = dict(
    pd_a=5,
    pd_b=2,
    pd_pad=2,
    R3=True,
    R3_T=8,
    R3_p=0.16,
    in_ch=3,
    bsn_base_ch=128,
    bsn_num_module=9,
    mode=["na", "na", "na", "na", "na", "na", "ss", "ss", "ss"],
    f_scale=2,
    ss_exp_factor=1.0,
)


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.0) -> torch.Tensor:
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


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class SSBlockNaive(nn.Module):
    def __init__(self, stride: int, in_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=stride, dilation=stride)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=stride, dilation=stride)
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._forward(x)


class SSBlock(SSBlockNaive):
    def __init__(self, stride: int, in_ch: int, f_scale: int = 2, ss_exp_factor: float = 1.0):
        super().__init__(stride, in_ch)
        embed = int(in_ch * ss_exp_factor)
        self.wqk = nn.Parameter(torch.zeros(size=(in_ch, embed)))
        nn.init.xavier_uniform_(self.wqk)
        self.stride = stride
        self.f_scale = f_scale

    def _pixel_unshuffle(self, x: torch.Tensor, c: int, f: int) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b 1 (c h) w")
        x = F.pixel_unshuffle(x, f)
        return rearrange(x, "b k (c h) w -> b (k c) h w", c=c)

    def _pixel_shuffle(self, x: torch.Tensor, c: int, f: int) -> torch.Tensor:
        x = rearrange(x, "b (f c) h w -> b f (c h) w", f=f * f, c=c)
        x = F.pixel_shuffle(x, f)
        return rearrange(x, "b f (c h) w -> b (f c) h w", f=1, c=c)

    def _pad_for_shuffle(self, x: torch.Tensor, f: int) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (f - h % f) % f
        pad_w = (f - w % f) % f
        if pad_h:
            x = F.pad(x, (0, 0, 0, pad_h))
        if pad_w:
            x = F.pad(x, (0, pad_w, 0, 0))
        return x, pad_h, pad_w

    def _get_attention(self, x: torch.Tensor, f: int) -> torch.Tensor:
        _, c, _, _ = x.shape
        xx = F.layer_norm(x, x.shape[-3:])
        xx, ph, pw = self._pad_for_shuffle(xx, f)
        xx = self._pixel_unshuffle(xx, c, f)
        xx = rearrange(xx, "b (f c) h w -> (b f) c h w", c=c, f=f * f)

        v, _, _ = self._pad_for_shuffle(x, f)
        v = self._pixel_unshuffle(v, c, f)
        v = rearrange(v, "b (f c) h w -> (b f) c h w", c=c, f=f * f)

        bsz, _, sh, sw = xx.shape
        qk = rearrange(xx, "b c h w -> (b h w) c")
        v = rearrange(v, "b c h w -> (b h w) c")
        qk = torch.mm(qk, self.wqk)
        qk = rearrange(qk, "(b h w) e -> b (h w) e", b=bsz, h=sh, w=sw)
        v = rearrange(v, "(b h w) e -> b (h w) e", b=bsz, h=sh, w=sw)

        qk = F.normalize(qk, dim=-1)
        attn = torch.bmm(qk, qk.transpose(1, 2))
        attn /= self.wqk.shape[1] ** 0.5
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = rearrange(out, "b (h w) e -> b e h w", b=bsz, h=sh, w=sw)
        out = rearrange(out, "(b f) c h w -> b (f c) h w", f=f * f, c=c)
        out = self._pixel_shuffle(out, c, f)
        if ph:
            out = out[:, :, :-ph, :]
        if pw:
            out = out[:, :, :, :-pw]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.f_scale * self.stride
        return x + self._forward(x + self._get_attention(x, f))


class DC_branchl(nn.Module):
    def __init__(self, stride: int, in_ch: int, num_module: int, mode: list[str] | str,
                 f_scale: int = 2, ss_exp_factor: float = 1.0):
        super().__init__()
        ly = [
            CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        if isinstance(mode, str):
            mode = [mode] * num_module
        assert len(mode) == num_module
        for bmode in mode:
            if bmode == "na":
                ly.append(SSBlockNaive(stride, in_ch))
            elif bmode == "ss":
                ly.append(SSBlock(stride, in_ch, f_scale=f_scale, ss_exp_factor=ss_exp_factor))
            else:
                raise ValueError(f"Invalid mode: {bmode}")
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SSBSNl(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9,
                 mode: list[str] | str = "ss", f_scale: int = 2, ss_exp_factor: float = 1.0):
        super().__init__()
        assert base_ch % 2 == 0
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.branch1 = DC_branchl(2, base_ch, num_module, mode, f_scale=f_scale, ss_exp_factor=ss_exp_factor)
        self.branch2 = DC_branchl(3, base_ch, num_module, mode, f_scale=f_scale, ss_exp_factor=ss_exp_factor)
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
        return self.tail(torch.cat([br1, br2], dim=1))


class SSBSN(nn.Module):
    def __init__(self, params: Dict[str, float]):
        super().__init__()
        self.pd_a = params["pd_a"]
        self.pd_b = params["pd_b"]
        self.pd_pad = params["pd_pad"]
        self.R3 = params["R3"]
        self.R3_T = params["R3_T"]
        self.R3_p = params["R3_p"]
        self.bsn = SSBSNl(
            in_ch=params["in_ch"],
            out_ch=params["in_ch"],
            base_ch=params["bsn_base_ch"],
            num_module=params["bsn_num_module"],
            mode=params["mode"],
            f_scale=params["f_scale"],
            ss_exp_factor=params["ss_exp_factor"],
        )

    def forward(self, img: torch.Tensor, pd: Optional[int] = None) -> torch.Tensor:
        pd = self.pd_a if pd is None else pd
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            pd_img = F.pad(img, (self.pd_pad, self.pd_pad, self.pd_pad, self.pd_pad))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            return pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        return pd_img_denoised[:, :, self.pd_pad:-self.pd_pad, self.pd_pad:-self.pd_pad]

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode="constant", value=0.0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode="constant", value=0.0)
        img_pd_bsn = self.forward(x, pd=self.pd_b)
        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]
        denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = (torch.rand_like(x) < self.R3_p)
            tmp_input = torch.clone(img_pd_bsn).detach()
            tmp_input[mask] = x[mask]
            tmp_input = F.pad(tmp_input, (self.pd_pad, self.pd_pad, self.pd_pad, self.pd_pad), mode="reflect")
            refined = self.bsn(tmp_input)
            denoised[..., t] = refined[:, :, self.pd_pad:-self.pd_pad, self.pd_pad:-self.pd_pad]
        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray


class SSBSNDenoiser:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None, disable_r3: bool = False):
        params = dict(SSBSN_PARAMS)
        if disable_r3:
            params["R3"] = False
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SSBSN(params).to(self.device)
        checkpoint = torch.load(weight_path or DEFAULT_WEIGHT, map_location=self.device)
        state = checkpoint.get("model_weight", checkpoint)
        if isinstance(state, dict) and "denoiser" in state:
            state = state["denoiser"]
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            missing = ", ".join(incompatible.missing_keys)
            raise RuntimeError(f"Missing keys in SS-BSN checkpoint: {missing}")
        self.model.eval()

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def denoise_image(self, image: Image.Image) -> DenoiseResult:
        tensor = self._prepare(image)
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            output = self.model.denoise(tensor)
        output = torch.clamp(output, 0.0, PIXEL_MAX)
        array = output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
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
    parser = argparse.ArgumentParser(description="Run SS-BSN denoising on a single sRGB image.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Where to save the denoised image")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default=None, help="Device spec such as 'cuda:0' or 'cpu'")
    parser.add_argument("--reference", default=None, help="Optional clean reference for PSNR")
    parser.add_argument("--disable-r3", action="store_true", help="Disable Random Replacing Refinement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weight_path = Path(args.weights).expanduser().resolve() if args.weights else DEFAULT_WEIGHT
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None

    denoiser = SSBSNDenoiser(weight_path=str(weight_path), device=args.device, disable_r3=args.disable_r3)
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
