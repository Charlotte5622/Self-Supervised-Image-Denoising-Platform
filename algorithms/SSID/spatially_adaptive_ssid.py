#!/usr/bin/env python3
"""Standalone Spatially Adaptive SSID inference entry point.

This file embeds the Spatially Adaptive SSID network definitions (BNN, LAN,
UNet) together with a convenience wrapper so that a single noisy sRGB image can
be denoised without depending on the original project codebase. All
hyper-parameters are hard-coded to mirror ``option/three_stage.json`` from the
paper's release.
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
# Basic paths & constants
###############################################################################
THIS_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("SPATIALLY_ADAPTIVE_SSID_WEIGHTS", THIS_DIR / "pretrained_models"))

BNN_BLINDSPOT = 9
LAN_BLINDSPOT = 3
STD_WINDOW = 7
ALPHA_LOWER = 1.0
ALPHA_UPPER = 5.0
PAD_MULTIPLE = 32
PIXEL_MIN = 0.0
PIXEL_MAX = 255.0

WEIGHTS = {
    "BNN": "BNN.pth",
    "LAN": "LAN.pth",
    "UNet": "UNet.pth",
}

###############################################################################
# Network definitions (copied from original repository)
###############################################################################


def rotate(x: torch.Tensor, angle: int) -> torch.Tensor:
    """Rotate BCHW tensor by multiples of 90 degrees clockwise."""
    h_dim, w_dim = 2, 3
    if angle == 0:
        return x
    if angle == 90:
        return x.flip(w_dim).transpose(h_dim, w_dim)
    if angle == 180:
        return x.flip(w_dim).flip(h_dim)
    if angle == 270:
        return x.flip(h_dim).transpose(h_dim, w_dim)
    raise ValueError("angle must be a multiple of 90")


class Crop2d(nn.Module):
    def __init__(self, crop: Tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.crop
        x0, x1 = left, x.shape[-1] - right
        y0, y1 = top, x.shape[-2] - bottom
        return x[:, :, y0:y1, x0:x1]


class Shift2d(nn.Module):
    def __init__(self, shift: Tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = shift
        y_a, y_b = abs(vert), 0
        x_a, x_b = abs(horz), 0
        if vert < 0:
            y_a, y_b = y_b, y_a
        if horz < 0:
            x_a, x_b = x_b, x_a
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shift_block(x)


class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x


class BNN(nn.Module):
    def __init__(self, blindspot: int, in_ch: int = 3, out_ch: int = 3, dim: int = 48):
        super().__init__()
        self.blindspot = blindspot
        in_channels = in_ch
        out_channels = out_ch

        self.encode_block_1 = nn.Sequential(
            ShiftConv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Shift2d((1, 0)),
            nn.MaxPool2d(2),
        )

        def _encode_block() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Shift2d((1, 0)),
                nn.MaxPool2d(2),
            )

        self.encode_block_2 = _encode_block()
        self.encode_block_3 = _encode_block()
        self.encode_block_4 = _encode_block()
        self.encode_block_5 = _encode_block()

        self.encode_block_6 = nn.Sequential(
            ShiftConv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        self.decode_block_5 = nn.Sequential(
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        def _decode_block() -> nn.Module:
            return nn.Sequential(
                ShiftConv2d(3 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.decode_block_4 = _decode_block()
        self.decode_block_3 = _decode_block()
        self.decode_block_2 = _decode_block()

        self.decode_block_1 = nn.Sequential(
            ShiftConv2d(2 * dim + in_channels, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(2 * dim, 2 * dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.shift = Shift2d(((self.blindspot + 1) // 2, 0))
        self.output_conv = ShiftConv2d(2 * dim, out_channels, 1)
        self.output_block = nn.Sequential(
            ShiftConv2d(8 * dim, 8 * dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ShiftConv2d(8 * dim, 2 * dim, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, a=0.1)
                    m.bias.data.zero_()
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x: torch.Tensor, shift: Optional[int] = None) -> torch.Tensor:
        if shift is not None:
            self.shift = Shift2d((shift, 0))
        else:
            self.shift = Shift2d(((self.blindspot + 1) // 2, 0))

        rotated = [rotate(x, rot) for rot in (0, 90, 180, 270)]
        x = torch.cat(rotated, dim=0)

        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        shifted = self.shift(x)
        rotated_batch = torch.chunk(shifted, 4, dim=0)
        aligned = [rotate(rotated, rot) for rotated, rot in zip(rotated_batch, (0, 270, 180, 90))]
        x = torch.cat(aligned, dim=1)
        x = self.output_block(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel: int = 64, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RB(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 1)
        self.cuca = CALayer(channel=filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0 = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.cuca(x)
        return out + c0


class NRB(nn.Module):
    def __init__(self, n: int, filters: int):
        super().__init__()
        nets = [RB(filters) for _ in range(n)]
        self.body = nn.Sequential(*nets)
        self.tail = nn.Conv2d(filters, filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tail(self.body(x))


class LAN(nn.Module):
    def __init__(self, blindspot: int, in_ch: int = 3, out_ch: Optional[int] = None, rbs: int = 6):
        super().__init__()
        assert blindspot % 2 == 1
        self.in_ch = in_ch
        self.out_ch = in_ch if out_ch is None else out_ch
        self.mid_ch = 64
        self.receptive_field = blindspot
        self.rbs = rbs

        layers = [nn.Conv2d(self.in_ch, self.mid_ch, 1), nn.ReLU()]
        for _ in range(self.receptive_field // 2):
            layers.extend([nn.Conv2d(self.mid_ch, self.mid_ch, 3, 1, 1), nn.ReLU()])
        layers.append(NRB(self.rbs, self.mid_ch))
        layers.append(nn.Conv2d(self.mid_ch, self.out_ch, 1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, zero_output: bool = False, dim: int = 48):
        super().__init__()
        self.zero_output = zero_output
        in_channels = in_ch
        out_channels = out_ch

        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        def _encode_block() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(2),
            )

        self.encode_block_2 = _encode_block()
        self.encode_block_3 = _encode_block()
        self.encode_block_4 = _encode_block()
        self.encode_block_5 = _encode_block()

        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        self.decode_block_5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        def _decode_block() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(dim * 3, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

        self.decode_block_4 = _decode_block()
        self.decode_block_3 = _decode_block()
        self.decode_block_2 = _decode_block()

        self.decode_block_1 = nn.Sequential(
            nn.Conv2d(dim * 2 + in_channels, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.output_conv = nn.Conv2d(dim * 2, out_channels, 1)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, a=0.1)
                    m.bias.data.zero_()
            if self.zero_output:
                self.output_conv.weight.zero_()
            else:
                nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool1 = self.encode_block_1(x)
        pool2 = self.encode_block_2(pool1)
        pool3 = self.encode_block_3(pool2)
        pool4 = self.encode_block_4(pool3)
        pool5 = self.encode_block_5(pool4)
        encoded = self.encode_block_6(pool5)

        upsample5 = self.decode_block_6(encoded)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.decode_block_4(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)
        x = self.output_conv(x)
        return x

###############################################################################
# Utility helpers
###############################################################################


def _resolve_weight_path(filename: str) -> Path:
    candidate = WEIGHTS_DIR / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find pretrained weight file `{filename}`. "
        "Place it inside `algorithm/pretrained_models` or set the "
        "SPATIALLY_ADAPTIVE_SSID_WEIGHTS environment variable to a folder "
        "that contains the weights."
    )


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _std_map(tensor: torch.Tensor, window_size: int = STD_WINDOW) -> torch.Tensor:
    assert window_size % 2 == 1, "Window size must be odd."
    pad = window_size // 2
    mean_rgb = tensor.mean(dim=1, keepdim=True)
    padded = F.pad(mean_rgb, (pad, pad, pad, pad), mode="reflect")
    unfolded = F.unfold(padded, kernel_size=window_size)
    unfolded = unfolded.view(tensor.shape[0], 1, window_size * window_size, tensor.shape[2], tensor.shape[3])
    centered = unfolded - unfolded.mean(dim=2, keepdim=True)
    variance = (centered * centered).mean(dim=2, keepdim=True)
    return torch.sqrt(variance).squeeze(2)


def _generate_alpha(bnn_output: torch.Tensor) -> torch.Tensor:
    std_map = _std_map(bnn_output)
    ratio = torch.full(
        (bnn_output.size(0), 1, std_map.size(-2), std_map.size(-1)),
        0.5,
        dtype=bnn_output.dtype,
        device=bnn_output.device,
    )
    ratio = torch.where(std_map < ALPHA_LOWER, torch.sigmoid(std_map - ALPHA_LOWER), ratio)
    ratio = torch.where(std_map > ALPHA_UPPER, torch.sigmoid(std_map - ALPHA_UPPER), ratio)
    return ratio.detach()


def _ensure_multiple(tensor: torch.Tensor, multiple: int = PAD_MULTIPLE) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)


def _to_numpy_image(tensor: torch.Tensor, crop_hw: Tuple[int, int]) -> np.ndarray:
    h, w = crop_hw
    tensor = tensor[..., :h, :w]
    tensor = torch.clamp(tensor, PIXEL_MIN, PIXEL_MAX)
    array = tensor.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    return array.astype(np.uint8)


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray
    stages: Dict[str, Optional[np.ndarray]]


class SpatiallyAdaptiveSSID:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bnn = BNN(blindspot=BNN_BLINDSPOT).to(self.device)
        self.lan = LAN(blindspot=LAN_BLINDSPOT).to(self.device)
        self.unet = UNet().to(self.device)
        self._load_pretrained_weights()
        for net in (self.bnn, self.lan, self.unet):
            net.eval()

    def _load_pretrained_weights(self) -> None:
        for name, module in ("BNN", self.bnn), ("LAN", self.lan), ("UNet", self.unet):
            weight_path = _resolve_weight_path(WEIGHTS[name])
            checkpoint = torch.load(weight_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    checkpoint = checkpoint["state_dict"]
                elif name in checkpoint and isinstance(checkpoint[name], dict):
                    checkpoint = checkpoint[name]
            state_dict = _clean_state_dict(checkpoint)
            module.load_state_dict(state_dict)

    def _prepare_tensor(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return _ensure_multiple(tensor, PAD_MULTIPLE)

    def _run_networks(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            bnn_out = self.bnn(tensor)
            lan_out = self.lan(tensor)
            unet_out = self.unet(tensor)
        return bnn_out, lan_out, unet_out

    def denoise(self, image: Image.Image) -> DenoiseResult:
        tensor, original_hw = self._prepare_tensor(image)
        bnn_out, lan_out, unet_out = self._run_networks(tensor)
        alpha = _generate_alpha(bnn_out)

        denoised = _to_numpy_image(unet_out, original_hw)
        stages = {
            "bnn": _to_numpy_image(bnn_out, original_hw),
            "lan": _to_numpy_image(lan_out, original_hw),
            "alpha": _to_numpy_image(alpha * PIXEL_MAX, original_hw) if alpha is not None else None,
        }
        return DenoiseResult(denoised=denoised, stages=stages)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    return Image.open(path).convert("RGB")


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
    elif array.shape[2] == 1:
        Image.fromarray(array.squeeze(-1), mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)


def _load_reference(reference_path: Path) -> np.ndarray:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {reference_path}")
    return np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    if pred.shape != ref.shape:
        raise ValueError(f"Predicted image shape {pred.shape} does not match reference {ref.shape}")
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise a single sRGB image with Spatially Adaptive SSID.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image.")
    parser.add_argument("--output", required=True, help="Path to save the denoised image.")
    parser.add_argument("--reference", default=None, help="Optional clean reference image to compute PSNR.")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save BNN, LAN, and alpha visualizations next to the output image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None

    denoiser = SpatiallyAdaptiveSSID()
    image = _load_image(input_path)
    result = denoiser.denoise(image)
    _save_image(result.denoised, output_path)

    if args.save_intermediate:
        for name, array in result.stages.items():
            if array is None:
                continue
            suffix = f"_{name}.png"
            _save_image(array, output_path.with_name(output_path.stem + suffix))

    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
