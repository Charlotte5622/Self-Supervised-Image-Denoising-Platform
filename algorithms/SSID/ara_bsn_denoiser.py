#!/usr/bin/env python3
"""Standalone ARA-BSN inference script with embedded network definitions."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT = THIS_DIR / "pretrained_models" / "ARA_BSN_SIDD.pth"

ARA_BSN_PARAMS = dict(
    pd_a=5,
    pd_b=2,
    pd_pad=2,
    R3=True,
    R3_T=8,
    R3_p=0.16,
    bsn="MMBSN",
    in_ch=3,
    bsn_base_ch=128,
    bsn_num_module=9,
    mm_mask_type="o_a45",
    mm_dcl1=2,
    mm_dcl2=7,
)

PIXEL_MIN = 0.0
PIXEL_MAX = 255.0

operation_seed_counter = 0


def _get_generator(device: torch.device) -> torch.Generator:
    global operation_seed_counter
    operation_seed_counter += 1
    gen = torch.Generator(device="cuda" if device.type == "cuda" else "cpu")
    gen.manual_seed(operation_seed_counter)
    return gen


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.0) -> torch.Tensor:
    if len(x.shape) == 3:
        c, h, w = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, h // f + 2 * pad, w // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(
            c, h + 2 * f * pad, w + 2 * f * pad
        )
    b, c, h, w = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(b, c, f, f, h // f + 2 * pad, w // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(
        b, c, h + 2 * f * pad, w + 2 * f * pad
    )


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0) -> torch.Tensor:
    if len(x.shape) == 3:
        c, h, w = x.shape
        pre_shuffle = x.view(c, f, h // f, f, w // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, h // f, w // f)
        if pad:
            pre_shuffle = pre_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(pre_shuffle, f)
    b, c, h, w = x.shape
    pre_shuffle = x.view(b, c, f, h // f, f, w // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, h // f, w // f)
    if pad:
        pre_shuffle = pre_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(pre_shuffle, f)


def random_transform(x: torch.Tensor, pd_factor: int):
    """
    Apply random transformations (rotation, flipping) to each sub-image after PD
    """
    b, c, h, w = x.shape
    assert h % pd_factor == 0, f"dim[-2] of input x {h} cannot be divided by {pd_factor}"
    assert w % pd_factor == 0, f"dim[-1] of input x {w} cannot be divided by {pd_factor}"

    sub_h = h // pd_factor
    sub_w = w // pd_factor
    transformed_x = x.clone()
    transform_ops = []

    for i in range(pd_factor):
        for j in range(pd_factor):
            rot_times = np.random.randint(0, 4)
            flip_type = np.random.randint(0, 5)
            transform_ops.append((rot_times, flip_type))

            start_h = i * sub_h
            start_w = j * sub_w
            sub_img = transformed_x[..., start_h:start_h + sub_h, start_w:start_w + sub_w]

            if flip_type == 1:
                sub_img = sub_img.flip(-1)
            elif flip_type == 2:
                sub_img = sub_img.flip(-2)
            elif flip_type == 3:
                sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-1)
            elif flip_type == 4:
                sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-2)

            if rot_times > 0:
                sub_img = torch.rot90(sub_img, rot_times, [-2, -1])

            transformed_x[..., start_h:start_h + sub_h, start_w:start_w + sub_w] = sub_img

    return transformed_x, transform_ops


def inverse_transform(x: torch.Tensor, transform_ops: list, pd_factor: int):
    """
    Restore transformations applied by random_transform
    """
    b, c, h, w = x.shape
    assert h % pd_factor == 0, f"dim[-2] of input x {h} cannot be divided by {pd_factor}"
    assert w % pd_factor == 0, f"dim[-1] of input x {w} cannot be divided by {pd_factor}"

    sub_h = h // pd_factor
    sub_w = w // pd_factor
    restored_x = x.clone()

    for idx, (rot_times, flip_type) in enumerate(transform_ops):
        i = idx // pd_factor
        j = idx % pd_factor
        start_h = i * sub_h
        start_w = j * sub_w
        sub_img = restored_x[..., start_h:start_h + sub_h, start_w:start_w + sub_w]

        if rot_times > 0:
            sub_img = torch.rot90(sub_img, -rot_times, [-2, -1])

        if flip_type == 1:
            sub_img = sub_img.flip(-1)
        elif flip_type == 2:
            sub_img = sub_img.flip(-2)
        elif flip_type == 3:
            sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-1)
        elif flip_type == 4:
            sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-2)

        restored_x[..., start_h:start_h + sub_h, start_w:start_w + sub_w] = sub_img

    return restored_x


def pd_down(x: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = x.shape
    x_down = F.pixel_unshuffle(x, pd_factor)
    out = x_down.view(b, c, pd_factor, pd_factor, h // pd_factor, w // pd_factor)
    out = out.permute(0, 2, 3, 1, 4, 5).reshape(b * pd_factor * pd_factor, c, h // pd_factor, w // pd_factor)
    return out


def pd_up(out: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = out.shape
    x_down = out.view(b // (pd_factor ** 2), pd_factor, pd_factor, c, h, w)
    x_down = x_down.permute(0, 3, 1, 2, 4, 5).reshape(b // (pd_factor ** 2), c * pd_factor * pd_factor, h, w)
    x_up = F.pixel_shuffle(x_down, pd_factor)
    return x_up


def random_shuffle_2x2(x: torch.Tensor):
    """Random shuffle with 2x2 windows."""
    b, c, h, w = x.shape
    window_size = 2
    assert h % window_size == 0 and w % window_size == 0, f"Image size must be multiple of {window_size}"
    x_reshaped = x.view(b * c, 1, h, w)
    patches = F.unfold(x_reshaped, kernel_size=window_size, stride=window_size)
    patches = patches.transpose(1, 2).reshape(-1, window_size * window_size)
    idx = torch.rand(patches.shape[0], window_size * window_size, device=x.device).argsort(dim=1)
    shuffled_patches = torch.gather(patches, 1, idx)
    shuffled_patches = shuffled_patches.view(b * c, -1, window_size * window_size).transpose(1, 2)
    shuffled = F.fold(
        shuffled_patches,
        output_size=(h, w),
        kernel_size=window_size,
        stride=window_size,
    )
    return shuffled.view(b, c, h, w)


def adaptive_shuffle_by_std(x: torch.Tensor, std_threshold=5, denoised=None, visualize=False):
    """Adaptively shuffle using 3x3 windows following official implementation."""
    b, c, h, w = x.shape
    window_size = 3
    stride = 3
    orig_h, orig_w = h, w
    h_remainder = h % window_size
    w_remainder = w % window_size
    h_trunc = None
    w_trunc = None
    if h_remainder > 0:
        h_trunc = x[:, :, -h_remainder:, :]
        x = x[:, :, :-h_remainder, :]
        if denoised is not None:
            denoised = denoised[:, :, :-h_remainder, :]
    if w_remainder > 0:
        w_trunc = x[:, :, :, -w_remainder:]
        x = x[:, :, :, :-w_remainder]
        if denoised is not None:
            denoised = denoised[:, :, :, :-w_remainder]
    b, c, h, w = x.shape
    denoised_patches = F.unfold(denoised, kernel_size=window_size, stride=stride)
    n_patches = denoised_patches.size(-1)
    denoised_patches = denoised_patches.transpose(1, 2).reshape(b, n_patches, c, window_size * window_size)
    std_values = torch.std(denoised_patches, dim=-1).mean(dim=-1)
    patches = F.unfold(x, kernel_size=window_size, stride=stride).transpose(1, 2)
    patches = patches.reshape(b, n_patches, c, window_size * window_size)
    shuffle_mask = (std_values <= std_threshold)
    rand_indices = torch.rand(b, n_patches, window_size * window_size, device=x.device).argsort(dim=-1)
    base_indices = torch.arange(window_size * window_size, device=x.device).view(1, 1, -1).expand(b, n_patches, -1)
    final_indices = torch.where(shuffle_mask.unsqueeze(-1), rand_indices, base_indices)
    shuffled_patches = torch.zeros_like(patches)
    for i in range(b):
        for j in range(n_patches):
            idx = final_indices[i, j]
            shuffled_patches[i, j] = patches[i, j, :, idx]
    shuffled_patches = shuffled_patches.reshape(b, n_patches, c * window_size * window_size).transpose(1, 2)
    output = F.fold(
        shuffled_patches,
        output_size=(h, w),
        kernel_size=window_size,
        stride=stride,
    )
    if h_remainder > 0 or w_remainder > 0:
        final_output = torch.zeros((b, c, orig_h, orig_w), device=x.device)
        final_output[:, :, :h, :w] = output
        if h_trunc is not None:
            final_output[:, :, -h_remainder:, :w] = h_trunc[:, :, :, :w]
        if w_trunc is not None:
            if h_trunc is not None:
                corner = x[:, :, -h_remainder:, -w_remainder:]
                final_output[:, :, -h_remainder:, -w_remainder:] = corner
            final_output[:, :, :h, -w_remainder:] = w_trunc
        output = final_output
    return output


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img: torch.Tensor, width: int = 4, mask_type: str = "random") -> torch.Tensor:
    n, c, h, w = img.shape
    mask = torch.zeros((n * h // width * w // width * width ** 2,), dtype=torch.int64, device=img.device)
    idx_list = torch.arange(0, width ** 2, device=img.device)
    rd_idx = torch.zeros((n * h // width * w // width,), dtype=torch.int64, device=img.device)
    if mask_type == "random":
        torch.randint(0, len(idx_list), rd_idx.shape, generator=_get_generator(img.device), device=img.device, out=rd_idx)
    elif mask_type == "batch":
        base = torch.randint(0, len(idx_list), (n,), generator=_get_generator(img.device), device=img.device)
        rd_idx = base.repeat(h // width * w // width)
    elif mask_type == "all":
        base = torch.randint(0, len(idx_list), (1,), generator=_get_generator(img.device), device=img.device)
        rd_idx = base.repeat(n * h // width * w // width)
    elif mask_type.startswith("fix_"):
        index = int(mask_type.split("_")[-1])
        rd_idx.fill_(index)
    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(0, mask.numel(), width ** 2, device=img.device)
    mask[rd_pair_idx] = 1
    mask = depth_to_space(mask.type_as(img).view(n, h // width, w // width, width ** 2).permute(0, 3, 1, 2), width)
    return mask.type(torch.int64)


def interpolate_mask(tensor: torch.Tensor, mask: torch.Tensor, mask_inv: torch.Tensor) -> torch.Tensor:
    n, c, h, w = tensor.shape
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]], dtype=np.float32)
    kernel = torch.from_numpy(kernel).to(tensor.device)
    kernel = kernel / kernel.sum()
    filtered = torch.nn.functional.conv2d(
        tensor.view(n * c, 1, h, w), kernel.view(1, 1, 3, 3), stride=1, padding=1
    )
    return filtered.view_as(tensor) * mask + tensor * mask_inv


class Masker:
    def __init__(self, width: int = 4, mode: str = "interpolate", mask_type: str = "all"):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img: torch.Tensor, mask_type: Optional[str] = None, mode: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_type = mask_type or self.mask_type
        mode = mode or self.mode
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones_like(mask) - mask
        if mode != "interpolate":
            raise NotImplementedError
        net_input = interpolate_mask(img, mask, mask_inv)
        return net_input, mask

    def train(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width ** 2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width ** 2, 1, h, w), device=img.device)
        for i in range(self.width ** 2):
            x, mask = self.mask(img, mask_type=f"fix_{i}")
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


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


class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kW // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kW // 2] = 0
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kW // 2] = 1
        self.mask[:, :, kH // 2, :] = 1
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

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
    def __init__(self, stride: int, in_ch: int, mask_type: str, num_module: int):
        super().__init__()
        if mask_type == "r":
            mask_layer = RowMaskedConv2d
        elif mask_type == "c":
            mask_layer = ColMaskedConv2d
        elif mask_type == "sz":
            mask_layer = SzMaskedConv2d
        elif mask_type == "fsz":
            mask_layer = fSzMaskedConv2d
        elif mask_type == "a45":
            mask_layer = angle45MaskedConv2d
        elif mask_type == "a135":
            mask_layer = angle135MaskedConv2d
        elif mask_type == "cha":
            mask_layer = chaMaskedConv2d
        elif mask_type == "fcha":
            mask_layer = fchaMaskedConv2d
        elif mask_type == "hui":
            mask_layer = huiMaskedConv2d
        else:
            mask_layer = CentralMaskedConv2d
        self.head = nn.Sequential(
            mask_layer(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[DCl(stride, in_ch) for _ in range(num_module)], nn.Conv2d(in_ch, in_ch, 1), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_ch * 2, in_ch, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y0 = self.head(x)
        conv1_1 = self.conv1_1(y0)
        y1 = self.body(conv1_1)
        cat0 = torch.cat([conv1_1, y1], dim=1)
        conv1_3 = self.conv1_3(cat0)
        conv1_2 = self.conv1_2(y0)
        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    def __init__(self, stride: int, in_ch: int, num_module: int):
        super().__init__()
        self.body = nn.Sequential(*[DCl(stride, in_ch) for _ in range(num_module)], nn.Conv2d(in_ch, in_ch, 1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MMBSN(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, DCL1_num: int = 2, DCL2_num: int = 7, mask_type: str = "o_a45"):
        super().__init__()
        assert base_ch % 2 == 0
        self.mask_types = mask_type.split("_")
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        if "o" in self.mask_types:
            self.branch1_1 = DC_branchl(2, base_ch, "central", DCL1_num)
            self.branch1_2 = DC_branchl(3, base_ch, "central", DCL1_num)
        if "c" in self.mask_types:
            self.branch2_1 = DC_branchl(2, base_ch, "c", DCL1_num)
            self.branch2_2 = DC_branchl(3, base_ch, "c", DCL1_num)
        if "r" in self.mask_types:
            self.branch3_1 = DC_branchl(2, base_ch, "r", DCL1_num)
            self.branch3_2 = DC_branchl(3, base_ch, "r", DCL1_num)
        if "sz" in self.mask_types:
            self.branch4_1 = DC_branchl(2, base_ch, "sz", DCL1_num)
            self.branch4_2 = DC_branchl(3, base_ch, "sz", DCL1_num)
        if "fsz" in self.mask_types:
            self.branch5_1 = DC_branchl(2, base_ch, "fsz", DCL1_num)
            self.branch5_2 = DC_branchl(3, base_ch, "fsz", DCL1_num)
        if "a45" in self.mask_types:
            self.branch6_1 = DC_branchl(2, base_ch, "a45", DCL1_num)
            self.branch6_2 = DC_branchl(3, base_ch, "a45", DCL1_num)
        if "a135" in self.mask_types:
            self.branch7_1 = DC_branchl(2, base_ch, "a135", DCL1_num)
            self.branch7_2 = DC_branchl(3, base_ch, "a135", DCL1_num)
        if "hui" in self.mask_types:
            self.branch8_1 = DC_branchl(2, base_ch, "hui", DCL1_num)
            self.branch8_2 = DC_branchl(3, base_ch, "hui", DCL1_num)
        if "cha" in self.mask_types:
            self.branch9_1 = DC_branchl(2, base_ch, "cha", DCL1_num)
            self.branch9_2 = DC_branchl(3, base_ch, "cha", DCL1_num)
        if "fcha" in self.mask_types:
            self.branch10_1 = DC_branchl(2, base_ch, "fcha", DCL1_num)
            self.branch10_2 = DC_branchl(3, base_ch, "fcha", DCL1_num)
        mask_number = len(self.mask_types)
        self.conv2_1 = nn.Sequential(nn.Conv2d(base_ch * mask_number, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(base_ch * mask_number, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.dc_branchl2_mask3 = DC_branchl2(2, base_ch, DCL2_num)
        self.dc_branchl2_mask5 = DC_branchl2(3, base_ch, DCL2_num)
        tail_in_ch = base_ch * (2 + 2 * mask_number)
        self.tail = nn.Sequential(
            nn.Conv2d(tail_in_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, masked: bool = True) -> torch.Tensor:
        mask_types = self.mask_types
        x = self.head(x)
        y1 = []
        y2 = []
        e = []
        if "o" in mask_types:
            e1_1, br1_1 = self.branch1_1(x)
            e1_2, br1_2 = self.branch1_2(x)
            y1.append(br1_1)
            y2.append(br1_2)
            e.extend([e1_1, e1_2])
        if "c" in mask_types:
            e2_1, br2_1 = self.branch2_1(x)
            e2_2, br2_2 = self.branch2_2(x)
            y1.append(br2_1)
            y2.append(br2_2)
            e.extend([e2_1, e2_2])
        if "r" in mask_types:
            e3_1, br3_1 = self.branch3_1(x)
            e3_2, br3_2 = self.branch3_2(x)
            y1.append(br3_1)
            y2.append(br3_2)
            e.extend([e3_1, e3_2])
        if "sz" in mask_types:
            e4_1, br4_1 = self.branch4_1(x)
            e4_2, br4_2 = self.branch4_2(x)
            y1.append(br4_1)
            y2.append(br4_2)
            e.extend([e4_1, e4_2])
        if "fsz" in mask_types:
            e5_1, br5_1 = self.branch5_1(x)
            e5_2, br5_2 = self.branch5_2(x)
            y1.append(br5_1)
            y2.append(br5_2)
            e.extend([e5_1, e5_2])
        if "a45" in mask_types:
            e6_1, br6_1 = self.branch6_1(x)
            e6_2, br6_2 = self.branch6_2(x)
            y1.append(br6_1)
            y2.append(br6_2)
            e.extend([e6_1, e6_2])
        if "a135" in mask_types:
            e7_1, br7_1 = self.branch7_1(x)
            e7_2, br7_2 = self.branch7_2(x)
            y1.append(br7_1)
            y2.append(br7_2)
            e.extend([e7_1, e7_2])
        if "hui" in mask_types:
            e8_1, br8_1 = self.branch8_1(x)
            e8_2, br8_2 = self.branch8_2(x)
            y1.append(br8_1)
            y2.append(br8_2)
            e.extend([e8_1, e8_2])
        if "cha" in mask_types:
            e9_1, br9_1 = self.branch9_1(x)
            e9_2, br9_2 = self.branch9_2(x)
            y1.append(br9_1)
            y2.append(br9_2)
            e.extend([e9_1, e9_2])
        if "fcha" in mask_types:
            e10_1, br10_1 = self.branch10_1(x)
            e10_2, br10_2 = self.branch10_2(x)
            y1.append(br10_1)
            y2.append(br10_2)
            e.extend([e10_1, e10_2])
        if y1:
            cat1 = torch.cat(y1, dim=1)
            cat2 = torch.cat(y2, dim=1)
        else:
            cat1 = cat2 = torch.zeros_like(x)
        conv2_1 = self.conv2_1(cat1)
        conv2_2 = self.conv2_2(cat2)
        e.append(self.dc_branchl2_mask3(conv2_1))
        e.append(self.dc_branchl2_mask5(conv2_2))
        cat3 = torch.cat(e, dim=1)
        return self.tail(cat3)


class ARABSN(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.pd_a = params["pd_a"]
        self.pd_b = params["pd_b"]
        self.pd_pad = params["pd_pad"]
        self.R3 = params["R3"]
        self.R3_T = params["R3_T"]
        self.R3_p = params["R3_p"]
        bsn_type = params.get("bsn", "MMBSN").lower()
        if bsn_type != "mmbsn":
            raise ValueError("This standalone script currently supports MMBSN-based checkpoints only")
        self.bsn = MMBSN(
            in_ch=params["in_ch"],
            out_ch=params["in_ch"],
            base_ch=params["bsn_base_ch"],
            DCL1_num=params.get("mm_dcl1", 2),
            DCL2_num=params.get("mm_dcl2", 7),
            mask_type=params.get("mm_mask_type", "o_a45"),
        )

    def one_forward(self, img: torch.Tensor, pd: int) -> torch.Tensor:
        b, c, h, w = img.shape
        if pd > 1:
            pad_h = (pd - h % pd) % pd
            pad_w = (pd - w % pd) % pd
            if pad_h or pad_w:
                img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            pad_h = pad_w = 0
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
            transformed, ops = random_transform(pd_img, pd)
            denoised = self.bsn(transformed)
            denoised = inverse_transform(denoised, ops, pd)
            restored = pixel_shuffle_up_sampling(denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            padded = F.pad(img, (p, p, p, p))
            restored = self.bsn(padded)
            restored = restored[:, :, p:-p, p:-p]
        return restored[:, :, :h, :w]

    def forward(self, img: torch.Tensor, pd: Optional[int] = None) -> torch.Tensor:
        pd_factor = pd if pd is not None else self.pd_a
        return self.one_forward(img, pd_factor)

    def denoise(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.denoise_mpd(x, **kwargs)

    def denoise_mpd(
        self,
        x: torch.Tensor,
        low: float = 1.0,
        high: float = 5.0,
        window_size: int = 3,
        scale1: float = 1.0,
        scale2: float = 3.0,
        lpd: int = 4,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        img_pd1 = self.forward(x, pd=1)[:, :, :h, :w]
        img_pd2 = x
        if h % self.pd_b != 0 or w % self.pd_b != 0:
            img_pd2 = F.pad(img_pd2, (0, (self.pd_b - w % self.pd_b) % self.pd_b, 0, (self.pd_b - h % self.pd_b) % self.pd_b))
        img_pd2 = self.forward(img_pd2, pd=self.pd_b)[:, :, :h, :w]
        img_pd4 = x
        if h % lpd != 0 or w % lpd != 0:
            img_pd4 = F.pad(img_pd4, (0, (lpd - w % lpd) % lpd, 0, (lpd - h % lpd) % lpd))
        img_pd4 = self.forward(img_pd4, pd=lpd)[:, :, :h, :w]

        def local_std(img: torch.Tensor) -> torch.Tensor:
            pad = window_size // 2
            gray = img.mean(dim=1, keepdim=True)
            padded = F.pad(gray, (pad, pad, pad, pad), mode="reflect")
            patches = F.unfold(padded, kernel_size=window_size)
            patches = patches.view(b, 1, window_size * window_size, h, w)
            diff = patches - patches.mean(dim=2, keepdim=True)
            var = (diff * diff).mean(dim=2)
            return torch.sqrt(var + 1e-8)

        seita = local_std(img_pd2)
        mask_low = seita <= low
        mask_mid = (seita > low) & (seita <= high)
        mask_high = seita > high
        img_pd_bsn = torch.zeros_like(x)
        img_pd_bsn = torch.where(mask_low, ((scale1 - 1.0) * img_pd2 + img_pd4) / max(scale1, 1e-6), img_pd_bsn)
        img_pd_bsn = torch.where(mask_mid, img_pd2, img_pd_bsn)
        img_pd_bsn = torch.where(mask_high, (img_pd1 + (scale2 - 1.0) * img_pd2) / scale2, img_pd_bsn)

        if not self.R3:
            return img_pd_bsn
        denoised = torch.empty(*x.shape, self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = torch.rand_like(x) < self.R3_p
            tmp_input = img_pd_bsn.clone()
            tmp_input[mask] = x[mask]
            p = self.pd_pad
            tmp_input = F.pad(tmp_input, (p, p, p, p), mode="reflect")
            refined = self.bsn(tmp_input)[:, :, p:-p, p:-p]
            denoised[..., t] = refined
        return torch.mean(denoised, dim=-1)


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray


class ARABSNDenoiser:
    def __init__(
        self,
        weight_path: Optional[str] = None,
        device: Optional[str] = None,
        disable_r3: bool = False,
        add_constant: float = 0.5,
        floor_output: bool = True,
        low: float = 1.0,
        high: float = 5.0,
        window_size: int = 3,
        scale1: float = 1.0,
        scale2: float = 3.0,
        lpd: int = 4,
    ):
        params = dict(ARA_BSN_PARAMS)
        if disable_r3:
            params["R3"] = False
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_constant = float(add_constant)
        self.floor_output = floor_output
        self.low = float(low)
        self.high = float(high)
        self.window_size = int(window_size)
        self.scale1 = float(scale1)
        self.scale2 = float(scale2)
        self.lpd = int(lpd)
        self.model = ARABSN(params).to(self.device)
        checkpoint = torch.load(weight_path or DEFAULT_WEIGHT, map_location=self.device)
        state = checkpoint.get("model_weight", checkpoint)
        if isinstance(state, dict) and "denoiser" in state:
            state = state["denoiser"]
        self.model.load_state_dict(state)
        self.model.eval()

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def denoise_image(self, image: Image.Image) -> DenoiseResult:
        tensor = self._prepare(image)
        with torch.inference_mode():
            output = self.model.denoise(
                tensor,
                low=self.low,
                high=self.high,
                window_size=self.window_size,
                scale1=self.scale1,
                scale2=self.scale2,
                lpd=self.lpd,
            )
        if self.add_constant:
            output = output + self.add_constant
        if self.floor_output:
            output = torch.floor(output)
        output = torch.clamp(output, PIXEL_MIN, PIXEL_MAX)
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
        raise ValueError("Prediction and reference must have the same shape")
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARA-BSN denoising on a single sRGB image.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Where to save the denoised image")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default=None, help="Device spec such as 'cuda:0' or 'cpu'")
    parser.add_argument("--reference", default=None, help="Optional clean reference image to compute PSNR")
    parser.add_argument("--disable-r3", action="store_true", help="Disable Random Replacement Refinement")
    parser.add_argument("--add-constant", type=float, default=0.5, help="Constant added before quantization (default 0.5)")
    parser.add_argument("--disable-floor", dest="floor_output", action="store_false", help="Do not floor the output intensities")
    parser.set_defaults(floor_output=True)
    parser.add_argument("--low", type=float, default=1.0, help="Local std threshold for low-frequency region")
    parser.add_argument("--high", type=float, default=5.0, help="Local std threshold for high-frequency region")
    parser.add_argument("--window-size", type=int, default=3, help="Window size for local std computation")
    parser.add_argument("--scale1", type=float, default=1.0, help="Blending factor for low-frequency fusion")
    parser.add_argument("--scale2", type=float, default=3.0, help="Blending factor for high-frequency fusion")
    parser.add_argument("--lpd", type=int, default=4, help="Additional PD factor used for low-frequency branch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weight_path = Path(args.weights).expanduser().resolve() if args.weights else DEFAULT_WEIGHT
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    denoiser = ARABSNDenoiser(
        weight_path=str(weight_path),
        device=args.device,
        disable_r3=args.disable_r3,
        add_constant=args.add_constant,
        floor_output=args.floor_output,
        low=args.low,
        high=args.high,
        window_size=args.window_size,
        scale1=args.scale1,
        scale2=args.scale2,
        lpd=args.lpd,
    )
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f"PSNR against {reference_path.name}: {score:.4f} dB")


if __name__ == "__main__":
    main()
