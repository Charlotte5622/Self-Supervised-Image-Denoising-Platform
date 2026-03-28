#!/usr/bin/env python3
"""Standalone MM-BSN inference script with embedded network definitions."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT_DIR = Path(os.environ.get("MMBSN_WEIGHTS", THIS_DIR / "pretrained_models"))
DEFAULT_WEIGHT_NAME = "MMBSN_SIDD_o_a45.pth"

MMBSN_PARAMS = dict(
    pd_a=5,
    pd_b=2,
    pd_pad=2,
    R3=True,
    R3_T=8,
    R3_p=0.16,
    in_ch=3,
    bsn_base_ch=128,
    DCL1_num=2,
    DCL2_num=7,
    mask_type="o_a45",
)

PIXEL_MIN = 0.0
PIXEL_MAX = 255.0


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

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH//2] = 0
        # if kH == 5:
        #     self.mask[:, :, 1:-1, 1:-1] = 0
        # else:
        #     pass
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
            self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d


class MMBSN(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, DCL1_num=2, DCL2_num=7, mask_type='o_fsz'):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly0 = []
        ly0 += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        self.mask_types = mask_type.split('_')
        mask_number = len(self.mask_types)
        DCL_number1 = DCL1_num
        DCL_number2 = DCL2_num

        if 'o' in self.mask_types:
            self.branch1_1 = DC_branchl(2, base_ch, 'central', DCL_number1)
            self.branch1_2 = DC_branchl(3, base_ch, 'central', DCL_number1)
        if 'c' in self.mask_types:
            self.branch2_1 = DC_branchl(2, base_ch, 'col', DCL_number1)
            self.branch2_2 = DC_branchl(3, base_ch, 'col', DCL_number1)
        if 'r' in self.mask_types:
            self.branch3_1 = DC_branchl(2, base_ch, 'row', DCL_number1)
            self.branch3_2 = DC_branchl(3, base_ch, 'row', DCL_number1)
        if 'sz' in self.mask_types:
            self.branch4_1 = DC_branchl(2, base_ch, 'sz', DCL_number1)
            self.branch4_2 = DC_branchl(3, base_ch, 'sz', DCL_number1)
        if 'fsz' in self.mask_types:
            self.branch5_1 = DC_branchl(2, base_ch, 'fsz', DCL_number1)
            self.branch5_2 = DC_branchl(3, base_ch, 'fsz', DCL_number1)
        if 'a45' in self.mask_types:
            self.branch6_1 = DC_branchl(2, base_ch, 'a45', DCL_number1)
            self.branch6_2 = DC_branchl(3, base_ch, 'a45', DCL_number1)
        if 'a135' in self.mask_types:
            self.branch7_1 = DC_branchl(2, base_ch, 'a135', DCL_number1)
            self.branch7_2 = DC_branchl(3, base_ch, 'a135', DCL_number1)
        if 'cha' in self.mask_types:
            self.branch9_1 = DC_branchl(2, base_ch, 'cha', DCL_number1)
            self.branch9_2 = DC_branchl(3, base_ch, 'cha', DCL_number1)
        if 'fcha' in self.mask_types:
            self.branch10_1 = DC_branchl(2, base_ch, 'fcha', DCL_number1)
            self.branch10_2 = DC_branchl(3, base_ch, 'fcha', DCL_number1)
        if 'hui' in self.mask_types:
            self.branch8_1 = DC_branchl(2, base_ch, 'hui', DCL_number1)
            self.branch8_2 = DC_branchl(3, base_ch, 'hui', DCL_number1)

        ly_c = []
        ly_c += [nn.Conv2d(base_ch*mask_number, base_ch, kernel_size=1)]
        ly_c += [nn.ReLU(inplace=True)]
        self.conv2_1 = nn.Sequential(*ly_c)
        self.conv2_2 = nn.Sequential(*ly_c)

        self.dc_branchl2_mask3 = DC_branchl2(2, base_ch, DCL_number2)
        self.dc_branchl2_mask5 = DC_branchl2(3, base_ch, DCL_number2)


        ly = []
        ly += [ nn.Conv2d(base_ch*(2+2*mask_number),  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)
    def forward(self, x):
        mask_types = self.mask_types

        x = self.head(x)
        y1 = []
        y2 = []
        e = []
        if 'o' in mask_types:
            e1_1, br1_1 = self.branch1_1(x)
            e1_2, br1_2 = self.branch1_2(x)
            y1.append(br1_1)
            y2.append(br1_2)
            e.append(e1_1)
            e.append(e1_2)
        if 'c' in mask_types:
            e2_1, br2_1 = self.branch2_1(x)
            e2_2, br2_2 = self.branch2_2(x)
            y1.append(br2_1)
            y2.append(br2_2)
            e.append(e2_1)
            e.append(e2_2)
        if 'r' in mask_types:
            e3_1, br3_1 = self.branch3_1(x)
            e3_2, br3_2 = self.branch3_2(x)
            y1.append(br3_1)
            y2.append(br3_2)
            e.append(e3_1)
            e.append(e3_2)
        if 'sz' in mask_types:
            e4_1, br4_1 = self.branch4_1(x)
            e4_2, br4_2 = self.branch4_2(x)
            y1.append(br4_1)
            y2.append(br4_2)
            e.append(e4_1)
            e.append(e4_2)
        if 'fsz' in mask_types:
            e5_1, br5_1 = self.branch5_1(x)
            e5_2, br5_2 = self.branch5_2(x)
            y1.append(br5_1)
            y2.append(br5_2)
            e.append(e5_1)
            e.append(e5_2)
        if 'a45' in mask_types:
            e6_1, br6_1 = self.branch6_1(x)
            e6_2, br6_2 = self.branch6_2(x)
            y1.append(br6_1)
            y2.append(br6_2)
            e.append(e6_1)
            e.append(e6_2)
        if 'a135' in mask_types:
            e7_1, br7_1 = self.branch7_1(x)
            e7_2, br7_2 = self.branch7_2(x)
            y1.append(br7_1)
            y2.append(br7_2)
            e.append(e7_1)
            e.append(e7_2)
        if 'cha' in mask_types:
            e9_1, br9_1 = self.branch9_1(x)
            e9_2, br9_2 = self.branch9_2(x)
            y1.append(br9_1)
            y2.append(br9_2)
            e.append(e9_1)
            e.append(e9_2)
        if 'fcha' in mask_types:
            e10_1, br10_1 = self.branch10_1(x)
            e10_2, br10_2 = self.branch10_2(x)
            y1.append(br10_1)
            y2.append(br10_2)
            e.append(e10_1)
            e.append(e10_2)
        if 'hui' in mask_types:
            e8_1, br8_1 = self.branch8_1(x)
            e8_2, br8_2 = self.branch8_2(x)
            y1.append(br8_1)
            y2.append(br8_2)
            e.append(e8_1)
            e.append(e8_2)

        cat1 = torch.cat(y1, dim=1)
        conv2_1 = self.conv2_1(cat1)
        dc_branchl2_m3 = self.dc_branchl2_mask3(conv2_1)

        cat2 = torch.cat(y2, dim=1)
        conv2_2 = self.conv2_2(cat2)
        dc_branchl2_m5 = self.dc_branchl2_mask5(conv2_2)

        e.append(dc_branchl2_m3)
        e.append(dc_branchl2_m5)

        cat3 = torch.cat(e, dim=1)
        
        return self.tail(cat3)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, mask_type, num_module):
        super().__init__()

        ly0 = []
        ly1_1 = []
        ly1_2 = []
        ly2 = []

        if mask_type == 'r':
            ly0 += [ RowMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        elif mask_type == 'c':
            ly0 += [ ColMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'sz':
            ly0 += [ SzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fsz':
            ly0 += [ fSzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a45':
            ly0 += [ angle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a135':
            ly0 += [ angle135MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'hui':
            ly0 += [huiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'cha':
            ly0 += [chaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fcha':
            ly0 += [fchaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        else:
            ly0 += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        ly1_1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly1_1 += [nn.ReLU(inplace=True)]
        self.conv1_1 = nn.Sequential(*ly1_1)
        self.conv1_2 = nn.Sequential(*ly1_1)

        ly2 += [ DCl(stride, in_ch) for _ in range(num_module) ]
        ly2 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly2 += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly2)

        ly1_2 += [nn.Conv2d(in_ch*2, in_ch, kernel_size=1)]
        ly1_2 += [nn.ReLU(inplace=True)]
        self.conv1_3 = nn.Sequential(*ly1_2)
        # self.conv1_4 = nn.Sequential(*ly1_2)

    def forward(self, x):
        # 盲加两个卷积
        y0 = self.head(x)
        # 一个1*1卷积
        conv1_1 = self.conv1_1(y0)
        # 经过若干DCL
        y1 = self.body(conv1_1)
        # 融合
        cat0 = torch.cat([conv1_1, y1], dim=1)

        # 1*1 卷积
        conv1_3 = self.conv1_3(cat0)

        conv1_2 = self.conv1_2(y0)

        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class MMBSNBlindSpot(nn.Module):
    def __init__(self, params: Dict[str, float]):
        super().__init__()
        self.pd_a = params['pd_a']
        self.pd_b = params['pd_b']
        self.pd_pad = params['pd_pad']
        self.R3 = params['R3']
        self.R3_T = params['R3_T']
        self.R3_p = params['R3_p']
        self.bsn = MMBSN(
            in_ch=params['in_ch'],
            out_ch=params['in_ch'],
            base_ch=params['bsn_base_ch'],
            DCL1_num=params['DCL1_num'],
            DCL2_num=params['DCL2_num'],
            mask_type=params['mask_type'],
        )

    def forward(self, img: torch.Tensor, pd: Optional[int] = None) -> torch.Tensor:
        pd = self.pd_a if pd is None else pd
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            pad = self.pd_pad
            pd_img = F.pad(img, (pad, pad, pad, pad))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            return pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        pad = self.pd_pad
        return pd_img_denoised[:, :, pad:-pad, pad:-pad]

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
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
            pad = self.pd_pad
            tmp_input = F.pad(tmp_input, (pad, pad, pad, pad), mode='reflect')
            if self.pd_pad == 0:
                denoised[..., t] = self.bsn(tmp_input)
            else:
                denoised[..., t] = self.bsn(tmp_input)[:, :, pad:-pad, pad:-pad]
        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


def _resolve_weight_path(weight_path: Optional[str]) -> Path:
    candidates = []
    if weight_path:
        candidates.append(Path(weight_path).expanduser())
    env_path = os.environ.get('MMBSN_WEIGHT_PATH')
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_WEIGHT_DIR / DEFAULT_WEIGHT_NAME)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError('Could not find MM-BSN weights in expected locations')


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if 'model_weight' in checkpoint and isinstance(checkpoint['model_weight'], dict):
        nested = checkpoint['model_weight']
        if 'denoiser' in nested and isinstance(nested['denoiser'], dict):
            return nested['denoiser']
        return nested
    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        nested = checkpoint['state_dict']
        if 'denoiser' in nested and isinstance(nested['denoiser'], dict):
            return nested['denoiser']
        return nested
    return checkpoint


@dataclass(frozen=True)
class MMBSNDenoiseResult:
    denoised: np.ndarray


class MMBSNDenoiser:
    def __init__(self, weight_path: Optional[str] = None, device: Optional[str] = None, disable_r3: bool = False):
        params = dict(MMBSN_PARAMS)
        if disable_r3:
            params['R3'] = False
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MMBSNBlindSpot(params).to(self.device)
        checkpoint_path = _resolve_weight_path(weight_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert('RGB'), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def denoise_image(self, image: Image.Image) -> MMBSNDenoiseResult:
        tensor = self._prepare_tensor(image)
        ctx = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
        with ctx():
            output = self.model.denoise(tensor)
        output = torch.clamp(output, PIXEL_MIN, PIXEL_MAX)
        array = output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return MMBSNDenoiseResult(denoised=array)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f'Input image does not exist: {path}')
    return Image.open(path).convert('RGB')


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode='RGB').save(path)


def _load_reference(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Reference image does not exist: {path}')
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    if pred.shape != ref.shape:
        raise ValueError(f'Predicted image shape {pred.shape} does not match reference {ref.shape}')
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MM-BSN denoising on a single sRGB image.')
    parser.add_argument('--input', required=True, help='Path to the noisy sRGB image')
    parser.add_argument('--output', required=True, help='Path to save the denoised image')
    parser.add_argument('--weights', default=None, help='Optional custom checkpoint path')
    parser.add_argument('--device', default=None, help="Optional device string, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument('--reference', default=None, help='Optional clean reference image to compute PSNR')
    parser.add_argument('--disable-r3', action='store_true', help='Disable Random Replacing Refinement')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None
    denoiser = MMBSNDenoiser(weight_path=args.weights, device=args.device, disable_r3=args.disable_r3)
    noisy = _load_image(input_path)
    result = denoiser.denoise_image(noisy)
    _save_image(result.denoised, output_path)
    if reference_path is not None:
        reference = _load_reference(reference_path)
        score = _psnr(result.denoised, reference)
        print(f'PSNR against {reference_path.name}: {score:.4f} dB')


if __name__ == '__main__':
    main()
