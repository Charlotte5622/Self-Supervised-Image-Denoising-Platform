#!/usr/bin/env python3
"""Standalone Complementary-BSN (AP-BSN) inference script."""

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
DEFAULT_WEIGHT = THIS_DIR / "pretrained_models" / "Complementary_BSN_SIDD.pth"
PIXEL_MIN = 0.0
PIXEL_MAX = 255.0

APBSN_PARAMS = dict(
    pd_a=5,
    pd_b=2,
    pd_pad=2,
    R3=True,
    R3_T=12,
    R3_p=0.16,
    in_ch=3,
    bsn_base_ch=128,
    bsn_num_module=9,
)

operation_seed_counter = 0


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


def random_arrangement(x: torch.Tensor, pd_factor: int) -> Tuple[torch.Tensor, np.ndarray]:
    seq2random = np.random.permutation(range(pd_factor * pd_factor))
    random2seq = np.zeros_like(seq2random)
    for i, idx in enumerate(seq2random):
        random2seq[idx] = i
    random_x = torch.zeros_like(x)
    b, c, h, w = x.shape
    assert h % pd_factor == 0 and w % pd_factor == 0
    sub_h = h // pd_factor
    sub_w = w // pd_factor
    for i in range(pd_factor):
        for j in range(pd_factor):
            rand_idx = seq2random[i * pd_factor + j]
            rand_j = rand_idx % pd_factor
            rand_i = rand_idx // pd_factor
            random_x[:, :, rand_i * sub_h : (rand_i + 1) * sub_h, rand_j * sub_w : (rand_j + 1) * sub_w] = x[
                :, :, i * sub_h : (i + 1) * sub_h, j * sub_w : (j + 1) * sub_w
            ]
    return random_x, random2seq


def inverse_random_arrangement(random_x: torch.Tensor, random2seq: np.ndarray, pd_factor: int) -> torch.Tensor:
    x = torch.zeros_like(random_x)
    b, c, h, w = random_x.shape
    sub_h = h // pd_factor
    sub_w = w // pd_factor
    for rand_idx in range(pd_factor * pd_factor):
        idx = random2seq[rand_idx]
        j = idx % pd_factor
        i = idx // pd_factor
        rand_j = rand_idx % pd_factor
        rand_i = rand_idx // pd_factor
        x[:, :, i * sub_h : (i + 1) * sub_h, j * sub_w : (j + 1) * sub_w] = random_x[
            :, :, rand_i * sub_h : (rand_i + 1) * sub_h, rand_j * sub_w : (rand_j + 1) * sub_w
        ]
    return x


def get_generator(device: torch.device) -> torch.Generator:
    global operation_seed_counter
    operation_seed_counter += 1
    gen_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(operation_seed_counter)
    return generator


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img: torch.Tensor, width: int = 4, mask_type: str = "random") -> torch.Tensor:
    n, c, h, w = img.shape
    mask = torch.zeros(n * h // width * w // width * width ** 2, dtype=torch.int64, device=img.device)
    idx_list = torch.arange(0, width ** 2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(n * h // width * w // width, dtype=torch.int64, device=img.device)
    if mask_type == "random":
        torch.randint(0, len(idx_list), rd_idx.shape, generator=get_generator(img.device), device=img.device, out=rd_idx)
    elif mask_type == "batch":
        base = torch.randint(0, len(idx_list), (n,), generator=get_generator(img.device), device=img.device)
        rd_idx = base.repeat(h // width * w // width)
    elif mask_type == "all":
        base = torch.randint(0, len(idx_list), (1,), generator=get_generator(img.device), device=img.device)
        rd_idx = base.repeat(n * h // width * w // width)
    elif "fix" in mask_type:
        index = int(mask_type.split("_")[-1])
        rd_idx.fill_(index)
    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(0, mask.numel(), width ** 2, dtype=torch.int64, device=img.device)
    mask[rd_pair_idx] = 1
    mask = mask.type_as(img).view(n, h // width, w // width, width ** 2).permute(0, 3, 1, 2)
    mask = depth_to_space(mask, width).type(torch.int64)
    return mask


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

    def mask(self, img: torch.Tensor, mask_type: Optional[str] = None, mode: Optional[str] = None):
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
            net_input, mask = self.mask(img, mask_type=f"fix_{i}")
            tensors[:, i, ...] = net_input
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
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
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=stride, dilation=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DC_branchl(nn.Module):
    def __init__(self, stride: int, in_ch: int, num_module: int):
        super().__init__()
        layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        ]
        layers.extend(DCl(stride, in_ch) for _ in range(num_module))
        layers += [nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class mask_conv(nn.Module):
    def __init__(self, stride: int, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, padding=stride - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class nomask_conv(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DBSNl(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 128, num_module: int = 9):
        super().__init__()
        assert base_ch % 2 == 0
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.branch1_maskconv = mask_conv(2, base_ch)
        self.branch2_maskconv = mask_conv(3, base_ch)
        self.branch1_nomaskconv = nomask_conv(base_ch)
        self.branch2_nomaskconv = nomask_conv(base_ch)
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

    def forward(self, x: torch.Tensor, is_masked: bool) -> torch.Tensor:
        x = self.head(x)
        if is_masked:
            br1 = self.branch1_maskconv(x)
            br2 = self.branch2_maskconv(x)
        else:
            br1 = self.branch1_nomaskconv(x)
            br2 = self.branch2_nomaskconv(x)
        br1 = self.branch1(br1)
        br2 = self.branch2(br2)
        x = torch.cat([br1, br2], dim=1)
        return self.tail(x)


class APBSN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.pd_a = params["pd_a"]
        self.pd_b = params["pd_b"]
        self.pd_pad = params["pd_pad"]
        self.R3 = params["R3"]
        self.R3_T = params["R3_T"]
        self.R3_p = params["R3_p"]
        self.bsn = DBSNl(
            in_ch=params["in_ch"],
            out_ch=params["in_ch"],
            base_ch=params["bsn_base_ch"],
            num_module=params["bsn_num_module"],
        )
        self.conv1 = nn.Sequential(nn.Conv2d(12, 3, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=1), nn.ReLU(inplace=True))
        self.masker = Masker(width=4, mode='interpolate', mask_type='all')

    def forward_mpd(self, img: torch.Tensor, pd: int) -> torch.Tensor:
        b, c, h, w = img.shape
        if h % pd != 0:
            img = F.pad(img, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        if w % pd != 0:
            img = F.pad(img, (0, pd - w % pd, 0, 0), mode='constant', value=0)
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            pd_img = F.pad(img, (self.pd_pad, self.pd_pad, self.pd_pad, self.pd_pad))
        pd_img, random2seq = random_arrangement(pd_img, pd)
        pd_img_denoised = self.bsn(pd_img, is_masked=True)
        pd_img_denoised = inverse_random_arrangement(pd_img_denoised, random2seq, pd)
        if pd > 1:
            return pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        return pd_img_denoised[:, :, self.pd_pad:-self.pd_pad, self.pd_pad:-self.pd_pad]

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        img_pd2 = x
        if h % self.pd_b != 0:
            img_pd2 = F.pad(img_pd2, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            img_pd2 = F.pad(img_pd2, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        img_pd2 = self.forward_mpd(img_pd2, pd=2)

        img_pd5 = x
        if h % self.pd_a != 0:
            img_pd5 = F.pad(img_pd5, (0, 0, 0, self.pd_a - h % self.pd_a), mode='constant', value=0)
        if w % self.pd_a != 0:
            img_pd5 = F.pad(img_pd5, (0, self.pd_a - w % self.pd_a, 0, 0), mode='constant', value=0)
        img_pd5 = self.forward_mpd(img_pd5, pd=5)[:, :, :h, :w]

        img_pd1 = self.bsn(x, is_masked=True)
        img_pd = 0.7 * img_pd5 + 0.3 * img_pd1
        img_pd = 0.2 * img_pd + 0.8 * img_pd2

        if not self.R3:
            return img_pd2[:, :, :h, :w]
        denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
        for t in range(self.R3_T):
            mask = torch.rand_like(x) < self.R3_p
            tmp_input = torch.clone(img_pd).detach()
            tmp_input[mask] = x[mask]
            tmp_input = F.pad(tmp_input, (self.pd_pad, self.pd_pad, self.pd_pad, self.pd_pad), mode='reflect')
            refined = self.bsn(tmp_input, is_masked=True)
            denoised[..., t] = refined[:, :, self.pd_pad:-self.pd_pad, self.pd_pad:-self.pd_pad]
        return torch.mean(denoised, dim=-1)[:, :, :h, :w]


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray


class ComplementaryBSNDenoiser:
    def __init__(
        self,
        weight_path: Optional[str] = None,
        device: Optional[str] = None,
        disable_r3: bool = False,
        add_constant: float = 0.5,
        floor_output: bool = True,
    ):
        params = dict(APBSN_PARAMS)
        if disable_r3:
            params["R3"] = False
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_constant = float(add_constant)
        self.floor_output = floor_output
        self.model = APBSN(params).to(self.device)
        checkpoint = torch.load(weight_path or DEFAULT_WEIGHT, map_location=self.device)
        state = checkpoint.get("model_weight", checkpoint)
        if isinstance(state, dict) and "denoiser" in state:
            state = state["denoiser"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def denoise_image(self, image: Image.Image) -> DenoiseResult:
        tensor = self._prepare(image)
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            output = self.model.denoise(tensor)
        if self.add_constant != 0.0:
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
        raise ValueError("Prediction and reference size mismatch")
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Complementary-BSN denoising on a single sRGB image.")
    parser.add_argument("--input", required=True, help="Path to the noisy sRGB image")
    parser.add_argument("--output", required=True, help="Where to save the denoised image")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default=None, help="Device spec such as 'cuda:0' or 'cpu'")
    parser.add_argument("--reference", default=None, help="Optional clean reference image to compute PSNR")
    parser.add_argument("--disable-r3", action="store_true", help="Disable Random Replacing Refinement")
    parser.add_argument("--add-constant", type=float, default=0.5, help="Constant added before quantization (default 0.5)")
    parser.add_argument(
        "--disable-floor",
        dest="floor_output",
        action="store_false",
        help="Do not floor the output intensities before saving",
    )
    parser.set_defaults(floor_output=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weight_path = Path(args.weights).expanduser().resolve() if args.weights else DEFAULT_WEIGHT
    reference_path = Path(args.reference).expanduser().resolve() if args.reference else None

    denoiser = ComplementaryBSNDenoiser(
        weight_path=str(weight_path),
        device=args.device,
        disable_r3=args.disable_r3,
        add_constant=args.add_constant,
        floor_output=args.floor_output,
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
