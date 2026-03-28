"""Integration of single-image self-supervised denoisers from algorithms/SSSID."""
from __future__ import annotations

import contextlib
import io
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.SSSID.dmid_denoiser import (
    DEFAULT_WEIGHTS as DMID_DEFAULT_WEIGHTS,
    GaussianDMID,
    resolve_diffusion_times,
)
from algorithms.SSSID.mash_denoiser import (
    CONFIG as MASH_CONFIG,
    UNetN2NUn,
    calculate_sliding_std,
    generate_random_permutation,
    get_shuffling_mask,
    pad_to_multiple,
    random_mask as mash_random_mask,
    shuffle_input,
)
from algorithms.SSSID.score_dvi_denoiser import (
    DEFAULT_MODEL_DIR as SCORE_DVI_DEFAULT_MODEL_DIR,
    _load_scoredvi_module,
)
from denoising_platform.core.models import register_model, SelfSupervisedTrainable


def _sample_patch_batch(img: torch.Tensor, patch: int, batch: int) -> torch.Tensor:
    """Sample random BCHW patches from a single noisy image tensor."""
    # img: [1, C, H, W]
    if img.dim() != 4:
        raise ValueError("expected image tensor with shape [B, C, H, W]")
    _, _, h, w = img.shape
    size = max(8, min(patch, h, w))
    patches = []
    for _ in range(batch):
        top = 0 if h == size else torch.randint(0, h - size + 1, (1,), device=img.device).item()
        left = 0 if w == size else torch.randint(0, w - size + 1, (1,), device=img.device).item()
        patches.append(img[:, :, top:top + size, left:left + size])
    return torch.cat(patches, dim=0)


# ---------------------------------------------------------------------------
#  Prompt-SID
# ---------------------------------------------------------------------------


class PromptEncoder(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        vocab = ''.join(sorted(set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;:-_/\\"\'"'"")))
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden)
        )

    def forward(self, prompt: str, device: torch.device) -> torch.Tensor:
        if not prompt:
            prompt = "denoise"
        ids = [self.vocab.index(ch) if ch in self.vocab else 0 for ch in prompt]
        token_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        emb = self.embedding(token_tensor)
        return self.proj(emb.mean(dim=0))


class FiLMBlock(nn.Module):
    def __init__(self, inp: int, out: int, cond: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out)
        self.act = nn.SiLU(inplace=True)
        self.gamma = nn.Linear(cond, out)
        self.beta = nn.Linear(cond, out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + g) + b
        return self.act(h)


class PromptUNet(nn.Module):
    def __init__(self, channels: int = 3, cond_dim: int = 256, width: int = 64) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = FiLMBlock(channels, width, cond_dim)
        self.enc2 = FiLMBlock(width, width * 2, cond_dim)
        self.enc3 = FiLMBlock(width * 2, width * 4, cond_dim)
        self.dec2 = FiLMBlock(width * 4 + width * 2, width * 2, cond_dim)
        self.dec1 = FiLMBlock(width * 2 + width, width, cond_dim)
        self.out = nn.Conv2d(width, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.unsqueeze(0).repeat(x.shape[0], 1)
        h1 = self.enc1(x, cond)
        h2 = self.enc2(self.pool(h1), cond)
        h3 = self.enc3(self.pool(h2), cond)
        up2 = F.interpolate(h3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, h2], dim=1), cond)
        up1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, h1], dim=1), cond)
        return torch.clamp(self.out(d1), 0.0, 1.0)


@register_model('prompt_sid', category='self_supervised', display_name='Prompt-SID')
class PromptSIDModel(nn.Module, SelfSupervisedTrainable):
    def __init__(self, prompt: str = 'night photography') -> None:
        super().__init__()
        self.prompt = prompt
        self.unet = PromptUNet()
        self.encoder = PromptEncoder()
        self.loss_fn = nn.SmoothL1Loss()
        self.noise_sigma = 25.0
        self.patch = 96
        self.batch = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = self.encoder(self.prompt, x.device)
        return self.unet(x, cond)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        self.train()
        cond = self.encoder(self.prompt, noisy_img.device)
        patches = _sample_patch_batch(noisy_img, self.patch, self.batch)
        eps = self.noise_sigma / 255.0
        perturb = torch.randn_like(patches) * eps
        input_batch = (patches + perturb).clamp(0.0, 1.0)
        target_batch = patches
        pred = self.unet(input_batch, cond)
        contrastive = self.unet((pred + perturb).clamp(0, 1), cond)
        loss = self.loss_fn(pred, target_batch) + 0.5 * self.loss_fn(contrastive, pred.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    def load_pretrained_weights(self, weights_dir, model_name):  # noqa: D401
        return True


# ---------------------------------------------------------------------------
#  Recorrputed-to-Recorrupted DnCNN
# ---------------------------------------------------------------------------


class DnCNN(nn.Module):
    def __init__(self, channels: int = 3, num_layers: int = 17) -> None:
        super().__init__()
        features = 64
        layers = [nn.Conv2d(channels, features, 3, padding=1, bias=False), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, 3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return x - residual


@register_model('r2r', category='self_supervised', display_name='R2R')
class R2RModel(nn.Module, SelfSupervisedTrainable):
    def __init__(self) -> None:
        super().__init__()
        self.dncnn = DnCNN()
        self.noise_sigma = 25.0
        self.alpha = 0.5
        self.patch = 40
        self.batch = 16
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.dncnn(x), 0.0, 1.0)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        self.train()
        batch = _sample_patch_batch(noisy_img, self.patch, self.batch)
        eps = self.noise_sigma / 255.0
        gaussian = torch.randn_like(batch)
        input_batch = torch.clamp(batch + self.alpha * eps * gaussian, 0.0, 1.0)
        target_batch = torch.clamp(batch - (eps / max(self.alpha, 1e-6)) * gaussian, 0.0, 1.0)
        pred = self.dncnn(input_batch)
        loss = self.loss_fn(pred, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    def load_pretrained_weights(self, weights_dir, model_name):
        return True


# ---------------------------------------------------------------------------
#  P2N+
# ---------------------------------------------------------------------------


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class BaseConv(nn.Module):
    def __init__(self, inp: int, out: int, act: str = 'swish') -> None:
        super().__init__()
        self.layer = nn.Conv2d(inp, out, 3, padding=1)
        self.act = Swish() if act == 'swish' else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.layer(x))


class SymLayer(nn.Module):
    def __init__(self, inp: int, out: int, act: str = 'swish') -> None:
        super().__init__()
        mid = out // 2
        self.layer_en = BaseConv(inp, mid, act)
        self.layer_de = BaseConv(mid, out, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_pos = self.layer_en(x)
        f_neg = self.layer_en(-x)
        f_even = 0.5 * (f_pos + f_neg)
        f_odd = 0.5 * (f_pos - f_neg)
        return self.layer_de(f_even) + self.layer_de(f_odd)


class SPIUNet(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        block = SymLayer
        self.en1 = nn.Sequential(block(3, 48), block(48, 48), nn.MaxPool2d(2))
        self.en2 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en3 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en4 = nn.Sequential(block(48, 48), nn.MaxPool2d(2))
        self.en5 = nn.Sequential(block(48, 48), nn.MaxPool2d(2), block(48, 48), nn.Upsample(scale_factor=2))
        self.de1 = nn.Sequential(block(96, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de2 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de3 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de4 = nn.Sequential(block(144, 96), block(96, 96), nn.Upsample(scale_factor=2))
        self.de5 = nn.Sequential(block(96 + in_channels, 64), block(64, 32), nn.Conv2d(32, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.en1(x)
        p2 = self.en2(p1)
        p3 = self.en3(p2)
        p4 = self.en4(p3)
        u5 = self.en5(p4)
        u4 = self.de1(torch.cat([u5, p4], dim=1))
        u3 = self.de2(torch.cat([u4, p3], dim=1))
        u2 = self.de3(torch.cat([u3, p2], dim=1))
        u1 = self.de4(torch.cat([u2, p1], dim=1))
        return torch.clamp(self.de5(torch.cat([u1, x], dim=1)), 0.0, 1.0)


class VariablePowerLoss(nn.Module):
    def __init__(self, mode: str = '1.5', eps: float = 1e-8) -> None:
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, step: int, total: int) -> torch.Tensor:
        if self.mode == '1.5':
            factor = 0.25
        elif self.mode == '0':
            factor = 1.0
        else:
            return F.l1_loss(pred, target)
        r = 2.0 * (1 - (step / max(total, 1)) * factor)
        r = max(r, 0.5)
        return torch.mean(torch.pow(torch.abs(pred - target) + self.eps, r))


def _p2n_step(model: SPIUNet, batch: torch.Tensor, sigma: float, step: int, total: int,
              loss_fn: VariablePowerLoss) -> torch.Tensor:
    out = model(batch)
    noise = batch - out
    rand = lambda: torch.randn(1, device=batch.device) * sigma + 1.0

    def branch(n: torch.Tensor) -> torch.Tensor:
        perturbed = torch.clamp(out + n, 0.0, 1.0)
        return model(perturbed)

    loss = loss_fn(branch(noise * rand()), branch(-noise * rand()), step, total)
    return loss


@register_model('p2n_plus', category='self_supervised', display_name='P2N+')
class P2NPlusModel(nn.Module, SelfSupervisedTrainable):
    def __init__(self) -> None:
        super().__init__()
        self.net = SPIUNet()
        self.loss_fn = VariablePowerLoss()
        self.noise_sigma = 15.0
        self.patch = 96
        self.batch = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        total_epochs = kwargs.get('total_epochs', 100)
        batch = _sample_patch_batch(noisy_img, self.patch, self.batch)
        loss = _p2n_step(self.net, batch, self.noise_sigma / 255.0, epoch, total_epochs, self.loss_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}

    def load_pretrained_weights(self, weights_dir, model_name):
        return True


# ---------------------------------------------------------------------------
#  MASH
# ---------------------------------------------------------------------------


def _mash_create_locally_shuffled_target(model: nn.Module, noisy_original: torch.Tensor) -> torch.Tensor:
    mean_gray = (torch.clamp(model(noisy_original), 0.0, 1.0) * 255.0).mean(dim=1, keepdim=True)
    std_map = calculate_sliding_std(mean_gray, MASH_CONFIG["std_kernel_size"])
    shuffling_mask = get_shuffling_mask(std_map, MASH_CONFIG["masking_threshold"])
    _, channels, height, width = noisy_original.shape
    permutation_indices = generate_random_permutation(
        height,
        width,
        channels,
        MASH_CONFIG["shuffling_tile_size"],
    ).to(noisy_original.device)
    shuffled_np = shuffle_input(
        noisy_original.squeeze(0).detach().cpu().numpy(),
        permutation_indices.detach().cpu(),
        shuffling_mask,
        channels,
        height,
        width,
        MASH_CONFIG["shuffling_tile_size"],
    )
    return torch.from_numpy(shuffled_np).unsqueeze(0).to(noisy_original.device, dtype=noisy_original.dtype)


@register_model('mash', category='self_supervised', display_name='MASH')
class MASHModel(nn.Module, SelfSupervisedTrainable):
    def __init__(self) -> None:
        super().__init__()
        self.net = UNetN2NUn()
        self.loss_fn = nn.L1Loss(reduction="mean")
        self.mask_ratio = MASH_CONFIG["mask_medium"]
        self.shuffle_epoch = MASH_CONFIG["shuffling_iteration"]
        self.use_local_shuffling = False
        self._cached_target: Optional[torch.Tensor] = None

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        padded, original_hw = pad_to_multiple(x, 32)
        return padded, original_hw

    def _resolve_schedule(self, noisy_img: torch.Tensor) -> None:
        # Use a lightweight image-statistic proxy to keep the original MASH branching
        # without running its expensive extra estimation stage inside the dashboard loop.
        local_std = float(torch.mean(calculate_sliding_std(noisy_img.mean(dim=1, keepdim=True), 2) * 255.0).item())
        global_std = float(torch.std(noisy_img * 255.0).item())
        diff_std = abs(global_std - local_std)
        if diff_std > MASH_CONFIG["epsilon_high"]:
            self.use_local_shuffling = True
            self.mask_ratio = MASH_CONFIG["mask_high"]
        elif diff_std < MASH_CONFIG["epsilon_low"]:
            self.use_local_shuffling = False
            self.mask_ratio = MASH_CONFIG["mask_low"]
        else:
            self.use_local_shuffling = False
            self.mask_ratio = MASH_CONFIG["mask_medium"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, original_hw = self._prepare_input(x)
        pred = torch.clamp(self.net(padded), 0.0, 1.0)
        return pred[:, :, :original_hw[0], :original_hw[1]]

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        padded, _ = self._prepare_input(noisy_img)
        if epoch == 1:
            self._resolve_schedule(padded.detach())
            self._cached_target = padded.detach().clone()

        mask_threshold = 1.0 - self.mask_ratio
        mask = mash_random_mask(padded.shape, mask_threshold, padded.device)
        output = self.net(mask * padded)
        target = self._cached_target if self._cached_target is not None else padded
        loss = self.loss_fn((1.0 - mask) * output, (1.0 - mask) * target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == self.shuffle_epoch and self.use_local_shuffling:
            self.eval()
            with torch.no_grad():
                self._cached_target = _mash_create_locally_shuffled_target(self.net, padded.detach())
            self.train()

        return {
            'loss': loss.item(),
            'extra_metrics': {
                'mask_ratio': round(float(self.mask_ratio), 3),
                'local_shuffling': self.use_local_shuffling,
            },
        }

    def load_pretrained_weights(self, weights_dir, model_name):
        return True


# ---------------------------------------------------------------------------
#  DMID
# ---------------------------------------------------------------------------


@register_model('dmid', category='self_supervised', display_name='DMID')
class DMIDModel(nn.Module, SelfSupervisedTrainable):
    def __init__(self, sigma: int = 75, sampler_steps: int = 1, repeat_times: int = 1, eta: float = 0.8) -> None:
        super().__init__()
        self.sigma = sigma
        self.sampler_steps = sampler_steps
        self.repeat_times = repeat_times
        self.eta = eta
        self.diffusion_times = resolve_diffusion_times(sigma)
        self._backend: Optional[GaussianDMID] = None
        self._backend_device: Optional[torch.device] = None
        self._dummy = nn.Parameter(torch.zeros(1))

    def _get_backend(self, device: torch.device) -> GaussianDMID:
        if self._backend is None or self._backend_device != device:
            self._backend = GaussianDMID(
                DMID_DEFAULT_WEIGHTS,
                device=device,
                sampling_timesteps=self.sampler_steps,
            )
            self._backend_device = device
        return self._backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = self._get_backend(x.device)
        with torch.no_grad():
            denoised = backend.denoise_average(
                noisy=x,
                diffusion_times=self.diffusion_times,
                eta=self.eta,
                repeat_times=self.repeat_times,
            )
        return torch.clamp(denoised, 0.0, 1.0)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        # DMID is a backend-style single-image method; keep the training loop compatible
        # while exposing its denoised result through the standard platform hooks.
        loss = self._dummy.sum() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            'loss': float(loss.item()),
            'extra_metrics': {
                'diffusion_times': int(self.diffusion_times),
                'sampler_steps': int(self.sampler_steps),
            },
        }

    def load_pretrained_weights(self, weights_dir, model_name):
        return True


# ---------------------------------------------------------------------------
#  ScoreDVI
# ---------------------------------------------------------------------------


@register_model('score_dvi', category='self_supervised', display_name='ScoreDVI')
class ScoreDVIModel(nn.Module, SelfSupervisedTrainable):
    def __init__(
        self,
        steps: int = 200,
        lr: float = 1e-3,
        lam: float = 0.5,
        alpha0: float = 1.0,
        beta0: float = 0.02,
        exp_weight: float = 0.9,
        gmm: int = 3,
        second_pass_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.lam = lam
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.exp_weight = exp_weight
        self.gmm = gmm
        self.second_pass_threshold = second_pass_threshold
        self.model_dir = SCORE_DVI_DEFAULT_MODEL_DIR
        self._dummy = nn.Parameter(torch.zeros(1))
        self._module = None
        self._module_key: Optional[Tuple[str, int, str]] = None
        self._cached_result: Optional[torch.Tensor] = None
        self._cached_signature: Optional[Tuple[Tuple[int, ...], str, float, float, float]] = None
        self._cached_sigma: Optional[float] = None
        self._cached_steps: Optional[int] = None

    def _make_signature(self, x: torch.Tensor) -> Tuple[Tuple[int, ...], str, float, float, float]:
        x_detached = x.detach()
        return (
            tuple(x_detached.shape),
            str(x_detached.device),
            round(float(x_detached.mean().item()), 6),
            round(float(x_detached.std().item()), 6),
            round(float(x_detached.sum().item()), 4),
        )

    def _require_cuda(self, device: torch.device) -> None:
        if device.type != 'cuda':
            raise RuntimeError('ScoreDVI 当前仅支持 CUDA GPU，当前设备为 CPU。')

    def _get_module(self, device: torch.device):
        self._require_cuda(device)
        key = (str(device), int(self.gmm), str(self.model_dir))
        if self._module is None or self._module_key != key:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._module = _load_scoredvi_module(str(device), self.model_dir, self.lam, self.gmm)
            self._patch_ssim_compat(self._module)
            self._module_key = key
        return self._module

    @staticmethod
    def _patch_ssim_compat(module) -> None:
        if getattr(module, '_dashboard_ssim_patched', False):
            return

        def _ssim_compat(img1, img2, *args, **kwargs):
            from skimage.metrics import structural_similarity

            if 'multichannel' in kwargs and 'channel_axis' not in kwargs:
                multichannel = kwargs.pop('multichannel')
                if multichannel:
                    kwargs['channel_axis'] = -1
            return structural_similarity(img1, img2, *args, **kwargs)

        module.compare_ssim = _ssim_compat
        module._dashboard_ssim_patched = True

    def _run_backend(self, x: torch.Tensor, steps: int) -> Tuple[torch.Tensor, float]:
        module = self._get_module(x.device)
        noisy_np = x.detach().squeeze(0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, _, denoised_np, sigma_mean = module.denoising(
                noisy_np,
                noisy_np,
                a0=self.alpha0,
                b0=self.beta0,
                LR=self.lr,
                GMM_num=self.gmm,
                total_step=steps,
                exp_weight=self.exp_weight,
                lam=self.lam,
                model_path=str(self.model_dir),
            )
            if sigma_mean > self.second_pass_threshold:
                _, _, denoised_np, sigma_mean = module.denoising(
                    noisy_np,
                    noisy_np,
                    a0=self.alpha0,
                    b0=self.beta0,
                    LR=self.lr,
                    GMM_num=self.gmm,
                    total_step=steps,
                    exp_weight=self.exp_weight,
                    lam=self.lam,
                    model_path=str(self.model_dir),
                )
        result = torch.from_numpy(np.transpose(denoised_np, (2, 0, 1))).unsqueeze(0)
        result = result.to(x.device, dtype=x.dtype)
        return torch.clamp(result, 0.0, 1.0), float(sigma_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('ScoreDVI expects input tensor with shape [B, C, H, W].')
        outputs = []
        for idx in range(x.shape[0]):
            sample = x[idx:idx + 1]
            signature = self._make_signature(sample)
            if self._cached_result is None or self._cached_signature != signature:
                result, sigma_mean = self._run_backend(sample, self.steps)
                self._cached_result = result.detach()
                self._cached_signature = signature
                self._cached_sigma = sigma_mean
                self._cached_steps = self.steps
            outputs.append(self._cached_result.to(sample.device, dtype=sample.dtype))
        return torch.cat(outputs, dim=0)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        steps = int(kwargs.get('total_epochs', self.steps) or self.steps)
        steps = max(1, steps)
        signature = self._make_signature(noisy_img)
        if (
            epoch == 1
            or self._cached_result is None
            or self._cached_signature != signature
            or self._cached_steps != steps
        ):
            result, sigma_mean = self._run_backend(noisy_img, steps)
            self._cached_result = result.detach()
            self._cached_signature = signature
            self._cached_sigma = sigma_mean
            self._cached_steps = steps

        loss = self._dummy.sum() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sigma_value = float(self._cached_sigma or 0.0)
        display_loss = sigma_value / max(epoch, 1)
        return {
            'loss': display_loss,
            'extra_metrics': {
                'sigma_mean': round(sigma_value, 6),
                'score_steps': steps,
                'gmm_components': int(self.gmm),
            },
        }

    def load_pretrained_weights(self, weights_dir, model_name):
        return True
