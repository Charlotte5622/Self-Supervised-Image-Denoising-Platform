"""RSCP2GAN Restormer integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.rscp2gan_restormer import (
    Restormer,
    _clean_state_dict,
    _extract_state_dict,
    _resolve_weight_path,
)
from algorithms.ssid_utils import find_weight_file


@register_model('rscp2gan', category='pretrained', display_name='RSCP2GAN Restormer')
class RSCP2GANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = Restormer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = x.clone()
        _, _, h_old, w_old = tensor.shape
        h_pad = (h_old // 8 + 1) * 8 - h_old
        w_pad = (w_old // 8 + 1) * 8 - w_old
        if h_pad:
            tensor = torch.cat([tensor, torch.flip(tensor, [2])], dim=2)[:, :, :h_old + h_pad, :]
        if w_pad:
            tensor = torch.cat([tensor, torch.flip(tensor, [3])], dim=3)[:, :, :, :w_old + w_pad]
        denoised = self.backend(tensor)
        denoised = denoised[:, :, :h_old, :w_old]
        return torch.clamp(denoised, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = find_weight_file(
            weights_dir,
            (f'{model_name}.pth', 'RSCP2GAN_model_250.pth'),
            fallback=lambda: _resolve_weight_path(None),
        )
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.backend.load_state_dict(state_dict)
        return True
