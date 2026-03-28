"""Complementary-BSN integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.complementary_bsn_denoiser import (
    APBSN,
    APBSN_PARAMS,
    DEFAULT_WEIGHT,
)
from algorithms.ssid_utils import find_weight_file

DEFAULT_ADD = 0.5


@register_model('complementary_bsn', category='pretrained', display_name='Complementary-BSN')
class ComplementaryBSNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = APBSN(dict(APBSN_PARAMS))
        self.add_constant = DEFAULT_ADD
        self.floor_output = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * 255.0
        denoised = self.backend.denoise(scaled)
        if self.add_constant:
            denoised = denoised + self.add_constant
        if self.floor_output:
            denoised = torch.floor(denoised)
        return torch.clamp(denoised / 255.0, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = find_weight_file(
            weights_dir,
            (f'{model_name}.pth', 'Complementary_BSN_SIDD.pth'),
            fallback=lambda: DEFAULT_WEIGHT,
        )
        checkpoint = torch.load(weight_path, map_location='cpu')
        state = checkpoint.get('model_weight', checkpoint)
        if isinstance(state, dict) and 'denoiser' in state:
            state = state['denoiser']
        self.backend.load_state_dict(state, strict=False)
        return True
