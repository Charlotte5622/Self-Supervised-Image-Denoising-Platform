"""ARA-BSN integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.ara_bsn_denoiser import (
    ARABSN,
    ARA_BSN_PARAMS,
    DEFAULT_WEIGHT,
)
from algorithms.ssid_utils import find_weight_file

DEFAULT_LOW = 1.0
DEFAULT_HIGH = 5.0
DEFAULT_WINDOW = 3
DEFAULT_SCALE1 = 1.0
DEFAULT_SCALE2 = 3.0
DEFAULT_LPD = 4
DEFAULT_ADD = 0.5


@register_model('arab_bsn', category='pretrained', display_name='ARA-BSN')
class ARABSNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = ARABSN(dict(ARA_BSN_PARAMS))
        self.low = DEFAULT_LOW
        self.high = DEFAULT_HIGH
        self.window_size = DEFAULT_WINDOW
        self.scale1 = DEFAULT_SCALE1
        self.scale2 = DEFAULT_SCALE2
        self.lpd = DEFAULT_LPD
        self.add_constant = DEFAULT_ADD
        self.floor_output = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * 255.0
        denoised = self.backend.denoise(
            scaled,
            low=self.low,
            high=self.high,
            window_size=self.window_size,
            scale1=self.scale1,
            scale2=self.scale2,
            lpd=self.lpd,
        )
        if self.add_constant:
            denoised = denoised + self.add_constant
        if self.floor_output:
            denoised = torch.floor(denoised)
        return torch.clamp(denoised / 255.0, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = find_weight_file(
            weights_dir,
            (f'{model_name}.pth', 'ARA_BSN_SIDD.pth'),
            fallback=lambda: DEFAULT_WEIGHT,
        )
        checkpoint = torch.load(weight_path, map_location='cpu')
        state = checkpoint.get('model_weight', checkpoint)
        if isinstance(state, dict) and 'denoiser' in state:
            state = state['denoiser']
        self.backend.load_state_dict(state)
        return True
