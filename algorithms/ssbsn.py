"""SS-BSN integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.ss_bsn_denoiser import (
    SSBSN,
    SSBSN_PARAMS,
    PIXEL_MAX,
    DEFAULT_WEIGHT,
)
from algorithms.ssid_utils import find_weight_file


@register_model('ssbsn', category='pretrained', display_name='SS-BSN (SIDD)')
class SSBSNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = SSBSN(dict(SSBSN_PARAMS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * PIXEL_MAX
        denoised = self.backend.denoise(scaled)
        return torch.clamp(denoised / PIXEL_MAX, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        candidates = (f'{model_name}.pth', 'SSBSN_SIDD.pth')
        weight_path = find_weight_file(weights_dir, candidates, fallback=lambda: DEFAULT_WEIGHT)
        checkpoint = torch.load(weight_path, map_location='cpu')
        state = checkpoint.get('model_weight', checkpoint)
        if isinstance(state, dict) and 'denoiser' in state:
            state = state['denoiser']
        missing = self.backend.load_state_dict(state, strict=False)
        if missing.missing_keys:
            raise RuntimeError(f'SS-BSN 权重缺少字段: {missing.missing_keys}')
        return True
