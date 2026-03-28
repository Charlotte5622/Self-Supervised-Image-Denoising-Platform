"""MM-BSN integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.mm_bsn_denoiser import (
    MMBSNBlindSpot,
    MMBSN_PARAMS,
    _clean_state_dict,
    _extract_state_dict,
    _resolve_weight_path,
    DEFAULT_WEIGHT_NAME,
)
from algorithms.ssid_utils import find_weight_file


@register_model('mmbsn', category='pretrained', display_name='MM-BSN (SIDD)')
class MMBSNModel(nn.Module):
    """Wraps MM-BSN Blind-Spot Network into the platform's interface."""

    def __init__(self):
        super().__init__()
        self.backend = MMBSNBlindSpot(dict(MMBSN_PARAMS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * 255.0
        denoised = self.backend.denoise(scaled)
        return torch.clamp(denoised / 255.0, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = find_weight_file(
            weights_dir,
            (
                f'{model_name}.pth',
                'MMBSN_SIDD_o_a45.pth',
                DEFAULT_WEIGHT_NAME,
            ),
            fallback=lambda: _resolve_weight_path(None),
        )
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.backend.load_state_dict(state_dict)
        return True
