"""AP-BSN integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from denoising_platform.core.models import register_model

from algorithms.SSID.ap_bsn_denoiser import (
    APBSN,
    APBSN_PARAMS,
    _clean_state_dict,
    _extract_state_dict,
    _resolve_weight_path,
)


@register_model('apbsn', category='pretrained', display_name='AP-BSN (SIDD)')
class APBSNModel(nn.Module):
    """Wraps the original AP-BSN denoiser to match the platform API."""

    def __init__(self):
        super().__init__()
        self.backend = APBSN(**APBSN_PARAMS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * 255.0
        with torch.no_grad():
            denoised = self.backend.denoise(x_scaled)
        denoised = torch.clamp(denoised, 0.0, 255.0)
        return torch.clamp(denoised / 255.0, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = _find_weight_path(weights_dir, model_name)
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        self.backend.load_state_dict(state_dict)
        return True


def _find_weight_path(weights_dir: Path, model_name: str) -> Path:
    candidates: Iterable[Path] = (
        weights_dir / f'{model_name}.pth',
        weights_dir / 'APBSN_SIDD.pth',
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return _resolve_weight_path(None)
