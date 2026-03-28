"""CVF-SID integration for the denoising dashboard."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from denoising_platform.core.models import register_model

from algorithms.SSID.cvf_sid_denoiser import (
    CVFModel,
    DEFAULT_WEIGHT,
)
from algorithms.ssid_utils import find_weight_file


@register_model('cvfsid', category='pretrained', display_name='CVF-SID')
class CVFSIDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = CVFModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = F.pad(x, (20, 20, 20, 20), mode='reflect')
        clean = self.backend(tensor)
        clean = clean[:, :, 20:-20, 20:-20]
        return torch.clamp(clean, 0.0, 1.0)

    def load_pretrained_weights(self, weights_dir: Path, model_name: str) -> bool:
        weight_path = find_weight_file(
            weights_dir,
            (f'{model_name}.pth', DEFAULT_WEIGHT.name),
            fallback=lambda: DEFAULT_WEIGHT,
        )
        checkpoint = torch.load(weight_path, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        missing = self.backend.load_state_dict(state, strict=False)
        if missing.missing_keys:
            raise RuntimeError(f'CVF-SID 权重缺少字段: {missing.missing_keys}')
        return True
