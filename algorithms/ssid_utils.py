"""Shared helpers for SSID-based model integrations."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional


def find_weight_file(
    weights_dir: Path,
    candidate_names: Iterable[Optional[str]],
    fallback: Optional[Callable[[], Path]] = None,
) -> Path:
    """Return the first existing weight path among candidates or use fallback."""

    for name in candidate_names:
        if not name:
            continue
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    if fallback is not None:
        return fallback()
    raise FileNotFoundError(
        "未找到预训练权重文件，请将 .pth 放到/home/trui/MyCodes/Project_denosing/algorithms/SSID/pretrained_models或设置相应环境变量"
    )

