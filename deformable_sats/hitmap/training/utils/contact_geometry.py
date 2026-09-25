"""
contact_geometry.py

깊이에 따라 접촉 반경을 계산하고, 거리 기반 가중치를 생성하는 유틸 함수 모음.

기본값은 인덴터 반경 R=2.5mm(지름 5mm)이며, 기존 코드와의 호환성을 위해
모델과 커널을 플래그로 선택할 수 있도록 설계했다.
"""

from __future__ import annotations

import math
from typing import Literal


RadiusModel = Literal["hertz", "geo", "geom"]
KernelType = Literal["gaussian", "linear"]


def contact_radius(depth_mm: float, R_mm: float = 2.5, model: RadiusModel = "hertz") -> float:
    """깊이(mm)와 인덴터 반경(mm)으로 접촉 반경(mm)을 계산한다.

    Args:
        depth_mm: 압입 깊이 (mm). 음수면 0으로 취급.
        R_mm: 인덴터 반경 (mm). 기본 2.5 (지름 5mm).
        model: "hertz" → a = sqrt(R·δ), "geo"/"geom" → a = sqrt(2Rδ − δ²).

    Returns:
        접촉 반경 a [mm]. 깊이가 0 이하인 경우 0.
    """

    if depth_mm <= 0 or R_mm <= 0:
        return 0.0

    if model in {"geo", "geom"}:
        val = 2 * R_mm * depth_mm - depth_mm * depth_mm
        return math.sqrt(val) if val > 0 else 0.0

    # default: Hertzian
    return math.sqrt(R_mm * depth_mm)


def contact_radius_tensor(depth_mm, R_mm: float = 2.5, model: RadiusModel = "hertz"):
    """Torch-friendly 접촉 반경 계산 (broadcast 지원).

    Args:
        depth_mm: torch.Tensor 또는 숫자. 음수는 0으로 클램프.
        R_mm: 인덴터 반경(mm)
        model: "hertz" 또는 "geo"/"geom"
    """

    import torch

    depth = torch.clamp(depth_mm, min=0.0)
    if model in {"geo", "geom"}:
        a_sq = torch.clamp(2 * R_mm * depth - depth * depth, min=0.0)
    else:
        a_sq = torch.clamp(R_mm * depth, min=0.0)
    return torch.sqrt(torch.clamp(a_sq, min=1e-12))


def radial_weight(dist_mm: float, a_mm: float, kernel: KernelType = "gaussian") -> float:
    """거리 기반 가중치 (라벨 커널).

    Args:
        dist_mm: 중심으로부터의 거리 (mm)
        a_mm: 접촉 반경 (mm)
        kernel: "gaussian" 또는 "linear"
    """

    if a_mm <= 0:
        return 0.0

    if kernel == "linear":
        return max(0.0, 1.0 - dist_mm / a_mm)

    # gaussian
    return math.exp(-(dist_mm * dist_mm) / (2 * a_mm * a_mm))


__all__ = ["contact_radius", "radial_weight", "RadiusModel", "KernelType"]
