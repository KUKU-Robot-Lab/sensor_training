"""
contact_label.py

깊이 기반 접촉 라벨(히트맵) 생성 유틸.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np

from .contact_geometry import contact_radius, radial_weight

KernelType = Literal["gaussian", "linear"]


def make_radial_label(
    center_xy_mm: Tuple[float, float],
    grid_min_mm: float,
    grid_max_mm: float,
    grid_step_mm: float,
    radius_mm: float,
    kernel: KernelType = "gaussian",
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """
    주어진 중심과 반경을 바탕으로 2D 라벨 히트맵을 생성한다.

    Returns:
        label: shape (H, W) ndarray, grid_min~grid_max 범위, grid_step 격자.
    """

    xs = np.arange(grid_min_mm, grid_max_mm + 1e-6, grid_step_mm)
    ys = np.arange(grid_min_mm, grid_max_mm + 1e-6, grid_step_mm)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    dx = xx - center_xy_mm[0]
    dy = yy - center_xy_mm[1]
    dist = np.sqrt(dx * dx + dy * dy)

    a = radius_mm
    if kernel == "gaussian":
        sigma = max(1e-6, a * sigma_scale)
        label = np.exp(-(dist * dist) / (2 * sigma * sigma))
    else:
        label = np.maximum(0.0, 1.0 - dist / max(1e-6, a))
    return label.astype(np.float32)


__all__ = ["make_radial_label", "KernelType", "contact_radius", "radial_weight"]
