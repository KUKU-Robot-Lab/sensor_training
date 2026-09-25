"""Glove ⇄ robot-hand transfer through MANO (stub).

Plan
- ``project_to_mano``: robot-hand taxel poses (URDF FK) → nearest MANO surface point/segment
  (retargeted hand), so robot and glove taxels live on one canonical hand.
- ``align_layouts``: correspondence between a glove layout and a robot-hand layout on that
  canonical hand (by segment + nearest position), used to map tokens / labels across embodiments.
"""
from __future__ import annotations

import numpy as np

from common.layouts import Layout


def project_to_mano(taxel_pos: np.ndarray, mano_vertices: np.ndarray, mano_segments: np.ndarray):
    raise NotImplementedError("MANO projection not implemented yet")


def align_layouts(src: Layout, dst: Layout, **kw) -> np.ndarray:
    raise NotImplementedError("layout alignment not implemented yet")
