"""Glove 7-IMU → MANO pose → taxel poses (stub).

Plan
- backbone: VIHand **VIFNet-S** (IMU-only variant), fine-tuned on our 7-site glove
  (wrist/palm/thumb/index/middle/ring/pinky) with camera-derived MANO labels
  recorded by ``acquisition.glove_logger``.
- output MANO pose θ (+ shape β per subject) → segment transforms (MANO kinematic tree).
- segment transforms → ``provider.transform_taxels`` (layout ``parent`` = MANO segment).
- the same MANO segments feed ``contact.self_touch`` (finger-segment distance labels).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from common.layouts import Layout


def finetune_vifnet_s(train_sessions: list[Path], out_dir: Path, **kw) -> Path:
    """Fine-tune VIFNet-S on glove sessions; returns checkpoint path. Stub."""
    raise NotImplementedError("VIFNet-S fine-tuning not implemented yet")


class GloveImu2ManoPoseProvider:
    def __init__(self, layout: Layout, checkpoint: str | Path,
                 imu_at: Callable[[float], np.ndarray]):
        if layout.parent_frame != "mano":
            raise ValueError("GloveImu2ManoPoseProvider needs a layout with parent_frame: mano")
        raise NotImplementedError("IMU→MANO inference not implemented yet (see module docstring)")
