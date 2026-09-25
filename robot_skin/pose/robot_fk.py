"""URDF forward kinematics → taxel poses (stub).

Plan
- parse the hand URDF once (e.g. ``yourdfpy``/``pinocchio``; keep the dependency optional),
- per timestamp: joint state ``q`` (aligned with ``common.timeline``) → link transforms,
- hand the ``{link: T}`` dict to ``provider.transform_taxels`` (layout ``parent`` = URDF link).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from common.layouts import Layout


class RobotFKPoseProvider:
    def __init__(self, layout: Layout, urdf_path: str | Path,
                 joint_state_at: Callable[[float], np.ndarray]):
        if layout.parent_frame != "urdf":
            raise ValueError("RobotFKPoseProvider needs a layout with parent_frame: urdf")
        raise NotImplementedError("URDF FK not implemented yet (see module docstring)")

    @property
    def n_taxels(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def pose_at(self, t: float):  # pragma: no cover - stub
        raise NotImplementedError
