"""TaxelPoseProvider — ``t → (positions[N,3], normals[N,3])`` in a common world/hand frame.

Every provider resolves each taxel's ``parent`` (URDF link or MANO segment) to a 4×4 pose and
maps the layout's local position/normal through it (:func:`transform_taxels`). Only the source of
the parent poses differs:

- :class:`StaticPoseProvider` — fixed parent transforms (bench pad, rigid mount). Implemented.
- :class:`TransformPoseProvider` — any ``t → {parent: T}`` callable. Implemented.
- ``robot_fk.RobotFKPoseProvider`` — URDF forward kinematics from joint states (``pose.urdf``).
- ``glove_imu2mano.GloveImu2ManoPoseProvider`` — 7 IMUs → MANO finger pose (``pose.imu_model``;
  VIFNet-S loader is a documented stub) → skeleton FK (``pose.mano``).
- ``mano.ManoPoseProvider`` — MANO pose callable → skeleton FK.
"""
from __future__ import annotations

from typing import Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from common.layouts import Layout


@runtime_checkable
class TaxelPoseProvider(Protocol):
    @property
    def n_taxels(self) -> int: ...

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions[N,3], normals[N,3])`` at time ``t`` (layout order)."""
        ...


def transform_taxels(layout: Layout, parent_T: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Map local taxel poses through per-parent 4×4 transforms. Missing parents → KeyError."""
    missing = set(layout.parents) - set(parent_T)
    if missing:
        raise KeyError(f"no transform for parents: {sorted(missing)}")
    pos_l, nrm_l = layout.positions, layout.normals
    pos = np.empty_like(pos_l)
    nrm = np.empty_like(nrm_l)
    for i, parent in enumerate(layout.parents):
        T = np.asarray(parent_T[parent], dtype=np.float64)
        R, p = T[:3, :3], T[:3, 3]
        pos[i] = R @ pos_l[i] + p
        nrm[i] = R @ nrm_l[i]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pos, nrm


class TransformPoseProvider:
    """Poses from a user callable ``t → {parent_name: 4×4}``."""

    def __init__(self, layout: Layout, parent_transforms: Callable[[float], Mapping[str, np.ndarray]]):
        self.layout = layout
        self._fn = parent_transforms

    @property
    def n_taxels(self) -> int:
        return self.layout.n

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return transform_taxels(self.layout, self._fn(float(t)))


class StaticPoseProvider(TransformPoseProvider):
    """Time-invariant poses. ``parent_transforms=None`` → identity for every parent."""

    def __init__(self, layout: Layout, parent_transforms: Mapping[str, np.ndarray] | None = None):
        tf = dict(parent_transforms) if parent_transforms is not None else {
            p: np.eye(4) for p in set(layout.parents)}
        self._cache = transform_taxels(layout, tf)
        super().__init__(layout, lambda _t: tf)

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self._cache[0].copy(), self._cache[1].copy()


def sample_poses(provider: TaxelPoseProvider, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a provider on a time grid → ``(pos[T,N,3], nrm[T,N,3])``."""
    out = [provider.pose_at(float(t)) for t in np.asarray(ts)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
