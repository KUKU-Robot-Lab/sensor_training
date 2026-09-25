"""URDF forward kinematics → taxel poses for a robot hand.

The layout's ``parent`` names are URDF links (``parent_frame: urdf``). Per time step the joint
state ``q`` (model order, see ``urdf.URDFModel.joint_names``) goes through
:meth:`URDFModel.fk <robot_skin.pose.urdf.URDFModel.fk>` and the resulting ``{link: T}`` through
``provider.transform_taxels``. Poses are in the URDF root-link frame (hand base) unless a
``base_T`` is given.

- :class:`RobotFKPoseProvider` — ``TaxelPoseProvider`` from a ``t → q`` callable (online use).
- :func:`taxel_poses_from_joints` — batched ``q[T,D] → (pos[T,N,3], nrm[T,N,3])`` (preprocessing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from common.layouts import Layout
from robot_skin.pose.provider import transform_taxels
from robot_skin.pose.urdf import URDFModel


def _check_layout(layout: Layout, model: URDFModel) -> None:
    if layout.parent_frame != "urdf":
        raise ValueError(f"layout {layout.name!r} has parent_frame {layout.parent_frame!r}; "
                         "robot FK needs parent_frame: urdf")
    missing = sorted(set(layout.parents) - set(model.link_names))
    if missing:
        raise ValueError(f"layout parents are not links of URDF {model.name!r}: {missing}. "
                         f"Rename them in the layout (URDF links: {list(model.link_names)})")


def _as_model(urdf: URDFModel | str | Path) -> URDFModel:
    return urdf if isinstance(urdf, URDFModel) else URDFModel.from_file(urdf)


class RobotFKPoseProvider:
    """``TaxelPoseProvider`` for a robot hand: ``joint_state_at(t) → q[D]`` → FK → taxel poses.

    ``urdf`` is a :class:`URDFModel` or a path to a ``.urdf`` file. ``base_T`` (4×4) optionally
    places the URDF root in another frame.
    """

    def __init__(self, layout: Layout, urdf: URDFModel | str | Path,
                 joint_state_at: Callable[[float], np.ndarray], *, base_T=None):
        if layout.parent_frame != "urdf":
            raise ValueError("RobotFKPoseProvider needs a layout with parent_frame: urdf")
        self.model = _as_model(urdf)
        _check_layout(layout, self.model)
        self.layout = layout
        self._fn = joint_state_at
        self._links = sorted(set(layout.parents))
        self.base_T = None if base_T is None else np.asarray(base_T, dtype=np.float64)
        if self.base_T is not None and self.base_T.shape != (4, 4):
            raise ValueError(f"base_T must be 4x4, got {self.base_T.shape}")

    @property
    def n_taxels(self) -> int:
        return self.layout.n

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(self._fn(float(t)), dtype=np.float64).reshape(-1)
        if q.shape[0] != self.model.n_dof:
            raise ValueError(f"joint_state_at returned {q.shape[0]} values; model has "
                             f"{self.model.n_dof} actuated joints {list(self.model.joint_names)}")
        return transform_taxels(self.layout, self.model.fk_numpy(q, base_T=self.base_T, links=self._links))


def taxel_poses_from_joints(layout: Layout, model: URDFModel | str | Path, q, *, base_T=None,
                            chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Batched robot-hand taxel poses: ``q[T,D]`` → ``(pos[T,N,3], nrm[T,N,3])`` float64.

    ``q`` columns must be in ``model.joint_names`` order (use ``model.reorder_q(q, names)`` for
    ``joint_state.npz``). ``base_T`` (4×4) places the URDF root. A single ``q[D]`` returns
    ``[N,3]`` arrays.
    """
    model = _as_model(model)
    _check_layout(layout, model)
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None]
    if q.ndim != 2 or q.shape[1] != model.n_dof:
        raise ValueError(f"q must be [T, {model.n_dof}], got {q.shape}")
    links = sorted(set(layout.parents))
    li = {ln: i for i, ln in enumerate(links)}
    idx = np.array([li[p] for p in layout.parents])
    pl = torch.as_tensor(layout.positions, dtype=torch.float64)
    nl = torch.as_tensor(layout.normals, dtype=torch.float64)
    bT = None if base_T is None else torch.as_tensor(np.asarray(base_T, dtype=np.float64))
    if bT is not None and bT.shape != (4, 4):
        raise ValueError(f"base_T must be a single 4x4 transform, got {tuple(bT.shape)}")
    T = q.shape[0]
    step = max(1, int(chunk))
    pos = np.empty((T, layout.n, 3))
    nrm = np.empty((T, layout.n, 3))
    with torch.no_grad():
        for s in range(0, T, step):
            e = min(T, s + step)
            fk = model.fk(torch.as_tensor(q[s:e]), base_T=bT, links=links)
            LT = torch.stack([fk[ln] for ln in links], dim=1)[:, idx]      # [t,N,4,4]
            R, p = LT[..., :3, :3], LT[..., :3, 3]
            pos[s:e] = (torch.matmul(R, pl[..., None])[..., 0] + p).numpy()
            n = torch.matmul(R, nl[..., None])[..., 0]
            nrm[s:e] = (n / n.norm(dim=-1, keepdim=True)).numpy()
    return (pos[0], nrm[0]) if single else (pos, nrm)
