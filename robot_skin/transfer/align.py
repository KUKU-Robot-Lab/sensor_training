"""Taxel correspondence between two skin layouts (glove ⇄ robot hand, or two glove builds).

:func:`align_layouts` maps every **destination** taxel to its ``k`` nearest **source** taxels,
never across finger groups (a thumb taxel only takes values from thumb taxels; groups from the
layout's ``finger_<f>`` / ``palm`` groups or the parent name):

- ``space="skeleton"`` (when both skeletons are given — cross-embodiment): canonical hand
  coordinates of :func:`robot_skin.transfer.project_to_skeleton` — position along the finger
  ``u`` (base 0 → tip 1), palmar/dorsal ``side`` and, on the palm, the lateral ``v`` (index side
  0 → pinky side 1) — so a human fingertip pad matches a robot fingertip pad although the hands
  differ in size, link count and frames: ``d = |Δu| + side_weight·|Δside|/2 + lateral_weight·|Δv|``
  (terms with an unknown coordinate are skipped);
- ``space="euclidean"``: distance between taxel positions given in one common frame (+ an optional
  normal-disagreement penalty) — e.g. two glove builds on the MANO hand, or robot taxels already
  registered to the human hand.

:func:`map_taxel_values` then carries per-taxel values (residual z, contact levels, labels,
features) from the source to the destination layout: inverse-distance weighted, nearest, or max
(for ordinal levels / contact flags). Destination taxels without a same-group source are marked
invalid and filled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.layouts import Layout, load_layout

from .skeleton import HAND_GROUPS, CapsuleSkeleton, finger_group, project_to_skeleton

__all__ = ["taxel_groups", "layout_rest_poses", "LayoutAlignment", "align_layouts", "map_taxel_values",
           "REDUCE_MODES"]

REDUCE_MODES = ("weighted", "nearest", "max")


def _layout(x: Any) -> Layout:
    return x if isinstance(x, Layout) else load_layout(x)


def taxel_groups(layout: Any) -> list[str]:
    """Finger group per taxel (``thumb`` … ``pinky``, ``palm``, ``other``): an explicit
    ``finger_<f>`` / ``palm`` layout group wins, else the parent segment / link name."""
    L = _layout(layout)
    out = []
    for t in L.taxels:
        g = None
        for grp in t.groups:
            if grp.startswith("finger_") and grp[7:] in HAND_GROUPS:
                g = grp[7:]
            elif grp == "palm":
                g = g or "palm"
        out.append(g or finger_group(t.parent))
    return out


def layout_rest_poses(layout: Any, *, skeleton: Any = None, urdf: Any = None, q: Any = None,
                      finger_pose: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Taxel ``(pos[N,3], nrm[N,3])`` of a layout in its hand frame: MANO layouts on the (flat or
    ``finger_pose``) MANO hand with ``global_orient = 0`` (as preprocessing), URDF layouts by FK at
    ``q`` (default zeros; ``urdf`` required), others as stored."""
    L = _layout(layout)
    if L.parent_frame == "mano":
        from ..pose.mano import ManoSkeleton, taxel_poses_from_hand

        fp = np.zeros((15, 3)) if finger_pose is None else np.asarray(finger_pose, np.float64).reshape(15, 3)
        return taxel_poses_from_hand(L, skeleton if skeleton is not None else ManoSkeleton(), np.zeros(3), fp)
    if L.parent_frame == "urdf":
        if urdf is None:
            raise ValueError(f"layout {L.name!r} is URDF-parented: pass urdf= (a URDFModel or path)")
        from ..pose.robot_fk import taxel_poses_from_joints
        from ..pose.urdf import URDFModel

        model = urdf if isinstance(urdf, URDFModel) else URDFModel.from_file(urdf)
        qv = np.zeros(model.n_dof) if q is None else np.asarray(q, np.float64)
        return taxel_poses_from_joints(L, model, np.clip(qv, model.lower, model.upper))
    return L.positions, L.normals


@dataclass
class LayoutAlignment:
    """``dst`` taxel ``i`` ← ``src`` taxels ``index[i, :k]`` with ``weight[i]`` (sum 1),
    ``distance[i]`` (in the matching space) and ``valid[i]`` (a same-group source exists within
    ``max_dist``)."""

    index: np.ndarray
    weight: np.ndarray
    distance: np.ndarray
    valid: np.ndarray
    n_src: int
    src_groups: list[str] = field(default_factory=list)
    dst_groups: list[str] = field(default_factory=list)
    space: str = "euclidean"

    @property
    def n_dst(self) -> int:
        return int(self.index.shape[0])

    @property
    def k(self) -> int:
        return int(self.index.shape[1])

    def matrix(self) -> np.ndarray:
        """Dense ``[N_dst, N_src]`` mixing matrix (rows of invalid taxels are 0)."""
        M = np.zeros((self.n_dst, self.n_src))
        for i in range(self.n_dst):
            if self.valid[i]:
                np.add.at(M[i], self.index[i], self.weight[i])
        return M

    def to_dict(self) -> dict:
        return {"index": self.index.tolist(), "weight": self.weight.tolist(), "distance": self.distance.tolist(),
                "valid": self.valid.tolist(), "n_src": self.n_src, "src_groups": list(self.src_groups),
                "dst_groups": list(self.dst_groups), "space": self.space}

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutAlignment":
        return cls(np.asarray(d["index"], np.int64), np.asarray(d["weight"], np.float64),
                   np.asarray(d["distance"], np.float64), np.asarray(d["valid"], bool), int(d["n_src"]),
                   list(d.get("src_groups", [])), list(d.get("dst_groups", [])), d.get("space", "euclidean"))

    def save(self, path: Any) -> None:
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Any) -> "LayoutAlignment":
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))


def _pairwise(space: str, s: dict, d: dict, *, side_weight: float, normal_weight: float,
              lateral_weight: float) -> np.ndarray:
    if space == "skeleton":
        du = np.abs(d["u"][:, None] - s["u"][None, :])
        ds = np.abs(d["side"][:, None] - s["side"][None, :]) / 2.0
        dv = np.abs(d["v"][:, None] - s["v"][None, :])
        return du + side_weight * np.nan_to_num(ds, nan=0.0) + lateral_weight * np.nan_to_num(dv, nan=0.0)
    D = np.linalg.norm(d["pos"][:, None, :] - s["pos"][None, :, :], axis=-1)
    if normal_weight and d.get("nrm") is not None and s.get("nrm") is not None:
        D = D + normal_weight * (1.0 - np.clip(d["nrm"] @ s["nrm"].T, -1.0, 1.0))
    return D


def align_layouts(src: Any, dst: Any, *, src_pos: np.ndarray | None = None, dst_pos: np.ndarray | None = None,
                  src_nrm: np.ndarray | None = None, dst_nrm: np.ndarray | None = None,
                  src_skeleton: CapsuleSkeleton | None = None, dst_skeleton: CapsuleSkeleton | None = None,
                  src_urdf: Any = None, dst_urdf: Any = None, k: int = 1, by_group: bool = True,
                  side_weight: float = 0.5, lateral_weight: float = 1.0, normal_weight: float = 0.0,
                  max_dist: float | None = None, eps: float = 1e-6) -> LayoutAlignment:
    """Map ``dst`` taxels to ``src`` taxels (module docstring).

    Positions default to :func:`layout_rest_poses` (MANO layouts on the flat hand; URDF layouts at
    ``q = 0`` with ``src_urdf`` / ``dst_urdf``). With both ``src_skeleton`` and ``dst_skeleton``
    (:class:`~robot_skin.transfer.CapsuleSkeleton` in the same frame as the respective positions)
    matching uses the canonical ``(u, side)`` coordinates, otherwise Euclidean distance (positions
    must share a frame). ``k`` sources per destination (inverse-distance weights); ``by_group``
    keeps finger groups apart; ``max_dist`` (in the matching space) invalidates far matches."""
    S, Dl = _layout(src), _layout(dst)
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    sg, dg = taxel_groups(S), taxel_groups(Dl)
    if src_pos is None:
        src_pos, sn = layout_rest_poses(S, urdf=src_urdf)
        src_nrm = sn if src_nrm is None else src_nrm
    if dst_pos is None:
        dst_pos, dn = layout_rest_poses(Dl, urdf=dst_urdf)
        dst_nrm = dn if dst_nrm is None else dst_nrm
    sp, dp = np.asarray(src_pos, np.float64).reshape(S.n, 3), np.asarray(dst_pos, np.float64).reshape(Dl.n, 3)
    space = "skeleton" if (src_skeleton is not None and dst_skeleton is not None) else "euclidean"
    if (src_skeleton is None) != (dst_skeleton is None):
        raise ValueError("give both src_skeleton and dst_skeleton (skeleton space) or neither (euclidean)")
    if space == "skeleton":
        ps = project_to_skeleton(sp, src_skeleton, taxel_groups=sg)
        pd = project_to_skeleton(dp, dst_skeleton, taxel_groups=dg)
        fs, fd = {"u": ps.u, "side": ps.side, "v": ps.v}, {"u": pd.u, "side": pd.side, "v": pd.v}
    else:
        fs = {"pos": sp, "nrm": None if src_nrm is None else np.asarray(src_nrm, np.float64).reshape(S.n, 3)}
        fd = {"pos": dp, "nrm": None if dst_nrm is None else np.asarray(dst_nrm, np.float64).reshape(Dl.n, 3)}
    Dm = _pairwise(space, fs, fd, side_weight=float(side_weight), normal_weight=float(normal_weight),
                   lateral_weight=float(lateral_weight))
    valid = np.ones(Dl.n, dtype=bool)
    if by_group:
        same = np.asarray([[a == b for b in sg] for a in dg])
        valid &= same.any(1)
        Dm = np.where(same, Dm, np.inf)
    kk = min(int(k), S.n)
    order = np.argsort(Dm, axis=1, kind="stable")[:, :kk]
    dist = np.take_along_axis(Dm, order, axis=1)
    finite = np.isfinite(dist)
    w = np.where(finite, 1.0 / (dist + eps), 0.0)
    w_sum = w.sum(1, keepdims=True)
    w = np.where(w_sum > 0, w / np.where(w_sum > 0, w_sum, 1.0), 0.0)
    if max_dist is not None:
        valid &= dist[:, 0] <= float(max_dist)
    valid &= finite[:, 0]
    return LayoutAlignment(order.astype(np.int64), w, dist, valid, S.n, sg, dg, space)


def map_taxel_values(values_src: Any, mapping: LayoutAlignment, *, taxel_axis: int = -1, reduce: str = "weighted",
                     fill: Any = None) -> np.ndarray:
    """Carry per-taxel values ``[..., N_src, ...]`` (taxel axis ``taxel_axis``) to the destination
    layout ``[..., N_dst, ...]``. ``reduce``: ``weighted`` (inverse-distance average — continuous
    values such as residual z or ΔS), ``nearest`` (the best match — any dtype), ``max`` (over the
    ``k`` matches — ordinal levels / contact flags). Invalid destinations get ``fill`` (default NaN
    for floats, −1 for ints, False for bools)."""
    if reduce not in REDUCE_MODES:
        raise ValueError(f"reduce must be one of {REDUCE_MODES}")
    v = np.asarray(values_src)
    ax = taxel_axis % v.ndim
    if v.shape[ax] != mapping.n_src:
        raise ValueError(f"values have {v.shape[ax]} taxels on axis {taxel_axis}, mapping expects {mapping.n_src}")
    x = np.moveaxis(v, ax, -1)                                          # [..., Ns]
    g = x[..., mapping.index]                                           # [..., Nd, k]
    if reduce == "nearest":
        out = g[..., 0]
    elif reduce == "max":
        out = g.max(-1)
    else:
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float64)
            g = x[..., mapping.index]
        out = np.sum(g * mapping.weight, axis=-1)
    if fill is None:
        fill = False if out.dtype == bool else (-1 if np.issubdtype(out.dtype, np.integer) else np.nan)
    out = np.where(mapping.valid, out, np.asarray(fill, dtype=out.dtype) if out.dtype != object else fill)
    return np.moveaxis(out, -1, ax)

