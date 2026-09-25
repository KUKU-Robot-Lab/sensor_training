"""Self-touch auto-labels from finger-segment distances.

Hand segments are capsules (``p0 → p1``, radius ``r``). A taxel is labelled self-touched when
its world position is within ``radius + margin`` of any segment that is *not* on its own
kinematic chain (own finger; palm taxels exclude the palm/wrist). Gives free contact labels for
fist / pinch / finger-crossing motions without any external sensor.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def finger_of(segment: str) -> str:
    """``index3`` / ``index_distal_link`` → ``index``; ``palm``/``wrist`` → ``palm``."""
    s = segment.lower()
    for f in FINGERS:
        if s.startswith(f):
            return f
    if s.startswith("palm") or s.startswith("wrist"):
        return "palm"
    return s


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points ``p[...,3]`` to segments ``a,b[...,3]`` (broadcast)."""
    ab = b - a
    denom = np.sum(ab * ab, axis=-1, keepdims=True)
    t = np.clip(np.sum((p - a) * ab, axis=-1, keepdims=True) / np.where(denom > 0, denom, 1.0), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab), axis=-1)


def self_touch_labels(taxel_pos: np.ndarray, taxel_parents: Sequence[str], seg_p0: np.ndarray,
                      seg_p1: np.ndarray, seg_radius: np.ndarray | float,
                      seg_names: Sequence[str], *, margin: float = 0.004) -> np.ndarray:
    """``taxel_pos[T,N,3]``, ``seg_p0/p1[T,S,3]`` → bool ``[T,N]``."""
    tp = np.asarray(taxel_pos, dtype=np.float64)
    a = np.asarray(seg_p0, dtype=np.float64)
    b = np.asarray(seg_p1, dtype=np.float64)
    if tp.ndim == 2:
        tp, a, b = tp[None], a[None], b[None]
    rad = np.broadcast_to(np.asarray(seg_radius, dtype=np.float64), (a.shape[1],))
    d = point_segment_distance(tp[:, :, None, :], a[:, None, :, :], b[:, None, :, :])  # [T,N,S]
    own = np.array([[finger_of(tpar) == finger_of(sn) for sn in seg_names] for tpar in taxel_parents])
    hit = (d <= rad[None, None, :] + margin) & ~own[None]
    return hit.any(-1)
