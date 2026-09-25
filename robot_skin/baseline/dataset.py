"""No-contact session → windowed training samples for :class:`BaselinePredictor`.

A session is already aligned on the master clock (``common.timeline``) and converted to ΔS%
(``common.signal.relative_change``). Each sample is the state at the *last* frame of a window
(pose, q, q̇) and the target is the window-mean ΔS (averages sensor noise; the no-contact
baseline is low-frequency). Windows containing saturated samples are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from common.signal import NormStats, saturation_mask


@dataclass
class NoContactSession:
    delta_pct: np.ndarray   # [T, N]
    pos: np.ndarray         # [T, N, 3]
    nrm: np.ndarray         # [T, N, 3]
    q: np.ndarray           # [T, D]
    qd: np.ndarray          # [T, D]
    saturated: np.ndarray | None = None  # [T, N] bool

    def __post_init__(self) -> None:
        T, N = np.shape(self.delta_pct)
        for name in ("pos", "nrm"):
            if np.shape(getattr(self, name)) != (T, N, 3):
                raise ValueError(f"{name} must be [T,N,3]={(T, N, 3)}, got {np.shape(getattr(self, name))}")
        if np.shape(self.q)[0] != T or np.shape(self.qd) != np.shape(self.q):
            raise ValueError("q/qd must be [T,D] and match")


class NoContactWindowDataset(Dataset):
    def __init__(self, sessions: list[NoContactSession], *, window: int = 20, stride: int = 5,
                 max_abs_pct: float | None = 90.0, joint_norm: tuple[NormStats, NormStats] | None = None):
        if window < 1 or stride < 1:
            raise ValueError("window and stride must be >= 1")
        pos, nrm, q, qd, y = [], [], [], [], []
        for s in sessions:
            sat = saturation_mask(delta_pct=s.delta_pct, max_abs_pct=max_abs_pct)
            if s.saturated is not None:
                sat = sat | np.asarray(s.saturated, dtype=bool)
            T = s.delta_pct.shape[0]
            for end in range(window - 1, T, stride):
                sl = slice(end - window + 1, end + 1)
                if sat[sl].any():
                    continue
                pos.append(s.pos[end]); nrm.append(s.nrm[end]); q.append(s.q[end]); qd.append(s.qd[end])
                y.append(s.delta_pct[sl].mean(0))
        if not y:
            raise ValueError("no valid windows (sessions too short or all saturated)")
        self.pos = np.stack(pos).astype(np.float32)
        self.nrm = np.stack(nrm).astype(np.float32)
        q_arr, qd_arr = np.stack(q), np.stack(qd)
        # joint-state normalisation: fit here unless given (reuse train stats for val/test)
        self.joint_norm = joint_norm or (NormStats.fit(q_arr), NormStats.fit(qd_arr))
        self.q = self.joint_norm[0].apply(q_arr)
        self.qd = self.joint_norm[1].apply(qd_arr)
        self.y = np.stack(y).astype(np.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"pos": torch.from_numpy(self.pos[i]), "nrm": torch.from_numpy(self.nrm[i]),
                "q": torch.from_numpy(self.q[i]), "qd": torch.from_numpy(self.qd[i]),
                "y": torch.from_numpy(self.y[i])}
