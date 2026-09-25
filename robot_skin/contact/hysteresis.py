"""Per-taxel hysteresis + debouncing of a contact probability (or score) stream.

A frame-wise detector output flickers around its threshold; downstream consumers (pseudo labels,
the safety filter, the policy's binary tactile mode) want stable on/off intervals. Two thresholds
and two dwell counts per taxel, causal (no look-ahead)::

    OFF ──(p ≥ on_thr for min_on consecutive ticks)──▶ ON
    ON  ──(p < off_thr for min_off consecutive ticks)──▶ OFF

With ``off_thr ≤ on_thr`` values between the thresholds keep the current state. The switch happens
*at* the ``min_on``-th (``min_off``-th) qualifying tick, i.e. with a delay of ``min_on − 1``
(``min_off − 1``) ticks. Non-finite inputs count as below ``off_thr`` (fail-safe: no phantom contact).
:meth:`HysteresisFilter.step` (online, ``[N]`` per tick) and :meth:`HysteresisFilter.run`
(offline, ``[T,N]``) produce identical outputs.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["HysteresisFilter"]


class HysteresisFilter:
    """Two-threshold, dwell-debounced on/off state per taxel (module docstring)."""

    def __init__(self, on_thr: float = 0.6, off_thr: float = 0.4, min_on: int = 1, min_off: int = 1,
                 n_taxels: int | None = None):
        if not off_thr <= on_thr:
            raise ValueError(f"need off_thr ≤ on_thr, got {off_thr} > {on_thr}")
        if int(min_on) < 1 or int(min_off) < 1:
            raise ValueError("min_on and min_off must be ≥ 1")
        self.on_thr, self.off_thr = float(on_thr), float(off_thr)
        self.min_on, self.min_off = int(min_on), int(min_off)
        self.n_taxels = None if n_taxels is None else int(n_taxels)
        self.reset()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | "HysteresisFilter" | None) -> "HysteresisFilter":
        if isinstance(cfg, HysteresisFilter):
            return cls(**cfg.config)
        return cls(**dict(cfg or {}))

    @property
    def config(self) -> dict:
        return {"on_thr": self.on_thr, "off_thr": self.off_thr, "min_on": self.min_on, "min_off": self.min_off}

    def reset(self) -> None:
        """Back to OFF with empty dwell counters (the taxel count is re-learned on the next step
        unless fixed in the constructor)."""
        self.state: np.ndarray | None = None
        self._cnt_on: np.ndarray | None = None
        self._cnt_off: np.ndarray | None = None
        if self.n_taxels is not None:
            self._init(self.n_taxels)

    def _init(self, n: int) -> None:
        self.state = np.zeros(n, dtype=bool)
        self._cnt_on = np.zeros(n, dtype=np.int64)
        self._cnt_off = np.zeros(n, dtype=np.int64)

    def step(self, p: np.ndarray) -> np.ndarray:
        """One tick: ``p[N]`` → on/off ``[N]`` bool (copy)."""
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if self.state is None:
            self._init(p.shape[0])
        elif p.shape[0] != self.state.shape[0]:
            raise ValueError(f"expected {self.state.shape[0]} taxels, got {p.shape[0]}")
        fin = np.isfinite(p)
        hi = fin & (p >= self.on_thr)
        lo = ~fin | (p < self.off_thr)
        s = self.state
        self._cnt_on = np.where(~s & hi, self._cnt_on + 1, 0)
        self._cnt_off = np.where(s & lo, self._cnt_off + 1, 0)
        turn_on = ~s & (self._cnt_on >= self.min_on)
        turn_off = s & (self._cnt_off >= self.min_off)
        s[turn_on] = True
        s[turn_off] = False
        self._cnt_on[turn_on] = 0
        self._cnt_off[turn_off] = 0
        return s.copy()

    def run(self, p: np.ndarray, *, reset: bool = True) -> np.ndarray:
        """Offline over ``p[T,N]`` (or ``[T]``) → bool of the same shape. Resets first unless
        ``reset=False`` (then it continues from the current state)."""
        p = np.asarray(p, dtype=np.float64)
        one_d = p.ndim == 1
        if one_d:
            p = p[:, None]
        if p.ndim != 2:
            raise ValueError(f"p must be [T, N] or [T], got {p.shape}")
        if reset:
            self.reset()
        out = np.empty(p.shape, dtype=bool)
        for t in range(p.shape[0]):
            out[t] = self.step(p[t])
        return out[:, 0] if one_d else out
