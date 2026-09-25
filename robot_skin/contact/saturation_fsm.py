"""Per-taxel saturation state machine with recovery and re-zero.

States::

    OK ──sat──▶ SATURATED ──!sat──▶ RECOVERING ──|r−offset|≤ok_pct for ok_sec──▶ OK
                    ▲                    │ └──sat──▶ SATURATED
                    │                    └──> max_recover_s ──▶ OK (re-zero: offset ← r)

Defaults mirror the channel quarantine in ``deformable_sats/sats/inference/run_dashboard.py``
(±1.5 % for 2 s releases, accept a new baseline after 30 s): a taxel that went through dropout
emits large spurious residuals while it recovers, so it is *untrusted* until it settles, and if
it settles somewhere else that level becomes its new zero.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class SatState(IntEnum):
    OK = 0
    SATURATED = 1
    RECOVERING = 2


class SaturationFSM:
    def __init__(self, n_taxels: int, *, ok_pct: float = 1.5, ok_sec: float = 2.0,
                 max_recover_s: float = 30.0):
        self.n = n_taxels
        self.ok_pct, self.ok_sec, self.max_recover_s = ok_pct, ok_sec, max_recover_s
        self.reset()

    def reset(self) -> None:
        self.state = np.full(self.n, SatState.OK, dtype=np.int8)
        self.offset = np.zeros(self.n, dtype=np.float32)   # re-zero offset subtracted from residual
        self._stable_t = np.zeros(self.n)                  # time within ok band while RECOVERING
        self._recover_t = np.zeros(self.n)                 # time spent RECOVERING
        self.rezero_count = np.zeros(self.n, dtype=np.int64)

    @property
    def trusted(self) -> np.ndarray:
        return self.state == SatState.OK

    def corrected(self, resid_pct: np.ndarray) -> np.ndarray:
        return (np.asarray(resid_pct, np.float32) - self.offset).astype(np.float32)

    def step(self, resid_pct: np.ndarray, saturated: np.ndarray, dt: float) -> np.ndarray:
        r = np.asarray(resid_pct, dtype=np.float32)
        sat = np.asarray(saturated, dtype=bool)
        s = self.state
        # any state → SATURATED
        s[sat] = SatState.SATURATED
        # SATURATED & !sat → RECOVERING
        leave = (s == SatState.SATURATED) & ~sat
        s[leave] = SatState.RECOVERING
        self._stable_t[leave] = 0.0
        self._recover_t[leave] = 0.0
        # RECOVERING bookkeeping (taxels that entered this tick start counting next tick)
        rec = (s == SatState.RECOVERING) & ~leave
        self._recover_t[rec] += dt
        in_band = np.abs(r - self.offset) <= self.ok_pct
        self._stable_t[rec & in_band] += dt
        self._stable_t[rec & ~in_band] = 0.0
        released = rec & (self._stable_t >= self.ok_sec - 1e-9)
        s[released] = SatState.OK
        timeout = rec & ~released & (self._recover_t >= self.max_recover_s - 1e-9)
        self.offset[timeout] = r[timeout]
        self.rezero_count[timeout] += 1
        s[timeout] = SatState.OK
        return s.copy()

    def run(self, resid_pct: np.ndarray, saturated: np.ndarray, dt: float) -> np.ndarray:
        """Batch over time: ``[T,N]`` → states ``[T,N]``."""
        return np.stack([self.step(r, m, dt) for r, m in zip(resid_pct, saturated)])
