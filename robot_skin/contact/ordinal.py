"""OrdinalQuantizer: residual → {NONE, WEAK, STRONG, SATURATED}.

Coarse levels are robust to per-taxel gain spread (domain randomisation targets exactly this)
and are what the ``ordinal`` policy observation uses.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from common.signal import press_intensity


class ContactLevel(IntEnum):
    NONE = 0
    WEAK = 1
    STRONG = 2
    SATURATED = 3


N_LEVELS = len(ContactLevel)


class OrdinalQuantizer:
    def __init__(self, weak_pct: float | np.ndarray = 3.0, strong_pct: float | np.ndarray = 15.0):
        self.weak = np.asarray(weak_pct, dtype=np.float32)
        self.strong = np.asarray(strong_pct, dtype=np.float32)
        if np.any(self.strong <= self.weak) or np.any(self.weak <= 0):
            raise ValueError("need 0 < weak_pct < strong_pct")

    def __call__(self, resid_pct: np.ndarray, saturated: np.ndarray | None = None) -> np.ndarray:
        p = press_intensity(resid_pct)
        lv = np.zeros(p.shape, dtype=np.int8)
        lv[p >= self.weak] = ContactLevel.WEAK
        lv[p >= self.strong] = ContactLevel.STRONG
        if saturated is not None:
            lv[np.asarray(saturated, dtype=bool)] = ContactLevel.SATURATED
        return lv

    @staticmethod
    def one_hot(levels: np.ndarray) -> np.ndarray:
        return np.eye(N_LEVELS, dtype=np.float32)[np.asarray(levels, dtype=np.int64)]
