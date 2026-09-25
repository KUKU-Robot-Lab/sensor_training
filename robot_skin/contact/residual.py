"""Residual = observed ΔS − predicted no-contact ΔS, and binary contact masks."""
from __future__ import annotations

import numpy as np

from common.signal import press_intensity


def residual(observed_pct: np.ndarray, predicted_pct: np.ndarray) -> np.ndarray:
    """ΔS residual in SATS sign (press → negative)."""
    return (np.asarray(observed_pct, np.float32) - np.asarray(predicted_pct, np.float32)).astype(np.float32)


def contact_mask(resid_pct: np.ndarray, threshold_pct: float | np.ndarray = 3.0,
                 trusted: np.ndarray | None = None) -> np.ndarray:
    """True where press intensity (−residual) ≥ threshold and (optionally) the taxel is trusted."""
    m = press_intensity(resid_pct) >= np.asarray(threshold_pct, dtype=np.float32)
    if trusted is not None:
        m &= np.asarray(trusted, dtype=bool)
    return m
