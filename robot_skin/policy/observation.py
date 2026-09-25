"""Policy observation builders for the tactile ablation: full / ordinal / binary / none.

Same builder in sim and on hardware, so the ablation isolates *what* tactile information the
policy gets:

- ``full``    : continuous residual ΔS (press-positive, /scale) + saturated flag  → 2 per taxel
- ``ordinal`` : one-hot {none, weak, strong, saturated}                        → 4 per taxel
- ``binary``  : contact mask (weak or above, excluding saturated)             → 1 per taxel
- ``none``    : no tactile features (proprioception only)                      → 0
"""
from __future__ import annotations

import numpy as np

from common.signal import press_intensity

from ..contact.ordinal import N_LEVELS, ContactLevel, OrdinalQuantizer

OBS_MODES = ("full", "ordinal", "binary", "none")
_PER_TAXEL = {"full": 2, "ordinal": N_LEVELS, "binary": 1, "none": 0}


def tactile_features(mode: str, resid_pct: np.ndarray, saturated: np.ndarray | None,
                     quantizer: OrdinalQuantizer, *, scale_pct: float = 20.0) -> np.ndarray:
    """``resid_pct[...,N]`` → ``[..., N*k]`` float32 features for ``mode``."""
    if mode not in OBS_MODES:
        raise ValueError(f"mode must be one of {OBS_MODES}, got {mode!r}")
    r = np.asarray(resid_pct, dtype=np.float32)
    sat = np.zeros(r.shape, bool) if saturated is None else np.asarray(saturated, dtype=bool)
    lead = r.shape[:-1]
    if mode == "none":
        return np.zeros(lead + (0,), dtype=np.float32)
    if mode == "full":
        f = np.stack([np.where(sat, 0.0, press_intensity(r) / scale_pct), sat.astype(np.float32)], -1)
    else:
        lv = quantizer(r, sat)
        if mode == "ordinal":
            f = OrdinalQuantizer.one_hot(lv)
        else:
            f = ((lv == ContactLevel.WEAK) | (lv == ContactLevel.STRONG)).astype(np.float32)[..., None]
    return f.reshape(lead + (-1,)).astype(np.float32)


class ObservationBuilder:
    def __init__(self, mode: str, n_taxels: int, proprio_dim: int,
                 quantizer: OrdinalQuantizer | None = None, *, scale_pct: float = 20.0):
        if mode not in OBS_MODES:
            raise ValueError(f"mode must be one of {OBS_MODES}, got {mode!r}")
        self.mode, self.n_taxels, self.proprio_dim = mode, n_taxels, proprio_dim
        self.quantizer = quantizer or OrdinalQuantizer()
        self.scale_pct = scale_pct

    @property
    def dim(self) -> int:
        return self.proprio_dim + _PER_TAXEL[self.mode] * self.n_taxels

    def __call__(self, proprio: np.ndarray, resid_pct: np.ndarray,
                 saturated: np.ndarray | None = None) -> np.ndarray:
        tac = tactile_features(self.mode, resid_pct, saturated, self.quantizer, scale_pct=self.scale_pct)
        obs = np.concatenate([np.asarray(proprio, dtype=np.float32), tac], axis=-1)
        assert obs.shape[-1] == self.dim
        return obs
