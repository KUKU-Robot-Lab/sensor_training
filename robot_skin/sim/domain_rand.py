"""Per-taxel domain randomisation of simulated ΔS (gain / offset / saturation / noise / dropout).

Real taxels differ in gain (±30 % is typical across a hand), sit on slightly different
baselines, and saturate at different levels. Simulated clean ΔS (SATS sign, press → negative)
is mapped to an observation::

    obs = clip(gain * clean + offset + noise, -sat, +sat);   saturated = |…| ≥ sat
    dropout taxels read -100 % and are saturated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaxelDomainParams:
    gain: np.ndarray      # [N]
    offset: np.ndarray    # [N] %
    sat: np.ndarray       # [N] % (positive)
    dropout: np.ndarray   # [N] bool


class TaxelDomainRandomizer:
    def __init__(self, *, gain_range=(0.7, 1.3), offset_pct_range=(-1.0, 1.0),
                 sat_pct_range=(40.0, 90.0), noise_pct_std: float = 0.3, dropout_prob: float = 0.0,
                 seed: int | None = None):
        self.gain_range, self.offset_range, self.sat_range = gain_range, offset_pct_range, sat_pct_range
        self.noise_std, self.dropout_prob = noise_pct_std, dropout_prob
        self.rng = np.random.default_rng(seed)

    def sample(self, n_taxels: int) -> TaxelDomainParams:
        r = self.rng
        return TaxelDomainParams(
            gain=r.uniform(*self.gain_range, n_taxels).astype(np.float32),
            offset=r.uniform(*self.offset_range, n_taxels).astype(np.float32),
            sat=r.uniform(*self.sat_range, n_taxels).astype(np.float32),
            dropout=r.random(n_taxels) < self.dropout_prob)

    def apply(self, clean_pct: np.ndarray, params: TaxelDomainParams) -> tuple[np.ndarray, np.ndarray]:
        """``clean_pct[...,N]`` → ``(obs_pct, saturated)`` of the same shape."""
        x = params.gain * np.asarray(clean_pct, dtype=np.float32) + params.offset
        if self.noise_std > 0:
            x = x + self.rng.normal(0.0, self.noise_std, x.shape).astype(np.float32)
        saturated = np.abs(x) >= params.sat
        x = np.clip(x, -params.sat, params.sat)
        x = np.where(params.dropout, np.float32(-100.0), x)
        saturated = saturated | np.broadcast_to(params.dropout, x.shape)
        return x.astype(np.float32), saturated
