"""Pressure-signal conventions shared by every tactile package.

Relative change ΔS [%]
----------------------
Identical to the SATS training convention (``deformable_sats/sats/training/dataset.py``)::

    ΔS = (raw - baseline) / baseline * 100

The mk555 barometric taxels *decrease* in raw value when pressed, so a press shows up as
**negative** ΔS (dropout goes toward -100 %). :data:`PRESS_SIGN` (= -1) captures that, and
:func:`press_intensity` returns the press-positive view (``PRESS_SIGN * ΔS``) that the
contact / ordinal code in ``robot_skin`` works with. Keeping ΔS itself in the SATS sign means
tensors can be fed to a frozen SATS model without any flip.

Also here: baseline estimation from an initial no-contact segment, per-taxel normalisation
(:class:`NormStats`) and :func:`saturation_mask`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Sign of ΔS under a press for mk555 barometric taxels (raw drops when pressed).
PRESS_SIGN: float = -1.0

#: Default ADC rails for the 24-bit barometric front-end (raw counts).
ADC_MIN: float = 0.0
ADC_MAX: float = float(2**24 - 1)


def relative_change(raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """ΔS [%] = (raw - baseline) / baseline * 100, broadcast over the last (taxel) axis.

    ``raw`` is ``[..., N]`` and ``baseline`` ``[N]`` (or broadcastable). Returns float32,
    matching ``sats/training/dataset.py``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if np.any(base == 0):
        raise ValueError("baseline contains zeros; relative change is undefined")
    return (((raw - base) / base) * 100.0).astype(np.float32)


def press_intensity(delta_pct: np.ndarray) -> np.ndarray:
    """Press-positive view of ΔS: ``PRESS_SIGN * ΔS`` (> 0 when pressed)."""
    return (PRESS_SIGN * np.asarray(delta_pct, dtype=np.float32)).astype(np.float32)


def estimate_baseline(
    raw: np.ndarray,
    *,
    t: np.ndarray | None = None,
    duration_s: float | None = 1.0,
    n_samples: int | None = None,
) -> np.ndarray:
    """Per-taxel baseline = median over the initial no-contact segment.

    The segment is either the first ``n_samples`` rows, or (with timestamps ``t``) all rows with
    ``t - t[0] < duration_s``. Median (not mean) so a stray early touch does not bias it.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError(f"raw must be non-empty [T, N], got {raw.shape}")
    if n_samples is not None:
        seg = raw[: max(1, int(n_samples))]
    elif t is not None and duration_s is not None:
        t = np.asarray(t, dtype=np.float64)
        if t.shape[0] != raw.shape[0]:
            raise ValueError("t and raw length mismatch")
        seg = raw[(t - t[0]) < float(duration_s)]
        if seg.shape[0] == 0:
            seg = raw[:1]
    else:
        raise ValueError("give n_samples, or t together with duration_s")
    return np.median(seg, axis=0)


@dataclass(frozen=True)
class NormStats:
    """Per-taxel affine normalisation ``x_norm = (x - offset) / scale``.

    Fit on training data only and ship alongside the model (``to_dict``/``save``).
    """

    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, *, method: str = "std", eps: float = 1e-6) -> "NormStats":
        """Fit over all leading axes of ``[..., N]``. ``method``: ``std`` | ``robust`` | ``none``."""
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(-1, x.shape[-1])
        if method == "std":
            off, sc = flat.mean(0), flat.std(0)
        elif method == "robust":
            off = np.median(flat, 0)
            q75, q25 = np.percentile(flat, [75, 25], axis=0)
            sc = (q75 - q25) / 1.349  # IQR → σ for a Gaussian
        elif method == "none":
            off, sc = np.zeros(flat.shape[1]), np.ones(flat.shape[1])
        else:
            raise ValueError(f"unknown method {method!r}")
        return cls(offset=off.astype(np.float32), scale=(np.maximum(sc, 0) + eps).astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.offset) / self.scale).astype(np.float32)

    def invert(self, x_norm: np.ndarray) -> np.ndarray:
        return (np.asarray(x_norm, dtype=np.float32) * self.scale + self.offset).astype(np.float32)

    def to_dict(self) -> dict:
        return {"offset": self.offset.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(offset=np.asarray(d["offset"], dtype=np.float32),
                   scale=np.asarray(d["scale"], dtype=np.float32))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text()))


def saturation_mask(
    raw: np.ndarray | None = None,
    delta_pct: np.ndarray | None = None,
    *,
    adc_min: float = ADC_MIN,
    adc_max: float = ADC_MAX,
    rail_margin: float = 0.0,
    max_abs_pct: float | None = 90.0,
) -> np.ndarray:
    """Boolean mask (True = saturated / untrustworthy) of the same shape as the inputs.

    A sample is saturated if its raw value sits on (or within ``rail_margin`` of) an ADC rail,
    or if ``|ΔS|`` reaches ``max_abs_pct`` (e.g. dropout toward -100 %). Either input may be
    omitted; at least one is required.
    """
    if raw is None and delta_pct is None:
        raise ValueError("need raw and/or delta_pct")
    mask = None
    if raw is not None:
        r = np.asarray(raw, dtype=np.float64)
        mask = (r <= adc_min + rail_margin) | (r >= adc_max - rail_margin)
    if delta_pct is not None and max_abs_pct is not None:
        d = np.abs(np.asarray(delta_pct, dtype=np.float64)) >= float(max_abs_pct)
        mask = d if mask is None else (mask | d)
    if mask is None:  # delta only, cap disabled
        mask = np.zeros(np.shape(delta_pct), dtype=bool)
    return mask
