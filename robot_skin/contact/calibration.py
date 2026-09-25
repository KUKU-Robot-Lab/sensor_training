"""ResidualCalibrator — baseline residual (ΔS %) → calibrated press z-score → contact level.

After the temporal baseline stage, ``residual = ΔS − baseline_pred`` (SATS sign: press → negative)
still carries per-taxel noise, drift and model error that differ between taxels. The calibrator is
fitted on **no-contact** frames (D1 validation split: data the baseline was not trained on) and
turns the residual into a press-positive z-score with unit robust spread on no-contact data::

    p   = press_intensity(residual) − c          c = median of p on no-contact (per taxel)
    σ_t = g · sqrt(σ² + exp(logvar_t))            with the baseline's predicted variance, or
    σ_t = σ                                        without it (use_logvar=False)
    z   = p / σ_t

``σ`` is the robust per-taxel spread (MAD · 1.4826, a Gaussian-consistent σ estimate) of what the
predicted variance does not explain — ``σ² = max(MADσ(p)² − median(exp(logvar)), 0) + floor²``
(without a predicted variance ``σ = max(MADσ(p), floor)``) — and
``g`` rescales so that ``MADσ(z) = 1`` on the calibration frames (the NLL-trained σ of
``baseline.temporal`` — Kendall & Gal, NeurIPS 2017 — is typically over- or under-confident on held-
out data; one scalar per taxel fixes that). Levels follow :class:`~robot_skin.contact.OrdinalQuantizer`
semantics (press-positive, NONE < WEAK < STRONG, SATURATED overrides) with thresholds in z units
**and** a % floor, so a taxel with a tiny σ cannot report contact for a 0.05 % wiggle::

    WEAK   ⇔ z ≥ weak_z   and p ≥ weak_floor_pct
    STRONG ⇔ z ≥ strong_z and p ≥ strong_floor_pct
    SATURATED where the taxel is saturated (rails / |ΔS| ≥ 90 %) or untrusted by the optional
    :class:`~robot_skin.contact.SaturationFSM` gate (``fsm`` config, recovery after dropouts)

The whole state (σ, c, g, thresholds, FSM config) is one JSON dict (:meth:`to_dict`) that the
online processor loads, so offline ``residual_z`` / ``contact_level`` and deployment use exactly the
same numbers (:func:`residual_levels` is the single offline entry point; per tick call
:meth:`transform` / :meth:`levels` on ``[N]`` vectors and step a ``SaturationFSM`` with the same
config).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.signal import press_intensity

from .ordinal import ContactLevel
from .saturation_fsm import SaturationFSM, SatState

__all__ = ["MAD_TO_SIGMA", "CALIBRATOR_FORMAT", "robust_sigma", "ResidualCalibrator", "saturation_gate",
           "residual_levels"]

#: MAD → σ for a Gaussian (1 / Φ⁻¹(3/4))
MAD_TO_SIGMA = 1.4826
CALIBRATOR_FORMAT = "robot_skin.contact.calibration/1"
_FSM_KEYS = ("enabled", "ok_pct", "ok_sec", "max_recover_s")


def robust_sigma(x: np.ndarray, valid: np.ndarray | None = None, axis: int = 0, *,
                 center: np.ndarray | float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column robust spread of ``x[T,N]`` over ``valid`` rows: ``(sigma = 1.4826·MAD, median,
    count)``; ``center`` replaces the median as the MAD centre. Columns without samples → NaN."""
    x = np.asarray(x, dtype=np.float64)
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
    m = np.isfinite(x) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(x))
    xs = np.where(m, x, np.nan)
    cnt = m.sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # all-NaN columns → NaN
        med = np.nanmedian(xs, axis=0)
        c = med if center is None else np.broadcast_to(np.asarray(center, dtype=np.float64), med.shape)
        mad = np.nanmedian(np.abs(xs - c), axis=0)
    return MAD_TO_SIGMA * mad, med, cnt


@dataclass
class ResidualCalibrator:
    """Per-taxel residual → z calibration (see module docstring). Build with :meth:`fit`."""

    sigma: np.ndarray                      # [N] % — spread not explained by the predicted variance
    center: np.ndarray                     # [N] % — press-positive no-contact median
    gain: np.ndarray                       # [N] — z rescale so that MADσ(z) = 1 on calibration data
    use_logvar: bool = False
    weak_z: float = 3.0
    strong_z: float = 8.0
    weak_floor_pct: float = 0.5
    strong_floor_pct: float = 3.0
    fsm: dict | None = None                # SaturationFSM gate: {enabled, ok_pct, ok_sec, max_recover_s}
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=np.float64).reshape(-1)
        self.center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        self.gain = np.asarray(self.gain, dtype=np.float64).reshape(-1)
        n = self.sigma.shape[0]
        if self.center.shape != (n,) or self.gain.shape != (n,):
            raise ValueError("sigma, center and gain must all be [N]")
        if not (np.all(np.isfinite(self.sigma)) and np.all(self.sigma >= 0) and np.all(np.isfinite(self.center))
                and np.all(np.isfinite(self.gain)) and np.all(self.gain > 0)):
            raise ValueError("sigma ≥ 0, gain > 0 and center must be finite")
        if not self.use_logvar and np.any(self.sigma <= 0):
            raise ValueError("sigma must be > 0 without a predicted variance")
        if not (0 < self.weak_z < self.strong_z):
            raise ValueError("need 0 < weak_z < strong_z")
        if not (0 <= self.weak_floor_pct <= self.strong_floor_pct):
            raise ValueError("need 0 ≤ weak_floor_pct ≤ strong_floor_pct")
        if self.fsm is not None:
            bad = set(self.fsm) - set(_FSM_KEYS)
            if bad:
                raise ValueError(f"unknown fsm keys {sorted(bad)}; valid: {_FSM_KEYS}")

    @property
    def n_taxels(self) -> int:
        return int(self.sigma.shape[0])

    # ── fitting ───────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, residual: np.ndarray, valid: np.ndarray | None = None, logvar: np.ndarray | None = None, *,
            weak_z: float = 3.0, strong_z: float = 8.0, weak_floor_pct: float = 0.5,
            strong_floor_pct: float = 3.0, sigma_floor_pct: float = 0.02, center: bool = True,
            min_samples: int = 50, fsm: Mapping[str, Any] | None = None) -> "ResidualCalibrator":
        """Fit on ``residual[T,N]`` (ΔS %, SATS sign) where ``valid[T,N]`` (no-contact, not
        saturated). With ``logvar[T,N]`` (the baseline's predicted log-variance) the z uses the
        predicted σ too. Taxels with fewer than ``min_samples`` valid samples get the median σ /
        gain of the others and centre 0 (listed in ``info["fallback_taxels"]``)."""
        r = np.asarray(residual, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError(f"residual must be [T, N], got {r.shape}")
        T, N = r.shape
        m = np.ones((T, N), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if m.shape != (T, N):
            raise ValueError(f"valid must be {(T, N)}, got {m.shape}")
        p = press_intensity(r).astype(np.float64)
        m = m & np.isfinite(p)
        v = None
        if logvar is not None:
            lv = np.asarray(logvar, dtype=np.float64)
            if lv.shape != (T, N):
                raise ValueError(f"logvar must be {(T, N)}, got {lv.shape}")
            v = np.exp(np.clip(lv, -30.0, 30.0))
            m = m & np.isfinite(lv)
        cnt = m.sum(0)
        ok = cnt >= int(min_samples)
        if not ok.any():
            raise ValueError(f"no taxel has ≥ {min_samples} valid no-contact samples to calibrate on")
        s_raw, med, _ = robust_sigma(p, m)
        c = np.where(ok & bool(center), med, 0.0)
        floor = float(sigma_floor_pct)
        if v is None:
            sigma = np.maximum(np.where(ok, s_raw, np.nan), floor)
            gain = np.ones(N)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v_med = np.nanmedian(np.where(m, v, np.nan), axis=0)
            sigma = np.sqrt(np.maximum(s_raw ** 2 - v_med, 0.0) + floor ** 2)
            z0 = (p - c) / np.sqrt(sigma ** 2 + v)
            g, _, _ = robust_sigma(z0, m, center=0.0)
            gain = np.where(ok, np.maximum(g, 1e-3), np.nan)
            sigma = np.where(ok, sigma, np.nan)
        fb = np.flatnonzero(~ok)
        if fb.size:
            sigma[fb] = np.nanmedian(sigma[ok])
            gain[fb] = np.nanmedian(gain[ok])
        info = {"n_samples": [int(x) for x in cnt], "fallback_taxels": [int(i) for i in fb],
                "raw_sigma_pct": [float(x) if np.isfinite(x) else None for x in s_raw]}
        return cls(sigma=sigma, center=c, gain=gain, use_logvar=v is not None, weak_z=float(weak_z),
                   strong_z=float(strong_z), weak_floor_pct=float(weak_floor_pct),
                   strong_floor_pct=float(strong_floor_pct), fsm=dict(fsm) if fsm else None, info=info)

    # ── transform ─────────────────────────────────────────────────────────
    def _check(self, x: np.ndarray, what: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.n_taxels:
            raise ValueError(f"{what} has {x.shape[-1]} taxels, calibrator {self.n_taxels}")
        return x

    def press(self, residual: np.ndarray) -> np.ndarray:
        """Centred press intensity ``p = −residual − c`` (%) ``[..., N]``, float32."""
        r = self._check(residual, "residual")
        return (press_intensity(r).astype(np.float64) - self.center).astype(np.float32)

    def scale(self, logvar: np.ndarray | None = None) -> np.ndarray:
        """Effective σ (%) — ``[N]`` without a predicted variance, else ``[..., N]``."""
        if not self.use_logvar:
            return (self.gain * self.sigma).astype(np.float32)
        if logvar is None:
            raise ValueError("this calibrator was fitted with the baseline log-variance: pass logvar")
        lv = self._check(logvar, "logvar")
        return (self.gain * np.sqrt(self.sigma ** 2 + np.exp(np.clip(lv, -30.0, 30.0)))).astype(np.float32)

    def transform(self, residual: np.ndarray, logvar: np.ndarray | None = None) -> np.ndarray:
        """Residual ``[..., N]`` (ΔS %, SATS sign) → press-positive z ``[..., N]`` float32 (non-finite
        residuals stay non-finite)."""
        return (self.press(residual).astype(np.float64) / self.scale(logvar)).astype(np.float32)

    def levels(self, z: np.ndarray, saturated: np.ndarray | None = None, *,
               press_pct: np.ndarray | None = None) -> np.ndarray:
        """z ``[..., N]`` → ``ContactLevel`` int8 (module docstring). The % floors apply when the
        centred press ``press_pct`` (:meth:`press`) is given; non-finite z → NONE (unless saturated)."""
        z = self._check(z, "z")
        ok = np.isfinite(z)
        weak = ok & (z >= self.weak_z)
        strong = ok & (z >= self.strong_z)
        if press_pct is not None:
            p = self._check(press_pct, "press_pct")
            weak &= p >= self.weak_floor_pct
            strong &= p >= self.strong_floor_pct
        lv = np.zeros(z.shape, dtype=np.int8)
        lv[weak] = ContactLevel.WEAK
        lv[strong] = ContactLevel.STRONG
        if saturated is not None:
            lv[np.broadcast_to(np.asarray(saturated, dtype=bool), z.shape)] = ContactLevel.SATURATED
        return lv

    def levels_from_residual(self, residual: np.ndarray, logvar: np.ndarray | None = None,
                             saturated: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(z, levels)`` of a residual (floors applied)."""
        z = self.transform(residual, logvar)
        return z, self.levels(z, saturated, press_pct=self.press(residual))

    # ── io ────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"format": CALIBRATOR_FORMAT, "sigma": [float(x) for x in self.sigma],
                "center": [float(x) for x in self.center], "gain": [float(x) for x in self.gain],
                "use_logvar": bool(self.use_logvar), "weak_z": float(self.weak_z), "strong_z": float(self.strong_z),
                "weak_floor_pct": float(self.weak_floor_pct), "strong_floor_pct": float(self.strong_floor_pct),
                "fsm": None if self.fsm is None else dict(self.fsm), "info": dict(self.info)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResidualCalibrator":
        d = dict(d)
        fmt = d.pop("format", CALIBRATOR_FORMAT)
        if fmt != CALIBRATOR_FORMAT:
            raise ValueError(f"not a {CALIBRATOR_FORMAT} dict (format {fmt!r})")
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ResidualCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def saturation_gate(residual: np.ndarray, saturated: np.ndarray, dt: float, *, ok_pct: float = 1.5,
                    ok_sec: float = 2.0, max_recover_s: float = 30.0, enabled: bool = True
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run :class:`SaturationFSM` over ``residual[T,N]`` / ``saturated[T,N]`` →
    ``(corrected_residual, untrusted[T,N], states[T,N])``: ``corrected = residual − offset``
    (re-zero after a timeout) and ``untrusted`` = state ≠ OK (saturated or still recovering). The
    online processor steps the same FSM per tick with the same config (``calibrator.fsm``), feeding
    ``nan_to_num(residual)`` to ``step`` and the raw residual to ``corrected`` as done here."""
    r = np.asarray(residual, dtype=np.float32)
    sat = np.asarray(saturated, dtype=bool)
    if not enabled:
        return r.copy(), sat.copy(), np.where(sat, SatState.SATURATED, SatState.OK).astype(np.int8)
    fsm = SaturationFSM(r.shape[1], ok_pct=ok_pct, ok_sec=ok_sec, max_recover_s=max_recover_s)
    corr = np.empty_like(r)
    states = np.empty(r.shape, dtype=np.int8)
    for t in range(r.shape[0]):
        rt = np.nan_to_num(r[t], nan=0.0)
        states[t] = fsm.step(rt, sat[t], dt)
        corr[t] = fsm.corrected(r[t])
    return corr, states != SatState.OK, states


def residual_levels(calibrator: ResidualCalibrator, residual: np.ndarray, saturated: np.ndarray | None = None,
                    logvar: np.ndarray | None = None, *, dt: float | None = None) -> dict[str, np.ndarray]:
    """The offline contact pipeline of one episode: optional FSM gate (``calibrator.fsm`` with
    ``enabled``; needs ``dt``) → z → levels. Returns ``{"residual_z" [T,N] float32, "contact_level"
    [T,N] int8, "untrusted" [T,N] bool, "press_pct" [T,N]}``."""
    r = np.asarray(residual, dtype=np.float32)
    sat = np.zeros(r.shape, dtype=bool) if saturated is None else np.asarray(saturated, dtype=bool)
    fcfg = dict(calibrator.fsm or {})
    if fcfg.get("enabled", False):
        if dt is None:
            raise ValueError("the calibrator's FSM gate needs dt")
        r, untrusted, _ = saturation_gate(r, sat, float(dt), **fcfg)
    else:
        untrusted = sat.copy()
    z = calibrator.transform(r, logvar if calibrator.use_logvar else None)
    p = calibrator.press(r)
    return {"residual_z": z, "contact_level": calibrator.levels(z, untrusted, press_pct=p),
            "untrusted": untrusted, "press_pct": p}
