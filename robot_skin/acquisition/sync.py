"""Cross-stream clock sync: tap-event envelopes, cross-correlation offsets, drift, 3-tap procedure.

Every stream is stamped on the host monotonic clock on arrival, but each device adds its own
latency (serial buffering, camera exposure → USB, IMU radio) and, for device-stamped streams, a
slowly drifting clock. The **3-tap sync** makes one sharp physical event visible to every
sensor at once:

1. At session start (block ``sync_start``) the subject, hand still, taps the sync pad with the
   index fingertip three times in a short–long rhythm (default gaps 0.6 s / 1.2 s) in view of all
   cameras. Each tap is a pressure transient (fingertip taxel), an IMU acceleration spike (impact)
   and a burst of image motion.
2. Repeat at the end (``sync_end``). Two windows give offset **and** drift; one gives offset only.

:func:`change_envelope` turns any stream into a rate-independent "activity" signal (per-channel
robustly scaled ``|dx/dt|`` summed over channels, resampled to ``hz`` and Gaussian-smoothed);
:func:`estimate_offset` cross-correlates two envelopes inside a window (normalised correlation,
parabolic sub-sample peak). The unequal tap spacing makes the correlation peak unique for lags up
to the shorter gap. :func:`fit_clock_drift` fits ``t_ref = scale·t + offset`` through the window
estimates; :func:`sync_session` runs it for every stream against the reference (``pressure``),
rewrites the timestamps in place (originals kept as ``t_host`` / ``timestamps_host.npy``, so it is
idempotent) and records the result in ``manifest.calibration["sync"]``.

Convention: ``offset_s`` is **added** to stream b's timestamps to align them with stream a
(a camera with +45 ms latency gets ≈ −0.045 s).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .manifest import SessionManifest
from .recorder import load_events, phases_from_events

__all__ = [
    "ClockModel", "DRIFT_MIN_DELTA_S", "OffsetEstimate", "SMOOTH_S_BY_KIND", "SYNC_KEY", "SYNCABLE_KINDS", "apply_clock_models", "apply_offset",
    "change_envelope", "detect_taps", "estimate_offset", "fit_clock_drift", "load_sync_signal", "stream_kind",
    "sync_session", "sync_windows",
]

log = logging.getLogger(__name__)

SYNC_KEY = "sync"
DEFAULT_HZ = 200.0
#: streams that carry a tap signature (hand_pose / object_pose are offline products on the
#: reference clock and are never shifted)
SYNCABLE_KINDS = ("pressure", "imu", "camera", "joint_state")
MAX_ABS_DRIFT = 1e-3          # |scale − 1| above this (1000 ppm) is rejected as implausible
#: envelope smoothing σ (s) per stream kind; a pair uses the larger one. Pressure edges and IMU
#: impact/lift-off spikes have the same shape → fine scale. Cameras (30 Hz) see the hand *motion*
#: around a tap, not the contact edges → compare at tap scale (blob centres; ≤ 1 frame error).
SMOOTH_S_BY_KIND = {"pressure": 0.02, "imu": 0.02, "joint_state": 0.03, "camera": 0.06}
#: per-window offset precision per kind; drift is fitted only when the start → end offset change
#: exceeds it (host-stamped streams do not drift, and fitting noise would tilt the clock model)
DRIFT_MIN_DELTA_S = {"pressure": 0.005, "imu": 0.005, "joint_state": 0.008, "camera": 0.034}


# ── value types ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OffsetEstimate:
    offset_s: float          # add to stream-b timestamps to align with stream a
    score: float             # normalised cross-correlation at the peak (−1 … 1)
    lag_samples: float       # peak lag in grid samples (sub-sample refined)
    at_limit: bool = False   # peak on the ±max_lag boundary → not trustworthy

    def __float__(self) -> float:
        return float(self.offset_s)


@dataclass(frozen=True)
class ClockModel:
    """Linear clock map ``t_ref = scale · t + offset_s``."""

    scale: float = 1.0
    offset_s: float = 0.0

    def apply(self, t):
        return self.scale * np.asarray(t, dtype=np.float64) + self.offset_s

    def inverse(self, t_ref):
        return (np.asarray(t_ref, dtype=np.float64) - self.offset_s) / self.scale

    @property
    def drift_ppm(self) -> float:
        return (self.scale - 1.0) * 1e6

    def to_dict(self) -> dict:
        return {"scale": float(self.scale), "offset_s": float(self.offset_s)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ClockModel":
        return cls(float(d.get("scale", 1.0)), float(d.get("offset_s", 0.0)))


def apply_offset(t, offset_s: float) -> np.ndarray:
    """Shift timestamps by a constant ``offset_s`` (the :func:`estimate_offset` convention)."""
    return np.asarray(t, dtype=np.float64) + float(offset_s)


# ── envelopes ────────────────────────────────────────────────────────────────
def _gauss_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0.3 or x.size < 3:
        return x
    r = int(math.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def change_envelope(t, x, *, hz: float = DEFAULT_HZ, t_range: tuple[float, float] | None = None,
                    smooth_s: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activity envelope of a multichannel stream on a uniform grid.

    ``x[T, ...]`` (channels flattened) → per-channel ``|Δx/Δt|`` scaled by its robust spread
    (MAD; constant channels ignored), summed over channels, linearly resampled onto
    ``grid = t_range[0] : 1/hz : t_range[1]`` and Gaussian-smoothed (σ = ``smooth_s``).
    Returns ``(grid, env, support)`` — ``support`` marks grid points inside the stream's data.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    X = np.asarray(x, dtype=np.float64).reshape(t.shape[0], -1)
    ok = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    t, X = t[ok], X[ok]
    order = np.argsort(t, kind="stable")
    t, X = t[order], X[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, X = t[keep], X[keep]
    if t.size < 3:
        raise ValueError("need at least 3 distinct samples for an envelope")
    dx = np.abs(np.diff(X, axis=0) / np.diff(t)[:, None])
    tm = 0.5 * (t[1:] + t[:-1])
    med = np.median(dx, axis=0)
    s = 1.4826 * np.median(np.abs(dx - med), axis=0)
    s = np.where(s > 0, s, dx.std(axis=0))
    use = s > 0
    e = (dx[:, use] / s[use]).sum(1) / max(int(use.sum()), 1) if use.any() else np.zeros(tm.shape)
    lo, hi = (tm[0], tm[-1]) if t_range is None else (float(t_range[0]), float(t_range[1]))
    if hi <= lo:
        raise ValueError("empty t_range")
    grid = lo + np.arange(int(math.floor((hi - lo) * hz)) + 1) / hz
    support = (grid >= tm[0]) & (grid <= tm[-1])
    env = np.where(support, np.interp(grid, tm, e), 0.0)
    return grid, _gauss_smooth(env, smooth_s * hz), support


def _zscore(env: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(env)
    if support.sum() < 3:
        return out
    v = env[support]
    mu, sd = v.mean(), v.std()
    out[support] = (v - mu) / sd if sd > 0 else 0.0
    return out


def estimate_offset(t_a, x_a, t_b, x_b, max_lag_s: float = 0.5, hz: float = DEFAULT_HZ, *,
                    window: tuple[float, float] | None = None, smooth_s: float = 0.02,
                    envelope: bool = True) -> OffsetEstimate:
    """Offset that aligns stream b to stream a by cross-correlating their event envelopes.

    ``x_*`` are raw signals ``[T, ...]`` (``envelope=True``: :func:`change_envelope` is applied)
    or precomputed envelopes ``[T]`` (``envelope=False``). ``window`` (a's clock) restricts the
    comparison to e.g. a sync block; b is taken over the window ± ``max_lag_s``. Returns an
    :class:`OffsetEstimate` (``t_b + offset_s`` ≈ ``t_a`` for the same physical event).
    """
    if not max_lag_s > 0 or not hz > 0:
        raise ValueError("max_lag_s and hz must be > 0")
    ta = np.asarray(t_a, dtype=np.float64).reshape(-1)
    tb = np.asarray(t_b, dtype=np.float64).reshape(-1)
    w0, w1 = (max(ta.min(), tb.min()), min(ta.max(), tb.max())) if window is None else map(float, window)
    if w1 <= w0:
        raise ValueError("streams/window do not overlap")
    lo, hi = w0 - max_lag_s, w1 + max_lag_s
    if envelope:
        ma = (ta >= w0 - 1.0 / hz * 4) & (ta <= w1 + 1.0 / hz * 4)
        mb = (tb >= lo - 4.0 / hz) & (tb <= hi + 4.0 / hz)
        if ma.sum() < 3 or mb.sum() < 3:
            raise ValueError("too few samples inside the sync window")
        _, ea, sa = change_envelope(ta[ma], np.asarray(x_a)[ma], hz=hz, t_range=(lo, hi), smooth_s=smooth_s)
        grid, eb, sb = change_envelope(tb[mb], np.asarray(x_b)[mb], hz=hz, t_range=(lo, hi), smooth_s=smooth_s)
        sa &= (grid >= w0) & (grid <= w1)
    else:
        grid = lo + np.arange(int(math.floor((hi - lo) * hz)) + 1) / hz
        xa, xb = np.asarray(x_a, dtype=np.float64).reshape(-1), np.asarray(x_b, dtype=np.float64).reshape(-1)
        ea = np.interp(grid, ta, xa, left=0.0, right=0.0)
        eb = np.interp(grid, tb, xb, left=0.0, right=0.0)
        sa = (grid >= max(w0, ta.min())) & (grid <= min(w1, ta.max()))
        sb = (grid >= tb.min()) & (grid <= tb.max())
    a, b = _zscore(ea, sa), _zscore(eb, sb)
    K = int(round(max_lag_s * hz))
    n = grid.size
    corr = np.full(2 * K + 1, -np.inf)
    for i, k in enumerate(range(-K, K + 1)):
        # b_corr[m] = b[m - k]  (b shifted later by k samples ⇔ offset = k / hz)
        if k >= 0:
            aa, bb = a[k:], b[:n - k]
        else:
            aa, bb = a[:n + k], b[-k:]
        den = math.sqrt(float(aa @ aa) * float(bb @ bb))
        if den > 0:
            corr[i] = float(aa @ bb) / den
    if not np.isfinite(corr).any():
        return OffsetEstimate(0.0, 0.0, 0.0, True)
    j = int(np.argmax(corr))
    delta = 0.0
    if 0 < j < 2 * K and np.isfinite(corr[j - 1]) and np.isfinite(corr[j + 1]):
        y0, y1, y2 = corr[j - 1], corr[j], corr[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    lag = (j - K) + delta
    return OffsetEstimate(offset_s=lag / hz, score=float(corr[j]), lag_samples=float(lag),
                          at_limit=j in (0, 2 * K))


def detect_taps(t, x, *, n: int = 3, hz: float = DEFAULT_HZ, window: tuple[float, float] | None = None,
                min_sep_s: float = 0.25, smooth_s: float = 0.03, thr: float = 5.0, rel: float = 0.2) -> np.ndarray:
    """Times of the (up to) ``n`` strongest activity peaks, sorted — e.g. the three sync taps in the
    pressure stream (QC: were they recorded?).

    Peaks must stand ``thr`` robust deviations (median / MAD of the envelope, i.e. relative to the
    quiet parts) above the median, reach ``rel`` × the strongest peak, and be ``min_sep_s`` apart
    (one peak per tap even though onset and release are separate edges).
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    X = np.asarray(x)
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])
        t, X = t[m], X[m]
    grid, env, sup = change_envelope(t, X, hz=hz, smooth_s=smooth_s)
    v = env[sup]
    if v.size < 3:
        return np.zeros(0)
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    scale = mad if mad > 0 else (float(v.std()) or 1.0)
    z = np.where(sup, (env - med) / scale, -np.inf)
    zmax = float(z.max())
    peaks = np.flatnonzero((z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:]) & (z[1:-1] >= thr)
                           & (z[1:-1] >= rel * zmax)) + 1
    chosen: list[int] = []
    for p in peaks[np.argsort(-z[peaks], kind="stable")]:
        if all(abs(grid[p] - grid[c]) >= min_sep_s for c in chosen):
            chosen.append(int(p))
        if len(chosen) == n:
            break
    return np.sort(grid[chosen]) if chosen else np.zeros(0)


def fit_clock_drift(t_stream, t_ref) -> ClockModel:
    """Least-squares ``t_ref ≈ scale · t_stream + offset`` from matched event times (≥ 2 distinct
    points for drift; one point → pure offset)."""
    ts = np.asarray(t_stream, dtype=np.float64).reshape(-1)
    tr = np.asarray(t_ref, dtype=np.float64).reshape(-1)
    if ts.shape != tr.shape or ts.size == 0:
        raise ValueError("t_stream and t_ref must be equal-length, non-empty")
    if ts.size == 1 or np.ptp(ts) < 1e-9:
        return ClockModel(1.0, float(np.mean(tr - ts)))
    A = np.c_[ts, np.ones_like(ts)]
    (scale, off), *_ = np.linalg.lstsq(A, tr, rcond=None)
    return ClockModel(float(scale), float(off))


# ── session-level ────────────────────────────────────────────────────────────
def sync_windows(events: Sequence[Mapping]) -> list[dict]:
    """Sync phases (``value.kind == "sync"`` or name starting with ``sync``) → ``[{name, t0, t1}]``."""
    out = []
    for p in phases_from_events(events):
        v = p["value"] if isinstance(p["value"], Mapping) else {}
        if v.get("kind") == "sync" or p["name"].startswith("sync"):
            out.append({"name": p["name"], "t0": float(p["t0"]), "t1": float(p["t1"])})
    return out


def stream_kind(manifest: SessionManifest, name: str) -> str:
    """Stream kind from its name / file (``camera_*`` → camera, ``pressure.npz`` → pressure …)."""
    if name.startswith("camera_"):
        return "camera"
    stem = Path(manifest.streams[name].file).stem
    for k in ("pressure", "imu", "joint_state", "hand_pose", "object_pose"):
        if name == k or stem == k:
            return k
    fields = " ".join(manifest.streams[name].fields)
    if "raw[" in fields:
        return "pressure"
    if ".quat" in fields:
        return "imu"
    if "q[" in fields:
        return "joint_state"
    return "other"


def _camera_times_path(d: Path) -> tuple[Path, Path]:
    return d / "timestamps.npy", d / "timestamps_host.npy"


def _gray_blocks(frames: np.ndarray, max_side: int = 16) -> np.ndarray:
    f = np.asarray(frames, dtype=np.float32).mean(-1)                      # [F,H,W]
    F, H, W = f.shape
    bh, bw = max(1, H // max_side), max(1, W // max_side)
    h, w = (H // bh) * bh, (W // bw) * bw
    return f[:, :h, :w].reshape(F, h // bh, bh, w // bw, bw).mean((2, 4)).reshape(F, -1)


def load_sync_signal(session_dir: str | Path, manifest: SessionManifest, name: str, *,
                     t_range: tuple[float, float] | None = None, original: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """``(t, x)`` used for sync: pressure raw, IMU acc, joint q, camera grey-level blocks (frames
    only inside ``t_range``). ``original`` → pre-sync host stamps when present."""
    d = Path(session_dir)
    kind = stream_kind(manifest, name)
    path = manifest.stream_path(d, name)
    if kind == "camera":
        tp, hp = _camera_times_path(path)
        t = np.load(hp if (original and hp.exists()) else tp)
        idx = np.arange(t.size) if t_range is None else np.flatnonzero((t >= t_range[0]) & (t <= t_range[1]))
        if (path / "frames.npy").exists():
            fr = np.load(path / "frames.npy", mmap_mode="r")
            frames = np.asarray(fr[idx]) if idx.size else np.zeros((0, 1, 1, 3), np.uint8)
        else:
            from PIL import Image
            frames = np.stack([np.asarray(Image.open(path / f"{i:06d}.jpg").convert("RGB")) for i in idx]) \
                if idx.size else np.zeros((0, 1, 1, 3), np.uint8)
        return t[idx], _gray_blocks(frames) if idx.size else np.zeros((0, 1))
    key = {"pressure": "raw", "imu": "acc", "joint_state": "q"}.get(kind)
    if key is None:
        raise ValueError(f"stream {name!r} ({kind}) has no sync signal")
    with np.load(path) as z:
        t = z["t_host"] if (original and "t_host" in z.files) else z["t"]
        x = z[key]
    if t_range is not None:
        m = (t >= t_range[0]) & (t <= t_range[1])
        t, x = t[m], x[m]
    return t, x


def _has_host_copy(d: Path, m: SessionManifest, name: str) -> bool:
    path = m.stream_path(d, name)
    if stream_kind(m, name) == "camera":
        return _camera_times_path(path)[1].exists()
    with np.load(path) as z:
        return "t_host" in z.files


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def apply_clock_models(session_dir: str | Path, models: Mapping[str, ClockModel | Mapping], *,
                       manifest: SessionManifest | None = None) -> list[str]:
    """Rewrite stream timestamps as ``model.apply(original host stamps)``; originals are kept
    (``t_host`` in npz, ``timestamps_host.npy`` for cameras) so re-applying never compounds.
    Also usable with models from another session of the same sitting. Returns the streams changed."""
    d = Path(session_dir)
    m = manifest or SessionManifest.load(d)
    done = []
    for name, model in models.items():
        cm = model if isinstance(model, ClockModel) else ClockModel.from_dict(model)
        if name not in m.streams:
            log.warning("clock model for unknown stream %r ignored", name)
            continue
        path = m.stream_path(d, name)
        if stream_kind(m, name) == "camera":
            tp, hp = _camera_times_path(path)
            orig = np.load(hp) if hp.exists() else np.load(tp)
            if not hp.exists():
                np.save(hp, orig)
            np.save(tp, cm.apply(orig))
        else:
            with np.load(path) as z:
                arrs = {k: z[k] for k in z.files}
            orig = arrs.get("t_host", arrs["t"])
            arrs["t_host"] = orig
            arrs["t"] = cm.apply(orig)
            _atomic_savez(path, arrs)
        done.append(name)
    return done


def sync_session(session_dir: str | Path, *, reference: str = "pressure", streams: Iterable[str] | None = None,
                 max_lag_s: float = 0.5, hz: float = DEFAULT_HZ, min_score: float = 0.3,
                 smooth_s: float | None = None, apply: bool = True, save: bool = True) -> dict:
    """3-tap sync of a recorded session (see module docstring).

    Estimates, per stream and sync window, the offset to ``reference``; fits a
    :class:`ClockModel` (drift needs two good windows whose offsets differ by more than the
    per-kind precision :data:`DRIFT_MIN_DELTA_S`, else the mean offset is used; implausible drift
    > 1000 ppm falls back to the mean offset too); with ``apply`` rewrites the timestamps (:func:`apply_clock_models`); records
    everything in ``manifest.calibration["sync"]`` (``save``). Returns that report.
    Default ``streams``: every IMU / camera / extra pressure stream (joint_state only on request).
    ``smooth_s`` None → per-kind :data:`SMOOTH_S_BY_KIND` (max of the pair).
    """
    d = Path(session_dir)
    m = SessionManifest.load(d)
    report = {"reference": reference, "method": "3tap_xcorr", "hz": hz, "max_lag_s": max_lag_s,
              "min_score": min_score, "applied": False, "windows": [], "streams": {}}
    if reference not in m.streams:
        report["status"] = "no_reference"
        return report
    wins = sync_windows(load_events(d))
    report["windows"] = wins
    if not wins:
        report["status"] = "no_sync_phases"
        if save:
            m.calibration[SYNC_KEY] = report
            m.save(d)
        return report
    if streams is None:
        streams = [n for n in m.streams if n != reference and stream_kind(m, n) in ("pressure", "imu", "camera")]
    pad = max_lag_s + 0.1
    models: dict[str, ClockModel] = {}
    k_ref = stream_kind(m, reference)
    for name in streams:
        if name not in m.streams or name == reference:
            continue
        sm = smooth_s if smooth_s is not None else max(SMOOTH_S_BY_KIND.get(k_ref, 0.02),
                                                       SMOOTH_S_BY_KIND.get(stream_kind(m, name), 0.02))
        rows = []
        for w in wins:
            rng = (w["t0"] - pad, w["t1"] + pad)
            try:
                tr, xr = load_sync_signal(d, m, reference, t_range=rng)
                ts, xs = load_sync_signal(d, m, name, t_range=rng)
                est = estimate_offset(tr, xr, ts, xs, max_lag_s, hz, window=(w["t0"], w["t1"]), smooth_s=sm)
                rows.append({"name": w["name"], "t": 0.5 * (w["t0"] + w["t1"]), "offset_s": est.offset_s,
                             "score": est.score, "at_limit": est.at_limit})
            except ValueError as e:
                rows.append({"name": w["name"], "t": 0.5 * (w["t0"] + w["t1"]), "offset_s": None, "score": 0.0,
                             "error": str(e)})
        good = [r for r in rows if r["offset_s"] is not None and r["score"] >= min_score and not r.get("at_limit")]
        entry = {"windows": rows, "smooth_s": sm}
        if not good:
            entry.update(status="low_score", scale=1.0, offset_s=None)
        else:
            ref_t = np.array([r["t"] for r in good])
            off = np.array([r["offset_s"] for r in good])
            min_delta = DRIFT_MIN_DELTA_S.get(stream_kind(m, name), 0.005)
            status, fitted = "ok", False
            model = ClockModel(1.0, float(off.mean()))
            if len(good) >= 2 and float(np.ptp(off)) > min_delta:
                drift = fit_clock_drift(ref_t - off, ref_t)
                if abs(drift.scale - 1.0) > MAX_ABS_DRIFT:
                    status = "drift_rejected"
                else:
                    model, fitted = drift, True
            models[name] = model
            entry.update(status=status, **model.to_dict(), drift_ppm=model.drift_ppm, drift_fitted=fitted,
                         n_windows=len(good))
        report["streams"][name] = entry
    if apply and models:
        todo = dict(models)
        if _has_host_copy(d, m, reference):          # an earlier sync shifted the new reference: restore
            todo[reference] = ClockModel()
        apply_clock_models(d, todo, manifest=m)
        report["applied"] = True
    report["status"] = "ok" if all(e["status"] in ("ok", "drift_rejected") for e in report["streams"].values()) \
        else "partial"
    if save:
        m.calibration[SYNC_KEY] = report
        m.save(d)
    return report
