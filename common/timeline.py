"""Align several timestamped streams onto one master clock (default 200 Hz).

Each :class:`Stream` is ``(t[T], values[T, ...])`` with its own (possibly irregular) rate —
e.g. pressure @ ~200 Hz, IMUs @ 100 Hz, camera @ 30 Hz, joint states @ 500 Hz. They are
resampled onto a shared grid with either linear interpolation (continuous signals) or
zero-order hold (discrete/state signals, "last value wins").

Raw-file parsing is **not** done here: mk555 ``.bin`` parsing stays canonical in
``deformable_sats/sats/preprocessing/bin_merge.py``. Loaders produce ``Stream`` objects and
hand them to :func:`align_streams`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

MASTER_HZ: float = 200.0
Method = Literal["linear", "zoh"]


@dataclass(frozen=True)
class Stream:
    t: np.ndarray        # [T] seconds, strictly increasing after `sanitize()`
    values: np.ndarray   # [T, ...]
    method: Method = "linear"

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        v = np.asarray(self.values)
        if t.ndim != 1 or v.shape[0] != t.shape[0]:
            raise ValueError(f"t {t.shape} / values {v.shape} mismatch")
        if self.method not in ("linear", "zoh"):
            raise ValueError(f"unknown method {self.method!r}")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "values", v)

    def sanitize(self) -> "Stream":
        """Sort by time and drop duplicate timestamps (keep last)."""
        order = np.argsort(self.t, kind="stable")
        t, v = self.t[order], self.values[order]
        keep = np.ones(t.shape[0], dtype=bool)
        keep[:-1] = t[1:] != t[:-1]
        return Stream(t[keep], v[keep], self.method)


def master_clock(t_start: float, t_end: float, hz: float = MASTER_HZ) -> np.ndarray:
    """Uniform grid ``t_start, t_start+1/hz, ... <= t_end``."""
    if hz <= 0 or t_end < t_start:
        raise ValueError("need hz > 0 and t_end >= t_start")
    n = int(np.floor((t_end - t_start) * hz + 1e-9)) + 1
    return t_start + np.arange(n, dtype=np.float64) / hz


def resample(stream: Stream, t_master: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample one stream. Returns ``(values[M, ...], valid[M])``.

    ``valid`` is False outside the stream's own time span (values there are edge-held for
    linear, and the first sample for ZOH before the start) so callers can crop or mask.
    """
    s = stream.sanitize()
    tm = np.asarray(t_master, dtype=np.float64)
    if s.t.shape[0] == 0:
        raise ValueError("empty stream")
    valid = (tm >= s.t[0]) & (tm <= s.t[-1])
    flat = s.values.reshape(s.t.shape[0], -1)
    if s.method == "zoh":
        idx = np.clip(np.searchsorted(s.t, tm, side="right") - 1, 0, s.t.shape[0] - 1)
        out = flat[idx]
    else:
        f = flat.astype(np.float64)
        out = np.stack([np.interp(tm, s.t, f[:, j]) for j in range(f.shape[1])], axis=1)
    return out.reshape((tm.shape[0],) + s.values.shape[1:]), valid


def align_streams(
    streams: Mapping[str, Stream],
    *,
    hz: float = MASTER_HZ,
    span: Literal["intersection", "union"] = "intersection",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align named streams onto a master clock.

    ``span="intersection"`` (default) uses the common overlap so every returned row is valid
    for every stream; ``"union"`` covers everything and reports per-stream ``valid`` masks.
    Returns ``(t_master, values_by_name, valid_by_name)``.
    """
    if not streams:
        raise ValueError("no streams")
    starts = [float(np.min(s.t)) for s in streams.values()]
    ends = [float(np.max(s.t)) for s in streams.values()]
    if span == "intersection":
        t0, t1 = max(starts), min(ends)
        if t1 < t0:
            raise ValueError("streams do not overlap in time")
    elif span == "union":
        t0, t1 = min(starts), max(ends)
    else:
        raise ValueError(f"unknown span {span!r}")
    tm = master_clock(t0, t1, hz)
    vals, valid = {}, {}
    for name, s in streams.items():
        vals[name], valid[name] = resample(s, tm)
    return tm, vals, valid
