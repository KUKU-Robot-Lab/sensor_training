"""Normalisation statistics of processed episodes (fit on the **train split only**).

:func:`compute_stats` returns one :class:`common.signal.NormStats` per key — ``x_norm = (x − offset)
/ scale`` per *feature*, where the trailing dims of an array are flattened (``q[T,D]`` → D features,
``delta_pct[T,N]`` → N taxels, ``hand_finger_pose[T,15,3]`` → 45, :data:`IMU_FEATURES` → F). Use
:func:`apply_stats` / :func:`invert_stats` to (un)normalise arrays of any leading shape.

Masks (``masks="auto"``) keep the statistics honest:
- ΔS-like keys (``delta_pct``, ``pressure_raw``, derived baseline / residual) exclude **saturated**
  samples per taxel (a dropout toward −100 % would dominate the std);
- hand-pose keys, and ``q`` / ``qd`` of glove episodes (they are the vision finger pose), use only
  ``hand_pose_valid`` frames — for ``qd`` additionally eroded by the derivative filter's footprint
  (:func:`qd_valid_mask`), so the velocity spikes where a held label jumps back after a gap never
  inflate the scale.

``method``: ``std`` (mean / std, accumulated per episode with Chan's parallel update, so episodes are
never concatenated), ``robust`` (median / IQR·/1.349 on at most ``max_samples`` frames per episode,
seeded) or ``none``. Scales get ``+ eps`` exactly like ``NormStats.fit``. Stats are saved as one JSON
(:func:`save_stats` / :func:`load_stats`) and shipped with every model that consumes them.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from common.signal import NormStats

from .episode import (
    D_BASELINE_PRED, D_RESIDUAL, D_RESIDUAL_Z, K_DELTA, K_HAND_FINGERS, K_HAND_GLOBAL, K_HAND_VALID, K_HAND_WRIST,
    K_PRESSURE_RAW, K_Q, K_QD, K_SATURATED, Episode,
)

__all__ = [
    "IMU_FEATURES", "SATURATION_MASKED", "HAND_MASKED", "STATS_FORMAT", "as_episodes", "episode_array",
    "q_valid_mask", "qd_valid_mask", "default_mask", "compute_stats", "apply_stats", "invert_stats", "save_stats",
    "load_stats",
]

#: pseudo-key: ``pose.imu_model.imu_features`` of the IMU arrays (see ``motion.episode_imu_features``)
IMU_FEATURES = "imu_features"
SATURATION_MASKED = (K_DELTA, K_PRESSURE_RAW, D_BASELINE_PRED, D_RESIDUAL, D_RESIDUAL_Z)
HAND_MASKED = (K_HAND_GLOBAL, K_HAND_FINGERS, K_HAND_WRIST)
STATS_FORMAT = 1
_METHODS = ("std", "robust", "none")


def as_episodes(episodes: Iterable[Episode | str | Path]) -> list[Episode]:
    """Episodes or episode directories → loaded (mmap) :class:`Episode` objects."""
    return [e if isinstance(e, Episode) else Episode.load(e) for e in episodes]


def episode_array(ep: Episode, key: str, *, imu_kw: Mapping[str, Any] | None = None) -> np.ndarray:
    """``[T, ...]`` values of ``key``: an array, a derived stage output, or :data:`IMU_FEATURES`."""
    if key == IMU_FEATURES:
        from .motion import episode_imu_features  # lazy: motion imports torch

        return episode_imu_features(ep, **dict(imu_kw or {}))
    if ep.has(key):
        return ep[key]
    if ep.has_derived(key):
        return ep.derived(key)
    raise KeyError(f"episode {ep.meta.episode_id!r} has no array {key!r}")


def _q_from_hand(ep: Episode) -> bool:
    src = (ep.meta.preprocessing or {}).get("q_source")
    return src == "hand_pose" or (src is None and ep.meta.kind == "glove")


def q_valid_mask(ep: Episode) -> np.ndarray:
    """``[T]`` frames whose ``q`` is a measurement: ``hand_pose_valid`` for glove episodes (q = vision
    finger pose; invalid frames hold the last label), all frames for joint-state q."""
    if ep.has(K_HAND_VALID) and _q_from_hand(ep):
        return np.asarray(ep[K_HAND_VALID], dtype=bool)
    return np.ones(ep.T, dtype=bool)


def qd_valid_mask(ep: Episode) -> np.ndarray:
    """``[T]`` frames whose ``qd`` is a measurement: :func:`q_valid_mask` eroded by the footprint of
    the preprocessing derivative filter (``datasets.build.qd_support`` with the episode's
    ``meta.preprocessing.config.qd``). Around a hand-label gap the held ``q`` jumps back to the true
    pose, which the Savitzky–Golay derivative turns into a velocity spike on frames whose own ``q``
    is valid; those frames are excluded here. Joint-state ``q`` → all frames."""
    qv = q_valid_mask(ep)
    if qv.all():
        return qv
    from .build import DEFAULTS, qd_support  # lazy: build pulls in the acquisition modules

    pre = ep.meta.preprocessing or {}
    qc = {**DEFAULTS["qd"], **((pre.get("config") or {}).get("qd") or {})}
    if pre.get("qd_source") == "file":
        return qv
    past, future = qd_support(float(ep.meta.hz), method=qc["method"], window_s=qc["window_s"],
                              polyorder=qc["polyorder"])
    bad = (~qv).astype(np.int64)
    pad = np.concatenate([np.repeat(bad[:1], past), bad, np.repeat(bad[-1:], future)])   # edge = filter padding
    c = np.concatenate([[0], np.cumsum(pad)])
    w = past + future + 1
    return (c[w:] - c[:-w]) == 0


def default_mask(ep: Episode, key: str) -> np.ndarray | None:
    """The ``masks="auto"`` rule: ``~saturated`` ``[T,N]`` for ΔS-like keys, ``hand_pose_valid``
    ``[T]`` for hand keys and glove ``q`` (:func:`q_valid_mask`), :func:`qd_valid_mask` for glove
    ``qd``, else None (all frames)."""
    if key in SATURATION_MASKED and ep.has(K_SATURATED):
        return ~np.asarray(ep[K_SATURATED], dtype=bool)
    if ep.has(K_HAND_VALID) and key in HAND_MASKED:
        return np.asarray(ep[K_HAND_VALID], dtype=bool)
    if ep.has(K_HAND_VALID) and _q_from_hand(ep):
        if key == K_Q:
            return q_valid_mask(ep)
        if key == K_QD:
            return qd_valid_mask(ep)
    return None


def _flat_mask(mask: np.ndarray | None, T: int, F: int) -> np.ndarray:
    if mask is None:
        return np.ones((T, F), dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != T:
        raise ValueError(f"mask has {m.shape[0]} rows, expected {T}")
    m = m.reshape(T, -1)
    if m.shape[1] == 1:
        return np.broadcast_to(m, (T, F))
    if m.shape[1] != F:
        raise ValueError(f"mask with {m.shape[1]} columns does not match {F} features")
    return m


def compute_stats(episodes: Iterable[Episode | str | Path], keys: Sequence[str] = (K_Q, K_QD), method: str = "std",
                  *, masks: str | Mapping[str, Any] | None = "auto", eps: float = 1e-6,
                  max_samples: int | None = 20000, seed: int = 0,
                  imu_kw: Mapping[str, Any] | None = None) -> dict[str, NormStats]:
    """Per-feature :class:`NormStats` for each key over ``episodes`` (use the train split).

    ``masks``: ``"auto"`` (:func:`default_mask`), None (all samples), or ``{key: "auto" | None |
    callable(ep) -> bool[T] | bool[T,F]}``. ``max_samples`` caps frames per episode for ``robust``.
    ``imu_kw`` go to ``motion.episode_imu_features`` for :data:`IMU_FEATURES`. Features without any
    valid sample get offset 0 / scale 1 (with a warning)."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    eps_ = float(eps)
    eps_list = as_episodes(episodes)
    if not eps_list:
        raise ValueError("no episodes")
    rng = np.random.default_rng(seed)
    out: dict[str, NormStats] = {}
    for key in keys:
        n = mean = m2 = None
        samples: list[list[np.ndarray]] | None = None
        shape = None
        for ep in eps_list:
            x = np.asarray(episode_array(ep, key, imu_kw=imu_kw), dtype=np.float64)
            T = x.shape[0]
            x = x.reshape(T, -1)
            if shape is None:
                shape = x.shape[1]
            elif x.shape[1] != shape:
                raise ValueError(f"{key!r}: episode {ep.meta.episode_id!r} has {x.shape[1]} features, others {shape}")
            rule = masks.get(key, "auto") if isinstance(masks, Mapping) else masks
            mk = default_mask(ep, key) if rule == "auto" else (rule(ep) if callable(rule) else None)
            m = _flat_mask(mk, T, shape) & np.isfinite(x)
            if method == "std":
                nb = m.sum(0).astype(np.float64)
                xs = np.where(m, x, 0.0)
                mb = xs.sum(0) / np.maximum(nb, 1.0)
                m2b = (np.where(m, x - mb, 0.0) ** 2).sum(0)
                if n is None:
                    n, mean, m2 = nb, mb, m2b
                else:
                    tot = n + nb
                    d = mb - mean
                    w = np.divide(nb, tot, out=np.zeros_like(tot), where=tot > 0)
                    mean = mean + d * w
                    m2 = m2 + m2b + d * d * np.divide(n * nb, tot, out=np.zeros_like(tot), where=tot > 0)
                    n = tot
            elif method == "robust":
                rows = np.arange(T)
                if max_samples is not None and T > max_samples:
                    rows = np.sort(rng.choice(T, int(max_samples), replace=False))
                if samples is None:
                    samples = [[] for _ in range(shape)]
                xr, mr = x[rows], m[rows]
                for j in range(shape):
                    samples[j].append(xr[mr[:, j], j])
        F = int(shape)
        if method == "none":
            off, sc = np.zeros(F), np.ones(F)
            empty = np.zeros(F, dtype=bool)
        elif method == "std":
            empty = n == 0
            off = np.where(empty, 0.0, mean)
            sc = np.where(empty, 1.0, np.sqrt(np.maximum(m2, 0.0) / np.maximum(n, 1.0)))
        else:
            off, sc, empty = np.zeros(F), np.ones(F), np.zeros(F, dtype=bool)
            for j in range(F):
                v = np.concatenate(samples[j]) if samples[j] else np.zeros(0)
                if v.size == 0:
                    empty[j] = True
                    continue
                off[j] = np.median(v)
                q75, q25 = np.percentile(v, [75, 25])
                sc[j] = (q75 - q25) / 1.349
        if empty.any():
            warnings.warn(f"compute_stats: {int(empty.sum())} feature(s) of {key!r} have no valid samples "
                          "(offset 0, scale 1)", stacklevel=2)
        out[key] = NormStats(offset=off.astype(np.float32), scale=(np.maximum(sc, 0.0) + eps_).astype(np.float32))
    return out


def apply_stats(stats: NormStats, x: np.ndarray) -> np.ndarray:
    """Normalise ``x[..., *feat_shape]`` with per-flattened-feature stats → float32, same shape.
    The trailing dims whose product equals the number of features are flattened."""
    x = np.asarray(x, dtype=np.float32)
    F = int(stats.offset.shape[0])
    lead = _lead_dims(x.shape, F)
    return stats.apply(x.reshape(lead + (F,))).reshape(x.shape)


def invert_stats(stats: NormStats, x_norm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`apply_stats`."""
    x = np.asarray(x_norm, dtype=np.float32)
    F = int(stats.offset.shape[0])
    return stats.invert(x.reshape(_lead_dims(x.shape, F) + (F,))).reshape(x.shape)


def _lead_dims(shape: tuple, F: int) -> tuple:
    prod = 1
    for k in range(len(shape) - 1, -1, -1):
        prod *= shape[k]
        if prod == F:
            return tuple(shape[:k])
        if prod > F:
            break
    if F == 1:
        return tuple(shape)
    raise ValueError(f"array of shape {shape} does not end in {F} features")


def save_stats(stats: Mapping[str, NormStats], path: str | Path, *, meta: Mapping[str, Any] | None = None) -> Path:
    """Write ``{"format", "stats": {key: {offset, scale}}, "meta"}`` JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"format": STATS_FORMAT, "stats": {k: v.to_dict() for k, v in stats.items()}, "meta": dict(meta or {})}
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return p


def load_stats(path: str | Path, *, return_meta: bool = False):
    """Read :func:`save_stats` output → ``{key: NormStats}`` (and ``meta`` with ``return_meta``)."""
    d = json.loads(Path(path).read_text())
    if int(d.get("format", STATS_FORMAT)) > STATS_FORMAT:
        raise ValueError(f"stats format {d['format']} is newer than supported {STATS_FORMAT}")
    stats = {k: NormStats.from_dict(v) for k, v in d.get("stats", {}).items()}
    return (stats, d.get("meta", {})) if return_meta else stats
