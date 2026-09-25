"""Raw session → processed :class:`~robot_skin.datasets.episode.Episode` (preprocessing).

:func:`preprocess_session` turns one raw session directory (``acquisition.manifest`` formats,
``docs/DATA_FORMAT.md``) into one Episode under ``<out_root>/<dataset>/<episode_id>/``. Every
training stage reads only Episodes, so all alignment / calibration / labelling decisions live here::

    pressure.npz ─ loader (injectable: mk555 .bin) ─ layout.by_channel ─ [low-pass] ─┐
    imu.npz ─ manifest.calibration offsets (G⁻¹ ⊗ q ⊗ q_off, R_offᵀ v) ─ continuity ─┤
    hand_pose.npz* ─ smooth_hand_labels (confidence gate, gap SLERP, low-pass) ─────┤ master clock
    joint_state.npz ─ URDFModel.reorder_q (URDF actuated order) ────────────────────┤ (200 Hz)
    object_pose.npz*, camera_<name>/timestamps.npy (zoh → cam_<name>_idx) ──────────┘
      (* offline products: read from their canonical file even when session.json does not list them)
      → segments (events.jsonl → ``acquisition.recorder.session_segments``)
      → baseline_raw (median over the first ``no_contact`` segment) → ΔS% (SATS sign) → saturated
      → q / qd (Savitzky–Golay) → taxel poses (MANO skeleton | URDF FK) → self_touch (capsules)
      → phase_id / meta.phases (events.jsonl) → contact_label (segments + self-touch)

Conventions (the contract of ``episode.py`` / ``docs/DATA_FORMAT.md``)
- Master clock: uniform ``1/hz`` grid on the **session clock** (``t`` values are multiples of
  ``1/hz``), spanning the overlap of the ``clock.reference`` streams (pressure + IMU / joint state).
  Streams that do not cover a frame are edge-held there, and a gap between two native samples wider
  than ``max(clock.max_gap_s, 4 × median spacing)`` is bridged; neither is a measurement — validity
  is carried by ``imu_valid`` / ``joint_state_valid`` (span and gaps), ``hand_pose_valid``,
  ``cam_<name>_idx = -1``, and pressure frames inside a gap are flagged ``saturated``.
- Continuous streams are linearly interpolated; quaternions are interpolated component-wise on
  continuity-fixed (same-hemisphere) sequences and renormalised (≈ SLERP at 200 Hz over ≤ 100 Hz
  samples); axis-angle labels go through quaternions for the same reason.
- ΔS [%] = (raw − baseline) / baseline · 100 (``common.signal.relative_change``; SATS sign: a press
  lowers raw → **negative** ΔS). ``baseline_raw`` is the per-taxel median of the first
  ``baseline.duration_s`` of the first ``no_contact`` segment (``common.signal.estimate_baseline``),
  over frames without a saturated taxel; fallbacks: ``manifest.baseline`` then the recording start.
- ``saturated`` = raw on/near an ADC rail (or non-finite: missing samples of an injected loader are
  bridged by interpolation so ΔS stays finite) at *either* bracketing raw sample (so a dropout
  never leaks into a half-interpolated "valid" value), inside a pressure sample gap, or
  ``|ΔS| ≥ max_abs_pct``.
- ``taxel_pos`` / ``taxel_nrm`` are in the **hand / robot base frame** (``episode.py`` contract):
  glove = MANO wrist-joint frame (``global_orient`` and ``wrist_pos`` removed, i.e. only the finger
  pose moves the taxels — the frame an online IMU→pose model reproduces and the one the robot's
  URDF root frame corresponds to); robot = URDF root link; otherwise the layout's own frame. World
  poses of glove taxels: ``R(hand_global_orient) · taxel_pos + hand_wrist_pos``.
- Glove ``q`` = smoothed MANO ``finger_pose`` flattened to 45 (joint order ``MANO_JOINTS[1:]``, xyz);
  robot ``q`` = ``joint_state`` in ``URDFModel.joint_names`` order. ``qd`` = Savitzky–Golay
  derivative (:func:`joint_velocity`), by default the **causal** variant (polynomial fit over the
  last 50 ms evaluated at its newest sample): an online controller recomputes exactly the same
  ``qd`` from its q ring buffer, so offline-trained stages match deployment (``qd.method: savgol``
  gives the smoother centred estimate for offline-only use).
- Segments: ``events.jsonl`` is the source of truth for the phase-derived ones (recorder sessions: an
  operator's boundary fix in the events reaches the labels without editing ``session.json``);
  synthetic / hand-written manifest segments are used as written.
- ``contact_label``: 0 inside ``no_contact`` segments, 1 where the geometric self-touch label is set,
  −1 otherwise (D2 object contact stays −1 for later pseudo-labelling); a geometric self-touch inside
  a ``no_contact`` segment is a conflict → −1 (``labels.conflict``).

The IMU calibration model and frames follow ``pose.imu_model`` (gyro / acc in each sensor frame,
acc = specific force incl. gravity); hand labels follow ``pose.vision_hand`` (MANO: Romero et al.,
SIGGRAPH Asia 2017; labels from HaMeR, Pavlakos et al., CVPR 2024). Raw mk555 ``.bin`` parsing stays
canonical in ``deformable_sats/sats/preprocessing/bin_merge.py`` — inject it as ``pressure.loader``.

Synthetic sessions (``datasets.synthetic``) carry ``gt_synthetic.npz``; it is not a stream, but when
present its tactile ground truth is resampled onto the master clock (optional ``gt_*`` arrays) and
compared with the processed ΔS (``meta.preprocessing["synthetic_gt"]``).

CLI (idempotent: existing episodes are skipped unless ``--force``)::

    python -m robot_skin.datasets.build --raw robot_skin/data/raw --out robot_skin/data/processed \\
        [--config robot_skin/configs/stages/preprocess.yaml] [--set baseline.duration_s=2] [--force]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from common.layouts import Layout, load_layout
from common.signal import estimate_baseline, relative_change, saturation_mask
from common.timeline import MASTER_HZ, Stream, master_clock, resample
from robot_skin.acquisition.manifest import MANIFEST_NAME, SessionManifest
from robot_skin.acquisition.recorder import load_events, phases_from_events, session_segments
from robot_skin.config import deep_merge

from .episode import (
    EPISODE_JSON, K_CONTACT_LABEL, K_DELTA, K_HAND_FINGERS, K_HAND_GLOBAL, K_HAND_VALID, K_HAND_WRIST,
    K_IMU_ACC, K_IMU_GYRO, K_IMU_QUAT, K_IMU_VALID, K_JOINT_VALID, K_OBJECT_POS, K_OBJECT_QUAT, K_PHASE,
    K_PRESSURE_RAW, K_Q, K_QD, K_SATURATED, K_SELF_TOUCH, K_T, K_TAXEL_NRM, K_TAXEL_POS, S_BASELINE_RAW,
    S_CHANNELS, Episode, EpisodeMeta, cam_idx_key,
)

__all__ = [
    "PREPROCESS_VERSION", "DEFAULTS", "CONFIG_PATH", "LAYOUT_FILE", "GT_FILE", "GT_KEYS", "HAND_Q_NAMES",
    "OFFLINE_STREAMS", "load_preprocess_config", "resolve_layout", "load_episode_layout", "load_pressure_npz",
    "joint_velocity", "qd_support", "camera_frame_index", "phase_ids", "contact_labels", "episode_dir",
    "preprocess_session", "find_sessions", "synthetic_origin", "build_all", "main",
]

log = logging.getLogger(__name__)

#: bump when the processed output changes for the same raw session + config
#: 2: glove taxel poses in the hand frame; 3: segments from events.jsonl, stream gaps / coverage
#: (imu_valid, joint_state_valid, pressure gaps flagged saturated), offline hand/object pose files
PREPROCESS_VERSION = "robot_skin.datasets.build/3"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "preprocess.yaml"
LAYOUT_FILE = "layout.yaml"            # copy of the session layout inside every episode
GT_FILE = "gt_synthetic.npz"           # = datasets.synthetic.GT_FILE (not imported: keeps torch-free paths light)
#: optional synthetic ground truth resampled onto the master clock (``synthetic_gt.store``)
GT_KEYS = {"artefact_pct": "gt_artefact_pct", "press_pct": "gt_press_pct", "contact": "gt_contact",
           "self_touch": "gt_self_touch", "object_contact": "gt_object_contact"}
_CAMERA_MODES = ("copy", "symlink", "none")
_QD_METHODS = ("savgol", "savgol_causal", "gradient")
_CONFLICT = ("unknown", "segment", "geometry")
#: offline products added to a session directory after recording (vision / mocap passes). Read from
#: this canonical file even when ``session.json`` does not list the stream (a real recording never
#: does: labels are made later, docs/DATA_ACQUISITION.md §9); ``save_hand_labels`` registers it.
OFFLINE_STREAMS = {"hand_pose": "hand_pose.npz", "object_pose": "object_pose.npz"}

#: Defaults of every preprocessing knob (``configs/stages/preprocess.yaml`` mirrors this dict).
DEFAULTS: dict[str, Any] = {
    "stage": "preprocess",
    "raw_root": "robot_skin/data/raw",
    "out_root": "robot_skin/data/processed",
    "master_hz": None,                 # None → manifest.master_hz (200)
    "layout": None,                    # override manifest.layout (built-in name or YAML path)
    "clock": {"reference": ["pressure", "imu", "joint_state"], "span": "intersection", "max_gap_s": 0.1},
    "pressure": {"loader": None, "lowpass_hz": None, "lowpass_order": 2, "adc_min": 0.0,
                 "adc_max": float(2 ** 24 - 1), "rail_margin": 0.0, "max_abs_pct": 90.0},
    "baseline": {"phase": None, "duration_s": 1.0, "trim_s": 0.0, "fallback_s": 1.0, "use_manifest": True},
    "imu": {"apply_calibration": True, "calibrate_if_missing": False, "calibration_phase": "imu_calibration",
            "vec_frame": "sensor"},
    "hand_pose": {"min_conf": 0.5, "max_gap_s": 0.25, "cutoff_hz": 6.0, "order": 2, "mano_model": None},
    "qd": {"method": "savgol_causal", "window_s": 0.05, "polyorder": 2, "source": "derivative"},
    "robot": {"urdf": None},
    "self_touch": {"enabled": True, "margin_m": 0.004},
    "labels": {"conflict": "unknown"},
    "cameras": {"copy_frames": "symlink", "max_age_s": None},
    "synthetic_gt": {"check": True, "store": True},
    "qc": {"skip_failed": True},
}

#: glove ``q`` column names: MANO finger joints (``pose.mano.MANO_JOINTS[1:]``) × axis-angle xyz
_MANO_FINGER_JOINTS = ("index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2",
                       "pinky3", "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3")
HAND_Q_NAMES: tuple[str, ...] = tuple(f"{j}_{a}" for j in _MANO_FINGER_JOINTS for a in "xyz")


# ── config ─────────────────────────────────────────────────────────────────
def load_preprocess_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ← YAML ``path`` ← ``overrides`` (nested dicts, deep-merged). Unknown
    top-level or section keys raise ``ValueError`` (typos would otherwise be silently ignored)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    _check_cfg(cfg)
    return cfg


def _check_cfg(cfg: Mapping[str, Any]) -> None:
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown preprocess config keys {unknown}; valid: {sorted(DEFAULTS)}")
    for sec, dv in DEFAULTS.items():
        if isinstance(dv, dict):
            if not isinstance(cfg.get(sec), Mapping):
                raise ValueError(f"preprocess config {sec!r} must be a mapping")
            bad = sorted(set(cfg[sec]) - set(dv))
            if bad:
                raise ValueError(f"unknown keys {bad} in preprocess config section {sec!r}; valid: {sorted(dv)}")
    if cfg["cameras"]["copy_frames"] not in _CAMERA_MODES and cfg["cameras"]["copy_frames"] not in (True, False):
        raise ValueError(f"cameras.copy_frames must be one of {_CAMERA_MODES}")
    if cfg["qd"]["method"] not in _QD_METHODS:
        raise ValueError(f"qd.method must be one of {_QD_METHODS}")
    if cfg["qd"]["source"] not in ("derivative", "file"):
        raise ValueError("qd.source must be 'derivative' or 'file'")
    if cfg["labels"]["conflict"] not in _CONFLICT:
        raise ValueError(f"labels.conflict must be one of {_CONFLICT}")
    if cfg["clock"]["span"] not in ("intersection", "pressure"):
        raise ValueError("clock.span must be 'intersection' or 'pressure'")
    mg = cfg["clock"]["max_gap_s"]
    if mg is not None and not (isinstance(mg, (int, float)) and not isinstance(mg, bool) and mg > 0):
        raise ValueError(f"clock.max_gap_s must be a positive number of seconds or null, got {mg!r}")


def _merge_cfg(cfg: Mapping[str, Any] | None) -> dict:
    if cfg is None:
        return load_preprocess_config()
    return load_preprocess_config(overrides=cfg)


def _cfg_hash(cfg: Mapping[str, Any]) -> str:
    keep = {k: v for k, v in cfg.items() if k not in ("raw_root", "out_root")}
    return hashlib.sha1(json.dumps(_jsonable(keep), sort_keys=True).encode()).hexdigest()[:12]


def _source_fingerprint(sdir: Path, man: SessionManifest) -> str:
    """Hash of what preprocessing reads from the raw session: the manifest and ``events.jsonl``
    (content), every stream file and camera ``timestamps.npy`` (size + a content hash of the time
    vector: npz ``t`` / the ``.npy``) and the unregistered offline products (:data:`OFFLINE_STREAMS`,
    also while absent). Changes when e.g. ``hand_pose.npz`` is added later, events are edited or a
    sync / ``apply_clock_models`` rewrites timestamps in place (same file sizes), but not when a
    dataset is merely copied (no mtimes)."""
    files = [MANIFEST_NAME, "events.jsonl"]
    for name, info in sorted(man.streams.items()):
        files.append(f"{info.file}/timestamps.npy" if name.startswith("camera_") else info.file)
    files += [f for name, f in OFFLINE_STREAMS.items() if name not in man.streams and f not in files]
    items = []
    for f in files:
        p = sdir / f
        items.append((f, p.stat().st_size, _content_hash(p)) if p.is_file() else (f, -1))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:12]


def _content_hash(p: Path) -> str | None:
    """sha1 of a small text file, of an npz's ``t`` vector or of an ``.npy`` (camera timestamps);
    None for other (large / foreign) files, which are fingerprinted by size only."""
    try:
        if p.suffix in (".json", ".jsonl"):
            return hashlib.sha1(p.read_bytes()).hexdigest()[:12]
        if p.suffix == ".npz":
            with np.load(p) as z:
                if "t" not in z.files:
                    return None
                t = np.ascontiguousarray(z["t"], dtype=np.float64)
            return hashlib.sha1(t.tobytes()).hexdigest()[:12]
        if p.suffix == ".npy":
            return hashlib.sha1(np.ascontiguousarray(np.load(p)).tobytes()).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a corrupt file must not break the skip check (the build reports it)
        return "unreadable"
    return None


def _stream_file(sdir: Path, man: SessionManifest, name: str, notes: list[str]) -> Path | None:
    """File of an optional stream: the manifest entry when the file exists, else — for the offline
    products (:data:`OFFLINE_STREAMS`) that are not registered — their canonical file in the session
    dir (noted)."""
    if name in man.streams:
        p = man.stream_path(sdir, name)
        if p.is_file():
            return p
        notes.append(f"stream {name!r}: {man.streams[name].file} not found (ignored)")
        return None
    f = OFFLINE_STREAMS.get(name)
    if f is None or not (sdir / f).is_file():
        return None
    started = _recording_start_epoch(man)
    if started is not None and (sdir / f).stat().st_mtime < started - 60.0:
        # offline labels are made from this take's recording; an older file is a leftover of an earlier
        # take (Recorder(overwrite=True) keeps files it does not write)
        notes.append(f"{f} is not registered in session.json and predates the recording (earlier take?): ignored")
        return None
    notes.append(f"{f} is not registered in session.json: read from the session directory")
    return sdir / f


def _recording_start_epoch(man: SessionManifest) -> float | None:
    rec = man.meta.get("recorder") if isinstance(man.meta, Mapping) else None
    ts = rec.get("started_utc") if isinstance(rec, Mapping) else None
    if not ts:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(ts)).timestamp()
    except ValueError:
        return None


def _gap_frames(t_src: np.ndarray, tm: np.ndarray, max_gap_s: float | None) -> np.ndarray:
    """``[T]`` master frames strictly inside a gap of a native stream: consecutive samples farther
    apart than ``max(max_gap_s, 4 × median spacing)`` (dropped samples / a stalled device — the
    linear interpolation across them is not a measurement). ``max_gap_s=None`` → none."""
    t = np.sort(np.asarray(t_src, dtype=np.float64).reshape(-1))
    tm = np.asarray(tm, dtype=np.float64)
    if max_gap_s is None or t.shape[0] < 2:
        return np.zeros(tm.shape[0], dtype=bool)
    dt = np.diff(t)
    thr = max(float(max_gap_s), 4.0 * float(np.median(dt)))
    k = np.searchsorted(t, tm, side="right") - 1                    # t[k] ≤ tm < t[k+1]
    inside = (k >= 0) & (k < t.shape[0] - 1)
    kk = np.clip(k, 0, t.shape[0] - 2)
    return inside & (dt[kk] > thr) & (tm > t[kk])


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Layout):                             # cfg layout override given as an object
        return _jsonable(x.to_dict(units="mm"))
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if callable(x):                                       # e.g. an injected loader in cfg
        return f"{getattr(x, '__module__', '?')}:{getattr(x, '__qualname__', repr(x))}"
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    return str(x)                                         # never let meta / config hashing crash


def _import_callable(spec: str | Callable | None) -> Callable | None:
    if spec is None or callable(spec):
        return spec
    mod, _, name = str(spec).partition(":")
    if not name:
        raise ValueError(f"loader must be 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(mod), name)


# ── layout ─────────────────────────────────────────────────────────────────
def resolve_layout(session_dir: str | Path, manifest: SessionManifest,
                   override: str | Path | Layout | None = None) -> tuple[Layout, str]:
    """Layout of a raw session and a reference string for the episode meta.

    Order: ``override`` → a relative YAML path in ``manifest.layout`` resolved against the session
    dir → the session-local copy (``meta.synthetic.layout_file`` or the YAML's file name inside the
    session dir, so moved datasets keep working) → an absolute YAML path → a built-in layout name.
    The reference is the built-in name, or ``""`` for a custom YAML (the episode then carries its own
    ``layout.yaml``)."""
    sdir = Path(session_dir)
    ref = override if override is not None else manifest.layout
    if isinstance(ref, Layout):
        return ref, ""
    ref = str(ref)
    p = Path(ref)
    if p.suffix not in (".yaml", ".yml"):
        return load_layout(ref), ref
    cands = []
    if not p.is_absolute():
        cands.append(sdir / p)
    local = (manifest.meta.get("synthetic") or {}).get("layout_file")
    if local:
        cands.append(sdir / str(local))
    cands += [sdir / p.name, p]
    for c in cands:
        if c.is_file():
            return load_layout(c), ""
    raise FileNotFoundError(f"layout {ref!r} of session {sdir} not found (tried {[str(c) for c in cands]})")


def load_episode_layout(episode: Episode | str | Path) -> Layout:
    """Layout of a processed episode: its own ``layout.yaml`` copy (always written by
    :func:`preprocess_session`), else ``meta.layout`` (built-in name or path)."""
    if isinstance(episode, Episode):
        root, name = episode.root, episode.meta.layout
    else:
        root = Path(episode)
        name = json.loads((root / EPISODE_JSON).read_text())["layout"]
    if root is not None and (Path(root) / LAYOUT_FILE).is_file():
        return load_layout(Path(root) / LAYOUT_FILE)
    return load_layout(name)


# ── stream loading ─────────────────────────────────────────────────────────
def load_pressure_npz(session_dir: str | Path, manifest: SessionManifest) -> tuple[np.ndarray, np.ndarray]:
    """Default pressure loader: ``pressure.npz`` ``t[T]`` (s) and ``raw[T,C]`` (channel order).
    For other formats inject ``pressure.loader`` (``(session_dir, manifest) -> (t, raw)``)."""
    if "pressure" not in manifest.streams:
        raise ValueError(f"session {session_dir} has no 'pressure' stream")
    p = manifest.stream_path(session_dir, "pressure")
    if p.suffix != ".npz":
        raise ValueError(f"pressure file {p.name!r} is not an npz; pass a pressure loader (cfg pressure.loader "
                         "= 'module:function', e.g. a wrapper around deformable_sats/sats/preprocessing/bin_merge.py)")
    with np.load(p) as z:
        return np.asarray(z["t"], dtype=np.float64), np.asarray(z["raw"], dtype=np.float64)


def _sorted_stream(t, *arrays):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    order = np.argsort(t, kind="stable")
    keep = np.ones(t.shape[0], dtype=bool)
    ts = t[order]
    keep[:-1] = ts[1:] != ts[:-1]                                     # duplicate stamps: keep last
    idx = order[keep]
    return (t[idx],) + tuple(None if a is None else np.asarray(a)[idx] for a in arrays)


def _resample(t_src, values, tm, method="linear"):
    return resample(Stream(np.asarray(t_src, dtype=np.float64), np.asarray(values), method), tm)


def _resample_quat(t_src, q, tm) -> tuple[np.ndarray, np.ndarray]:
    """Quaternions ``[Ts,...,4]`` → master clock: continuity fix, component-wise linear
    interpolation, renormalisation (≈ SLERP for small steps)."""
    from robot_skin.geometry.rotations import quat_fix_continuity

    t_src, q = _sorted_stream(t_src, q)
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = quat_fix_continuity(q, axis=0)
    v, valid = _resample(t_src, q, tm)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return quat_fix_continuity(v, axis=0), valid


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase low-pass along axis 0 (scipy Butterworth ``filtfilt``; centred moving-average
    fallback with ≈ the same −3 dB point). Non-causal: document it when enabled."""
    if cutoff_hz <= 0:
        raise ValueError("lowpass cutoff must be > 0")
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:  # pragma: no cover - scipy is normally present
        butter = None
    if butter is not None:
        b, a = butter(int(order), min(0.99, cutoff_hz / (0.5 * fs)))
        if x.shape[0] > 3 * max(len(a), len(b)):
            return filtfilt(b, a, x, axis=0)
    w = int(round(0.443 * fs / cutoff_hz))
    w = min(w + (w + 1) % 2, x.shape[0] - (x.shape[0] + 1) % 2)
    if w < 3:
        return x
    k = np.ones(w) / w
    pad = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w // 2, 0)], 0)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def joint_velocity(q: np.ndarray, hz: float, *, method: str = "savgol_causal", window_s: float = 0.05,
                   polyorder: int = 2) -> np.ndarray:
    """Smoothed time derivative of ``q[T,D]`` sampled at ``hz`` → ``[T,D]`` float32 (units/s).

    ``savgol``: centred Savitzky–Golay derivative (scipy ``savgol_filter``, ``mode="interp"``);
    ``savgol_causal``: the same polynomial fit evaluated at the *last* sample of each window
    (history only, start edge-padded) — what an online controller can reproduce exactly;
    ``gradient``: central differences + centred moving average. Without scipy the Savitzky–Golay
    modes fall back to differences + moving average (causal for ``savgol_causal``)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"q must be [T, D], got {q.shape}")
    if method not in _QD_METHODS:
        raise ValueError(f"method must be one of {_QD_METHODS}")
    T = q.shape[0]
    if T < 2:
        return np.zeros(q.shape, dtype=np.float32)
    dt = 1.0 / float(hz)
    w = _qd_window(hz, window_s, polyorder)
    try:
        from scipy.signal import savgol_coeffs, savgol_filter
    except ImportError:  # pragma: no cover - scipy is normally present
        savgol_filter = None
    if method == "savgol" and savgol_filter is not None and T >= w:
        return savgol_filter(q, w, int(polyorder), deriv=1, delta=dt, axis=0, mode="interp").astype(np.float32)
    if method == "savgol_causal" and savgol_filter is not None:
        c = savgol_coeffs(w, int(polyorder), deriv=1, delta=dt, pos=w - 1, use="dot")   # c · q[t-w+1..t]
        pad = np.concatenate([np.repeat(q[:1], w - 1, axis=0), q], axis=0)
        win = np.lib.stride_tricks.sliding_window_view(pad, w, axis=0)                  # [T,D,w]
        return (win @ c).astype(np.float32)
    k = max(1, min(w, T))
    if method == "savgol_causal":
        d = np.diff(q, axis=0, prepend=q[:1]) / dt
        pad = np.concatenate([np.repeat(d[:1], k - 1, axis=0), d], axis=0)
        return np.lib.stride_tricks.sliding_window_view(pad, k, axis=0).mean(-1).astype(np.float32)
    d = np.gradient(q, dt, axis=0)
    if k < 3:
        return d.astype(np.float32)
    k -= (k + 1) % 2
    pad = np.concatenate([np.repeat(d[:1], k // 2, 0), d, np.repeat(d[-1:], k // 2, 0)], 0)
    return np.lib.stride_tricks.sliding_window_view(pad, k, axis=0).mean(-1).astype(np.float32)


def _qd_window(hz: float, window_s: float, polyorder: int) -> int:
    """Odd Savitzky–Golay window length (frames) used by :func:`joint_velocity`."""
    w = max(int(polyorder) + 2, int(round(float(window_s) * float(hz))))
    return w + (w + 1) % 2


def qd_support(hz: float, *, method: str = "savgol_causal", window_s: float = 0.05,
               polyorder: int = 2) -> tuple[int, int]:
    """``(past, future)`` frames of ``q`` that :func:`joint_velocity` reads to form ``qd[t]``
    (frames ``t−past … t+future``). A ``qd`` sample is only a measurement when all of them are:
    a held / gap-filled ``q`` segment next to a jump back to the true pose makes a spurious
    velocity spike over exactly this footprint (``motion`` / ``stats`` erode their masks by it)."""
    if method not in _QD_METHODS:
        raise ValueError(f"method must be one of {_QD_METHODS}")
    w = _qd_window(hz, window_s, polyorder)
    if method == "savgol_causal":
        return w, 0                                      # w − 1 (+1 covers the no-scipy difference fallback)
    if method == "savgol":
        return w // 2, w // 2
    k = w - (w + 1) % 2                                  # gradient: centred difference + moving average
    return k // 2 + 1, k // 2 + 1


def camera_frame_index(frame_t: np.ndarray, t: np.ndarray, *, max_age_s: float | None = None) -> np.ndarray:
    """``[T]`` int32 index of the latest frame with timestamp ≤ ``t`` (zoh), −1 before the first
    frame (and, with ``max_age_s``, when the latest frame is older than that — e.g. a dropout).
    Indices refer to the file order of ``frame_t`` even if the timestamps are not sorted."""
    ft = np.asarray(frame_t, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64)
    if ft.shape[0] == 0:
        return np.full(t.shape[0], -1, dtype=np.int32)
    order = np.argsort(ft, kind="stable")
    fs = ft[order]
    k = np.searchsorted(fs, t, side="right") - 1
    idx = np.where(k >= 0, order[np.clip(k, 0, None)], -1)
    if max_age_s is not None:
        age = t - fs[np.clip(k, 0, None)]
        idx = np.where((k >= 0) & (age > float(max_age_s)), -1, idx)
    return idx.astype(np.int32)


# ── labels ─────────────────────────────────────────────────────────────────
def phase_ids(phases: Sequence[Mapping[str, Any]], t: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """``phase_id[T]`` int16 (−1 = no phase) and the name vocabulary (order of first appearance).
    A frame belongs to a phase when ``t0 ≤ t < t1``; nested phases: the innermost (later-starting,
    shorter) wins."""
    t = np.asarray(t, dtype=np.float64)
    names: list[str] = []
    for p in sorted(phases, key=lambda p: (float(p["t0"]), float(p["t1"]))):
        if p["name"] not in names:
            names.append(str(p["name"]))
    pid = np.full(t.shape[0], -1, dtype=np.int16)
    for p in sorted(phases, key=lambda p: (float(p["t0"]), -(float(p["t1"]) - float(p["t0"])))):
        m = (t >= float(p["t0"])) & (t < float(p["t1"]))
        pid[m] = names.index(str(p["name"]))
    return pid, names


def _span_mask(t: np.ndarray, spans: Sequence[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.shape[0], dtype=bool)
    for a, b in spans:
        m |= (t >= float(a)) & (t < float(b))
    return m


def contact_labels(t: np.ndarray, segments: Sequence[Mapping[str, Any]], self_touch: np.ndarray | None,
                   n_taxels: int, *, conflict: str = "unknown") -> tuple[np.ndarray, dict]:
    """Contact label policy → ``(contact_label[T,N] int8, counts)``.

    0 in ``no_contact`` segments (``t0 ≤ t < t1``); 1 where ``self_touch``; −1 elsewhere (object
    contact is never inferred here). A self-touch inside a ``no_contact`` segment is a conflict:
    ``unknown`` → −1, ``segment`` → 0, ``geometry`` → 1."""
    if conflict not in _CONFLICT:
        raise ValueError(f"conflict must be one of {_CONFLICT}")
    T = np.asarray(t).shape[0]
    lab = np.full((T, n_taxels), -1, dtype=np.int8)
    nc = _span_mask(np.asarray(t, dtype=np.float64),
                    [(s["t0"], s["t1"]) for s in segments if s.get("label") == "no_contact"])
    lab[nc] = 0
    n_conf = 0
    if self_touch is not None:
        st = np.asarray(self_touch, dtype=bool)
        if st.shape != (T, n_taxels):
            raise ValueError(f"self_touch must be {(T, n_taxels)}, got {st.shape}")
        lab[st] = 1
        conf = st & nc[:, None]
        n_conf = int(conf.sum())
        if conflict == "unknown":
            lab[conf] = -1
        elif conflict == "segment":
            lab[conf] = 0
    counts = {"n_no_contact": int((lab == 0).sum()), "n_contact": int((lab == 1).sum()),
              "n_unknown": int((lab == -1).sum()), "n_conflict": n_conf}
    return lab, counts


# ── helpers per stream ─────────────────────────────────────────────────────
def _imu_sites(layout: Layout, file_sites: Sequence[str] | None, S: int,
               notes: list[str]) -> tuple[list[int], list[str]]:
    """Column indices into the IMU file (layout site order when the layout names sites)."""
    names = [s.name for s in layout.imu_sites]
    if file_sites is None:
        if names and len(names) == S:
            return list(range(S)), names
        return list(range(S)), [f"imu{i}" for i in range(S)]
    file_sites = [str(s) for s in file_sites]
    if not names:
        return list(range(S)), file_sites
    missing = [n for n in names if n not in file_sites]
    if missing:
        notes.append(f"IMU sites {missing} of the layout are missing from imu.npz")
    extra = [n for n in file_sites if n not in names]
    if extra:
        notes.append(f"IMU sites {extra} are not in the layout (dropped)")
    keep = [n for n in names if n in file_sites]
    return [file_sites.index(n) for n in keep], keep


def _wrist_index(layout: Layout, sites: Sequence[str]) -> int:
    parent = {s.name: s.parent for s in layout.imu_sites}
    for i, s in enumerate(sites):
        if parent.get(s) == "wrist" or s == "wrist":
            return i
    return 0


def _resolve_session_file(sdir: Path, ref: str | None, *, as_given: bool = False) -> Path | None:
    """A file referenced by a session: relative to the session dir, then its file name inside the
    session dir; ``as_given`` (config paths) also tries the path itself (absolute / CWD-relative)."""
    if not ref:
        return None
    p = Path(str(ref))
    cands = [p] if p.is_absolute() else [sdir / p]
    cands.append(sdir / p.name)
    if as_given and not p.is_absolute():
        cands.append(p)
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def _first_baseline_rows(t: np.ndarray, spans, sat_any: np.ndarray, trim_s: float) -> tuple[np.ndarray, tuple]:
    for a, b in sorted(spans):
        rows = (t >= a + trim_s) & (t < b) & ~sat_any
        if rows.sum() >= 2:
            return rows, (float(a), float(b))
    return np.zeros(t.shape[0], dtype=bool), ()


# ── main entry ─────────────────────────────────────────────────────────────
def episode_dir(out_root: str | Path, manifest: SessionManifest) -> Path:
    """``<out_root>/<dataset>/<episode_id>`` (episode_id = session_id, path separators replaced)."""
    eid = str(manifest.session_id).replace("/", "_").replace("\\", "_").strip() or "episode"
    return Path(out_root) / manifest.dataset / eid


def preprocess_session(session_dir: str | Path, out_root: str | Path | None = None,
                       cfg: Mapping[str, Any] | None = None, *, pressure_loader: Callable | None = None,
                       overwrite: bool = False) -> Episode:
    """Preprocess one raw session into an :class:`Episode` (see module docstring).

    ``out_root`` = processed root; the episode is written atomically to
    ``<out_root>/<dataset>/<episode_id>/`` (temp dir + rename; an existing episode raises
    ``FileExistsError`` unless ``overwrite``, which replaces it — derived stage outputs included).
    ``out_root=None`` returns the episode in memory only (no camera dirs). ``cfg``: overrides of
    :data:`DEFAULTS` (nested dict, e.g. the stage YAML). ``pressure_loader(session_dir, manifest)
    -> (t[T], raw[T,C])`` replaces the npz loader (e.g. mk555 ``.bin``)."""
    cfg = _merge_cfg(cfg)
    sdir = Path(session_dir).resolve()
    man = SessionManifest.load(sdir)
    notes: list[str] = []
    final = episode_dir(out_root, man) if out_root is not None else None
    if final is not None and (final / EPISODE_JSON).exists() and not overwrite:
        raise FileExistsError(f"episode {final} exists (pass overwrite=True / --force)")
    pre: dict[str, Any] = {"version": PREPROCESS_VERSION, "config_hash": _cfg_hash(cfg),
                           "source_fingerprint": _source_fingerprint(sdir, man), "notes": notes,
                           "synthetic": synthetic_origin(man)}
    layout, layout_ref = resolve_layout(sdir, man, cfg["layout"])
    N = layout.n
    hz = float(cfg["master_hz"] or man.master_hz or MASTER_HZ)
    events = load_events(sdir)
    pc = cfg["pressure"]

    # ── native streams ────────────────────────────────────────────────────
    loader = pressure_loader or _import_callable(pc["loader"]) or load_pressure_npz
    t_p, raw_c = loader(sdir, man)
    t_p, raw_c = _sorted_stream(t_p, np.asarray(raw_c, dtype=np.float64))
    if raw_c.ndim != 2 or raw_c.shape[0] < 2:
        raise ValueError(f"pressure raw must be [T≥2, C], got {raw_c.shape}")
    if int(layout.channels.max()) >= raw_c.shape[1]:
        raise ValueError(f"layout {layout.name!r} uses channel {int(layout.channels.max())} but pressure has "
                         f"{raw_c.shape[1]} channels")
    raw_n = layout.by_channel(raw_c)                                            # [Tp,N] layout order

    imu = None
    if "imu" in man.streams and man.stream_path(sdir, "imu").is_file():
        with np.load(man.stream_path(sdir, "imu")) as z:
            imu = {k: z[k] for k in z.files}
    joints = None
    if "joint_state" in man.streams and man.stream_path(sdir, "joint_state").is_file():
        with np.load(man.stream_path(sdir, "joint_state")) as z:
            joints = {k: z[k] for k in z.files}
    spans = {"pressure": (float(t_p[0]), float(t_p[-1]))}
    if imu is not None:
        spans["imu"] = (float(np.min(imu["t"])), float(np.max(imu["t"])))
    if joints is not None:
        spans["joint_state"] = (float(np.min(joints["t"])), float(np.max(joints["t"])))

    # ── master clock ──────────────────────────────────────────────────────
    ref = ["pressure"] if cfg["clock"]["span"] == "pressure" else \
        [s for s in cfg["clock"]["reference"] if s in spans] or ["pressure"]
    a = max(spans[s][0] for s in ref)
    b = min(spans[s][1] for s in ref)
    t0 = np.ceil(a * hz - 1e-6) / hz
    if b < t0:
        raise ValueError(f"streams {ref} do not overlap in time ({spans})")
    tm = master_clock(t0, b + 1e-9, hz)
    T = tm.shape[0]
    pre.update(master_hz=hz, t_span=[float(tm[0]), float(tm[-1])], clock_reference=ref,
               stream_spans={k: list(v) for k, v in spans.items()})
    arrays: dict[str, np.ndarray] = {K_T: tm}
    max_gap = cfg["clock"]["max_gap_s"]
    gaps: dict[str, int] = {}

    # ── segments: events.jsonl is the source of truth (recorder sessions; a hand-edited boundary
    #    reaches labels and the baseline window), synthetic / hand-written manifests as written ──
    t_end = float(max(tm[-1], t_p[-1]))
    segments, seg_notes = session_segments(man, events, t_end)
    notes += seg_notes

    # ── pressure → ΔS, saturation ─────────────────────────────────────────
    bad_raw = ~np.isfinite(raw_n)                      # e.g. missing packets from an injected loader
    if bad_raw.any():
        raw_n = raw_n.copy()
        for i in np.flatnonzero(bad_raw.any(0)):
            ok = ~bad_raw[:, i]
            raw_n[~ok, i] = np.interp(t_p[~ok], t_p[ok], raw_n[ok, i]) if ok.any() else 0.0
        notes.append(f"{int(bad_raw.sum())} non-finite raw pressure samples: bridged by interpolation and "
                     "flagged saturated (like rail samples)")
    rails = saturation_mask(raw=raw_n, adc_min=pc["adc_min"], adc_max=pc["adc_max"],
                            rail_margin=pc["rail_margin"], max_abs_pct=None) | bad_raw
    raw_f = raw_n
    if pc["lowpass_hz"]:
        raw_f = raw_n.copy()
        for i in np.flatnonzero(rails.any(0) & ~rails.all(0)):              # bridge rail samples first
            ok = ~rails[:, i]
            raw_f[~ok, i] = np.interp(t_p[~ok], t_p[ok], raw_n[ok, i])
        fs = 1.0 / float(np.median(np.diff(t_p)))
        raw_f = _lowpass(raw_f, fs, float(pc["lowpass_hz"]), int(pc["lowpass_order"]))
        raw_f[rails] = raw_n[rails]
        notes.append(f"pressure low-pass {pc['lowpass_hz']} Hz (zero-phase, non-causal)")
    raw_m, _ = _resample(t_p, raw_f, tm)
    sat_native, _ = _resample(t_p, rails.astype(np.float64), tm)
    sat_native = sat_native > 1e-9
    gap_p = _gap_frames(t_p, tm, max_gap)          # dropped samples: the bridged ramp is no measurement
    gaps["pressure"] = int(gap_p.sum())
    if gap_p.any():
        sat_native |= gap_p[:, None]
        notes.append(f"pressure: {int(gap_p.sum())} frames inside sample gaps > {max_gap} s: flagged saturated")
    base, bsrc = _estimate_baseline(tm, raw_m, sat_native, man, layout, events, segments, cfg, notes)
    dead = ~(np.isfinite(base) & (base > 0))
    if dead.any():
        notes.append(f"non-positive baseline for taxels {[layout.taxels[i].id for i in np.flatnonzero(dead)]} "
                     "(dead channel?): marked saturated, ΔS = 0")
    delta = relative_change(raw_m, np.where(dead, 1.0, base))
    delta[:, dead] = 0.0
    sat = saturation_mask(raw=raw_m, delta_pct=delta, adc_min=pc["adc_min"], adc_max=pc["adc_max"],
                          rail_margin=pc["rail_margin"], max_abs_pct=pc["max_abs_pct"]) | sat_native
    sat[:, dead] = True
    arrays[K_PRESSURE_RAW] = raw_m.astype(np.float64)
    arrays[K_DELTA] = delta.astype(np.float32)
    arrays[K_SATURATED] = sat
    pre["baseline"] = bsrc
    pre["saturated_frac"] = float(sat.mean())
    if dead.any():
        pre["dead_taxels"] = [int(i) for i in np.flatnonzero(dead)]

    # ── IMU ──────────────────────────────────────────────────────────────
    imu_sites: list[str] = []
    if imu is not None:
        imu_sites = _process_imu(imu, tm, layout, man, sdir, cfg, arrays, pre, notes)
        gaps["imu"] = _stream_valid(K_IMU_VALID, "imu", imu["t"], tm, max_gap, arrays, notes)

    # ── hand pose (glove labels; an offline product, possibly not registered in the manifest) ──
    hand = None
    hand_file = _stream_file(sdir, man, "hand_pose", notes)
    if hand_file is not None:
        hand = _process_hand(hand_file, tm, cfg, arrays, pre, notes)

    # ── q / qd ───────────────────────────────────────────────────────────
    joint_names: list[str] = []
    model = None
    qc = cfg["qd"]
    if joints is not None:
        joint_names, model = _process_joints(joints, tm, hz, sdir, man, cfg, arrays, pre, notes)
        gaps["joint_state"] = _stream_valid(K_JOINT_VALID, "joint_state", joints["t"], tm, max_gap, arrays, notes)
    elif hand is not None:
        arrays[K_Q] = arrays[K_HAND_FINGERS].reshape(T, 45).astype(np.float32)
        arrays[K_QD] = joint_velocity(arrays[K_Q], hz, method=qc["method"], window_s=qc["window_s"],
                                      polyorder=qc["polyorder"])
        joint_names = list(HAND_Q_NAMES)
        pre["q_source"] = "hand_pose"
    else:
        pre["q_source"] = None
        if man.kind == "glove":
            notes.append("no hand_pose.npz: q/qd absent (IMU-model finger pose is derived by a later stage)")

    # ── taxel poses, self-touch ───────────────────────────────────────────
    st = _taxel_poses(layout, man, hand, model, tm, cfg, arrays, pre, notes)

    # ── object pose ──────────────────────────────────────────────────────
    obj_file = _stream_file(sdir, man, "object_pose", notes)
    if obj_file is not None:
        with np.load(obj_file) as z:
            t_o, pos_o = _sorted_stream(z["t"], z["pos"])
            arrays[K_OBJECT_POS] = _resample(t_o, pos_o.astype(np.float64), tm)[0].astype(np.float32)
            if "quat" in z.files:
                arrays[K_OBJECT_QUAT] = _resample_quat(z["t"], z["quat"], tm)[0].astype(np.float32)
        syn = man.meta.get("synthetic") or {}
        pre["object_frame"] = syn.get("object_frame", man.meta.get("object_frame", "world"))

    # ── phases, segments, contact labels ──────────────────────────────────
    pre["invalid_frames"] = gaps
    phases = phases_from_events(events, t_end=t_end)
    if not phases and segments:
        phases = [{"name": s["label"], "t0": s["t0"], "t1": s["t1"], "value": None, "closed": True,
                   "source": "segments"} for s in segments]
        notes.append("no phase events: phases taken from manifest segments")
    pid, phase_names = phase_ids(phases, tm)
    arrays[K_PHASE] = pid
    meta_phases = []
    for p in phases:
        v = p.get("value") if isinstance(p.get("value"), Mapping) else {}
        d = {"name": str(p["name"]), "t0": round(float(p["t0"]), 6), "t1": round(float(p["t1"]), 6)}
        for k in ("contact", "labels", "kind", "speed"):          # phase_start value (recorder / synthetic)
            if k in v:
                d[k] = list(v[k]) if k == "labels" else v[k]
        if not p.get("closed", True):
            d["closed"] = False
        meta_phases.append(d)
    lab, counts = contact_labels(tm, segments, st, N, conflict=cfg["labels"]["conflict"])
    arrays[K_CONTACT_LABEL] = lab
    pre["contact_label"] = counts
    pre["segments"] = [{"t0": float(s["t0"]), "t1": float(s["t1"]), "label": str(s["label"])} for s in segments]
    markers = [{"t": float(e["t"]), "name": e.get("name", ""), "value": e.get("value")}
               for e in events if e["type"] == "marker"]
    if markers:
        pre["markers"] = markers

    # ── cameras ──────────────────────────────────────────────────────────
    cameras, cam_src = [], {}
    for cam in man.cameras:
        d = man.stream_path(sdir, f"camera_{cam}")
        ts_file = d / "timestamps.npy"
        if not ts_file.is_file():
            notes.append(f"camera {cam!r}: no timestamps.npy (skipped)")
            continue
        ft = np.load(ts_file)
        idx = camera_frame_index(ft, tm, max_age_s=cfg["cameras"]["max_age_s"])
        nf = _n_frames(d)
        if nf is not None and nf != ft.shape[0]:
            notes.append(f"camera {cam!r}: {nf} frames but {ft.shape[0]} timestamps (indices clipped)")
            idx = np.where(idx >= nf, nf - 1, idx).astype(np.int32)
        arrays[cam_idx_key(cam)] = idx
        cameras.append(cam)
        cam_src[cam] = d

    # ── task / instruction ────────────────────────────────────────────────
    task = _task_meta(man, events)

    # ── synthetic ground truth sidecar ────────────────────────────────────
    if (sdir / GT_FILE).is_file() and (cfg["synthetic_gt"]["check"] or cfg["synthetic_gt"]["store"]):
        _synthetic_gt(sdir / GT_FILE, tm, layout, arrays, base, cfg, pre)

    static = {S_BASELINE_RAW: np.asarray(base, dtype=np.float64), S_CHANNELS: layout.channels.astype(np.int64)}
    lo = cfg["layout"]
    pre["layout_source"] = layout_ref or (f"override:{lo.name}" if isinstance(lo, Layout)
                                          else f"override:{lo}" if lo is not None else str(man.layout))
    pre["layout_file"] = LAYOUT_FILE
    pre["config"] = _jsonable(cfg)
    meta = EpisodeMeta(
        episode_id=episode_dir(".", man).name, dataset=man.dataset, kind=man.kind,
        layout=layout_ref if layout_ref else (str(final / LAYOUT_FILE) if final is not None else layout.name),
        n_taxels=N, hz=hz, subject=str(man.subject), joint_names=list(joint_names), imu_sites=list(imu_sites),
        cameras=cameras, phases=_jsonable(meta_phases), phase_names=phase_names, task=_jsonable(task),
        source_session=str(sdir), preprocessing=_jsonable(pre))
    ep = Episode(meta, arrays, static)
    if final is None:
        return ep
    return _write_episode(ep, final, layout, cam_src, cfg, overwrite)


def _estimate_baseline(tm, raw_m, sat_m, man, layout, events, segs, cfg, notes):
    bc = cfg["baseline"]
    sat_any = sat_m.any(1)
    rows, span, src = np.zeros(tm.shape[0], dtype=bool), (), None
    if bc["phase"]:
        ph = [p for p in phases_from_events(events) if p["name"] == bc["phase"]]
        if ph:
            rows, span = _first_baseline_rows(tm, [(ph[0]["t0"], ph[0]["t1"])], sat_any, float(bc["trim_s"]))
            src = f"phase:{bc['phase']}"
        else:
            notes.append(f"baseline phase {bc['phase']!r} not found; using the first no_contact segment")
    if not rows.any():
        rows, span = _first_baseline_rows(tm, [(s["t0"], s["t1"]) for s in segs if s["label"] == "no_contact"],
                                          sat_any, float(bc["trim_s"]))
        src = "no_contact_segment"
    if rows.any():
        base = estimate_baseline(raw_m[rows], t=tm[rows], duration_s=float(bc["duration_s"]))
        used = tm[rows][(tm[rows] - tm[rows][0]) < float(bc["duration_s"])]
        return base, {"source": src, "segment": list(span), "t0": float(used[0]), "t1": float(used[-1]),
                      "n_frames": int(used.shape[0])}
    if bc["use_manifest"] and man.baseline is not None:
        b = np.asarray(man.baseline, dtype=np.float64)
        if b.shape[0] > int(layout.channels.max()):
            notes.append("no usable no_contact segment: baseline from manifest.baseline")
            return layout.by_channel(b), {"source": "manifest"}
    notes.append(f"no usable no_contact segment: baseline from the first {bc['fallback_s']} s of the recording")
    ok = ~sat_any if (~sat_any).sum() >= 2 else np.ones(tm.shape[0], dtype=bool)
    base = estimate_baseline(raw_m[ok], t=tm[ok], duration_s=float(bc["fallback_s"]))
    return base, {"source": "initial", "duration_s": float(bc["fallback_s"])}


def _process_imu(imu, tm, layout, man, sdir, cfg, arrays, pre, notes) -> list[str]:
    from robot_skin.pose.imu_model import (
        apply_imu_offsets, apply_imu_offsets_to_vectors, imu_calibration_from_dict,
    )

    q = np.asarray(imu["quat"], dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"imu quat must be [T,S,4], got {q.shape}")
    cols, sites = _imu_sites(layout, imu.get("sites"), q.shape[1], notes)
    q = q[:, cols]
    gyro = None if "gyro" not in imu else np.asarray(imu["gyro"], dtype=np.float64)[:, cols]
    acc = None if "acc" not in imu else np.asarray(imu["acc"], dtype=np.float64)[:, cols]
    ic = cfg["imu"]
    vec_frame = str(ic.get("vec_frame") or "sensor")
    if vec_frame not in ("sensor", "world"):
        raise ValueError(f"imu.vec_frame must be 'sensor' or 'world', got {vec_frame!r}")
    # vec_frame: frame of the stored gyro/acc — what pose.imu_model.imu_features(vec_frame=...) must be told
    info: dict[str, Any] = {"calibrated": False, "calibration_source": None, "wrist_index": _wrist_index(layout, sites),
                            "frame": "sensor", "vec_frame": vec_frame}
    offsets = world = None
    calib = dict(man.calibration or {})
    if (calib.get("imu_offsets") is not None and calib.get("imu_sites") is None and imu.get("sites") is not None
            and np.asarray(calib["imu_offsets"]).reshape(-1, 4).shape[0] == len(imu["sites"])):
        # offsets are in IMU *file* site order (DATA_FORMAT §1.8) while ``sites`` is the layout
        # order the columns were just permuted to: name them so they are reordered, not misapplied
        calib["imu_sites"] = [str(x) for x in imu["sites"]]
        if ic["apply_calibration"]:
            notes.append("calibration has no imu_sites: imu_offsets taken in imu.npz site order")
    if ic["apply_calibration"]:
        offsets, world = imu_calibration_from_dict(calib, sites)
        if offsets is not None:
            info["calibration_source"] = "manifest"
        elif ic["calibrate_if_missing"]:
            from robot_skin.acquisition.calibration import calibrate_session_imu

            c = calibrate_session_imu(sdir, phase=ic["calibration_phase"], layout=layout,
                                      skeleton=_skeleton(cfg), save=False)
            if c is not None:
                offsets, world = imu_calibration_from_dict(c, sites)
                info["calibration_source"] = f"phase:{ic['calibration_phase']}"
                info["quality"] = c.get("imu_calibration_quality")
        if offsets is None:
            notes.append("no IMU calibration in manifest.calibration: IMU streams left in the raw sensor frames")
    if offsets is not None:
        q = apply_imu_offsets(q, offsets, world=world)
        # sensor-frame vectors: R_offᵀ v (segment frame); world-frame vectors: G⁻¹ v (model world)
        vkw = {"vec_frame": vec_frame, "world": world}
        gyro = None if gyro is None else apply_imu_offsets_to_vectors(gyro, offsets, **vkw)
        acc = None if acc is None else apply_imu_offsets_to_vectors(acc, offsets, **vkw)
        info.update(calibrated=True, frame="segment", world_aligned=world is not None)
    arrays[K_IMU_QUAT] = _resample_quat(imu["t"], q, tm)[0].astype(np.float32)
    t_i = np.asarray(imu["t"], dtype=np.float64)
    for key, v in ((K_IMU_GYRO, gyro), (K_IMU_ACC, acc)):
        if v is not None:
            ts, vs = _sorted_stream(t_i, v)
            arrays[key] = _resample(ts, vs, tm)[0].astype(np.float32)
    pre["imu"] = info
    return sites


def _stream_valid(key: str, name: str, t_src, tm: np.ndarray, max_gap_s: float | None,
                  arrays: dict, notes: list[str]) -> int:
    """``arrays[key]`` = ``[T]`` bool: master frames the native stream actually measured — inside its
    own time span (``clock.span: pressure`` edge-holds it outside) and not inside a sample gap
    (:func:`_gap_frames`). Returns the number of invalid frames (noted when > 0)."""
    t = np.asarray(t_src, dtype=np.float64).reshape(-1)
    span = (tm >= np.min(t)) & (tm <= np.max(t))
    gap = _gap_frames(t, tm, max_gap_s)
    valid = span & ~gap
    arrays[key] = valid
    if (~span).any():
        notes.append(f"{name}: {int((~span).sum())} frames outside its time span (edge-held): {key} false")
    if gap.any():
        notes.append(f"{name}: {int(gap.sum())} frames inside sample gaps > {max_gap_s} s: {key} false")
    return int((~valid).sum())


def _process_hand(path, tm, cfg, arrays, pre, notes):
    import torch

    from robot_skin.geometry.rotations import aa_to_quat, quat_to_aa
    from robot_skin.pose.vision_hand import load_hand_labels, smooth_hand_labels

    hc = cfg["hand_pose"]
    hl = load_hand_labels(path)
    if hl["t"].shape[0] == 0:                         # e.g. the vision pass found no hand at all
        notes.append(f"{path.name} has no labels: treated as absent (no hand pose / q)")
        return None
    sm = smooth_hand_labels(hl["t"], hl["global_orient"], hl["finger_pose"], hl["wrist_pos"], hl["confidence"],
                            min_conf=float(hc["min_conf"]), max_gap_s=float(hc["max_gap_s"]),
                            cutoff_hz=None if hc["cutoff_hz"] is None else float(hc["cutoff_hz"]),
                            order=int(hc["order"]))
    t_h = sm["t"]
    rot = np.concatenate([sm["global_orient"][:, None], sm["finger_pose"]], axis=1).astype(np.float64)
    qh = aa_to_quat(torch.as_tensor(rot)).numpy()                                   # [Th,16,4]
    qm, inside = _resample_quat(t_h, qh, tm)
    aa = quat_to_aa(torch.as_tensor(qm)).numpy()
    wp, _ = _resample(t_h, sm["wrist_pos"].astype(np.float64), tm)
    vf, _ = _resample(t_h, sm["valid"].astype(np.float64), tm)
    valid = inside & (vf >= 1.0 - 1e-6)
    arrays[K_HAND_GLOBAL] = aa[:, 0].astype(np.float32)
    arrays[K_HAND_FINGERS] = aa[:, 1:].astype(np.float32)
    arrays[K_HAND_WRIST] = wp.astype(np.float32)
    arrays[K_HAND_VALID] = valid
    pre["hand_pose"] = {"valid_frac": float(valid.mean()), "label_valid_frac": float(np.mean(sm["valid"])),
                        "n_labels": int(t_h.shape[0])}
    return {"go": aa[:, 0], "fp": aa[:, 1:], "wp": wp, "valid": valid}


def _process_joints(joints, tm, hz, sdir, man, cfg, arrays, pre, notes):
    from robot_skin.pose.urdf import URDFModel

    t_j = np.asarray(joints["t"], dtype=np.float64)
    q = np.asarray(joints["q"], dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"joint_state q must be [T,D], got {q.shape}")
    names = [str(n) for n in joints["names"]] if "names" in joints else [f"joint{i}" for i in range(q.shape[1])]
    qd_file = np.asarray(joints["qd"], dtype=np.float64) if "qd" in joints else None
    cfg_urdf = cfg["robot"]["urdf"]
    urdf_ref = cfg_urdf or man.meta.get("urdf")
    urdf = _resolve_session_file(sdir, urdf_ref, as_given=bool(cfg_urdf))
    if cfg_urdf and urdf is None:           # an explicit config path must not silently degrade q / poses
        raise FileNotFoundError(f"cfg robot.urdf {cfg_urdf!r} not found (tried relative to {sdir} and as given)")
    model = None
    if urdf is not None:
        model = URDFModel.from_file(urdf)
        missing = [n for n in model.joint_names if n not in names]
        if len(missing) == len(model.joint_names):
            raise ValueError(f"joint_state names {names[:4]}… match none of the URDF {urdf.name} joints "
                             f"{list(model.joint_names)[:4]}… (namespace prefix / other robot?): q would be all "
                             "zeros — rename the driver joints or give the matching URDF (robot.urdf)")
        if missing:
            msg = (f"joint_state lacks {len(missing)}/{len(model.joint_names)} URDF joints {missing}: "
                   "q / qd are 0 there (static joints)")
            notes.append(msg)
            log.warning("%s: %s", sdir, msg)
            pre["zero_filled_joints"] = missing
        with warnings.catch_warnings():               # reported above (notes + meta), once per session
            warnings.simplefilter("ignore")
            q = model.reorder_q(q, names)
            qd_file = None if qd_file is None else model.reorder_q(qd_file, names)
        names = list(model.joint_names)
        pre["urdf"] = str(urdf)
    else:
        msg = (f"URDF {urdf_ref!r} not found" if urdf_ref else "no URDF (cfg robot.urdf / manifest.meta.urdf)")
        msg += ": q kept in joint_state (driver) order, taxel poses static"
        notes.append(msg)
        log.warning("%s: %s", sdir, msg)
    ts, qs, qds = _sorted_stream(t_j, q, qd_file)
    qm = _resample(ts, qs, tm)[0]
    arrays[K_Q] = qm.astype(np.float32)
    qc = cfg["qd"]
    if qc["source"] == "file" and qds is not None:
        arrays[K_QD] = _resample(ts, qds, tm)[0].astype(np.float32)
        pre["qd_source"] = "file"
    else:
        arrays[K_QD] = joint_velocity(qm, hz, method=qc["method"], window_s=qc["window_s"], polyorder=qc["polyorder"])
        pre["qd_source"] = qc["method"]
    pre["q_source"] = "joint_state"
    return names, model


def _skeleton(cfg):
    from robot_skin.pose.mano import ManoSkeleton

    path = cfg["hand_pose"]["mano_model"]
    return ManoSkeleton.from_mano_pkl(path) if path else ManoSkeleton()


def _taxel_poses(layout, man, hand, model, tm, cfg, arrays, pre, notes):
    """Taxel poses (hand / robot base frame, see module docstring) + geometric self-touch;
    returns self_touch[T,N] or None."""
    T, N = tm.shape[0], layout.n
    st = None
    if layout.parent_frame == "mano" and hand is not None:
        from robot_skin.pose.mano import self_touch_from_hand, taxel_poses_from_hand

        sk = _skeleton(cfg)
        # hand frame: global_orient = 0, wrist at the origin (only the finger pose moves taxels);
        # self-touch distances are invariant to that rigid transform
        go0 = np.zeros_like(np.asarray(hand["go"], dtype=np.float64))
        pos, nrm = taxel_poses_from_hand(layout, sk, go0, hand["fp"], None)
        src, frame = "hand_pose", "mano_wrist"
        if cfg["self_touch"]["enabled"]:
            st = self_touch_from_hand(layout, sk, go0, hand["fp"], None,
                                      margin=float(cfg["self_touch"]["margin_m"])) & hand["valid"][:, None]
            arrays[K_SELF_TOUCH] = st
            pre["self_touch"] = {"margin_m": float(cfg["self_touch"]["margin_m"]), "frac": float(st.mean())}
    elif layout.parent_frame == "mano":
        from robot_skin.pose.mano import taxel_poses_from_hand

        p0, n0 = taxel_poses_from_hand(layout, _skeleton(cfg), np.zeros(3), np.zeros((15, 3)))
        pos, nrm = np.broadcast_to(p0, (T, N, 3)), np.broadcast_to(n0, (T, N, 3))
        src, frame = "rest", "mano_wrist"
        notes.append("glove without hand pose: taxel poses of the flat rest hand (constant)")
    elif layout.parent_frame == "urdf" and model is not None:
        from robot_skin.pose.robot_fk import taxel_poses_from_joints

        pos, nrm = taxel_poses_from_joints(layout, model, arrays[K_Q].astype(np.float64))
        src, frame = "urdf", "urdf_root"
    else:
        pos = np.broadcast_to(layout.positions, (T, N, 3))
        nrm = np.broadcast_to(layout.normals, (T, N, 3))
        src, frame = "static", "layout"
    arrays[K_TAXEL_POS] = np.ascontiguousarray(pos, dtype=np.float32)
    arrays[K_TAXEL_NRM] = np.ascontiguousarray(nrm, dtype=np.float32)
    pre["taxel_pose_source"] = src
    pre["taxel_frame"] = frame
    return st


def _n_frames(cam_dir: Path) -> int | None:
    npy = cam_dir / "frames.npy"
    if npy.is_file():
        return int(np.load(npy, mmap_mode="r").shape[0])
    jpgs = list(cam_dir.glob("*.jpg"))
    return len(jpgs) if jpgs else None


def _task_meta(man: SessionManifest, events: Sequence[Mapping]) -> dict | None:
    """``manifest.task`` (all keys kept) + instruction / success from the last ``instruction`` /
    ``success`` event (keyed on the event *type*) when the manifest lacks them."""
    task = dict(man.task) if man.task else None
    ins = [e for e in events if e["type"] == "instruction" and isinstance(e.get("value"), str) and e["value"].strip()]
    suc = [e for e in events if e["type"] == "success" and e.get("value") is not None]
    if task is None and not ins:
        return None
    task = task or {}
    if not task.get("instruction") and ins:
        task["instruction"] = ins[-1]["value"].strip()
        task.setdefault("instruction_source", "events")
    if task.get("success") is None and suc:
        task["success"] = bool(suc[-1]["value"])
    return task


def _synthetic_gt(path: Path, tm, layout, arrays, base, cfg, pre) -> None:
    with np.load(path) as z:
        g = {k: z[k] for k in z.files}
    t_g, order_ok = g["t"], True
    if "channels" in g and not np.array_equal(np.asarray(g["channels"]), layout.channels):
        order_ok = False
    rep: dict[str, Any] = {"layout_order_match": order_ok}
    if not order_ok:
        pre["synthetic_gt"] = rep
        return
    lin = {k: _resample(t_g, np.asarray(g[k], dtype=np.float64), tm)[0]
           for k in ("artefact_pct", "press_pct", "delta_true_pct", "noise_pct") if k in g}
    flags = {k: _resample(t_g, np.asarray(g[k], dtype=np.float64), tm)[0] > 0.5
             for k in ("contact", "self_touch", "object_contact", "saturated") if k in g}
    if cfg["synthetic_gt"]["check"] and "delta_true_pct" in lin:
        target = lin["delta_true_pct"] + lin.get("noise_pct", 0.0)
        gt_sat = _resample(t_g, np.asarray(g.get("saturated", np.zeros((len(t_g), layout.n))), dtype=np.float64),
                           tm)[0] > 1e-9
        ok = ~arrays[K_SATURATED] & ~gt_sat
        err = arrays[K_DELTA].astype(np.float64) - target
        e = np.abs(err[ok])
        rep.update(delta_abs_err_pct={"median": float(np.median(e)) if e.size else None,
                                      "p99": float(np.percentile(e, 99)) if e.size else None},
                   n_frames=int(ok.sum()))
        if "baseline_raw" in g:
            rep["baseline_rel_err"] = [float(v) for v in base / np.asarray(g["baseline_raw"], dtype=np.float64) - 1.0]
        if "saturated" in flags:
            s_gt = flags["saturated"]
            rep["saturated_recall"] = float(arrays[K_SATURATED][s_gt].mean()) if s_gt.any() else None
    if cfg["synthetic_gt"]["store"]:
        for k, key in GT_KEYS.items():
            if k in lin:
                arrays[key] = lin[k].astype(np.float32)
            elif k in flags:
                arrays[key] = flags[k]
        rep["stored"] = sorted(v for v in GT_KEYS.values() if v in arrays)
    pre["synthetic_gt"] = rep


def _write_episode(ep: Episode, final: Path, layout: Layout, cam_src: Mapping[str, Path], cfg, overwrite) -> Episode:
    if (final / EPISODE_JSON).exists() and not overwrite:
        raise FileExistsError(f"episode {final} exists (pass overwrite=True / --force)")
    if final.exists() and not (final / EPISODE_JSON).exists() and any(final.iterdir()):
        raise FileExistsError(f"{final} exists and is not an episode directory; refusing to replace it")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ep.save(tmp)
        (tmp / LAYOUT_FILE).write_text(yaml.safe_dump(layout.to_dict(units="mm"), sort_keys=False))
        mode = cfg["cameras"]["copy_frames"]
        mode = {True: "copy", False: "none"}.get(mode, mode)
        for cam, src in cam_src.items():
            dst = tmp / f"camera_{cam}"
            if mode == "symlink":
                try:
                    os.symlink(src.resolve(), dst, target_is_directory=True)
                    continue
                except OSError:
                    log.warning("symlink of %s failed; copying the frames instead", src)
            if mode in ("symlink", "copy"):
                shutil.copytree(src, dst, symlinks=False)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    ep.root = final
    return ep


# ── many sessions / CLI ────────────────────────────────────────────────────
def find_sessions(root: str | Path) -> list[Path]:
    """Session dirs (containing ``session.json``) at or below ``root``, sorted. Hidden dirs and
    processed episodes are skipped. A missing ``root`` raises ``FileNotFoundError`` (a typo in
    ``--raw`` must not look like an empty, successful build)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"raw root {root} does not exist")
    if (root / MANIFEST_NAME).is_file():
        return [root]
    out = []
    for p in root.rglob(MANIFEST_NAME):
        rel = p.parent.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or (p.parent / EPISODE_JSON).exists():
            continue
        out.append(p.parent)
    return sorted(out)


def synthetic_origin(man: SessionManifest) -> str | None:
    """``"fake_recorder"`` for a ``record … --fake`` session (``meta.fake``), ``"generator"`` for a
    ``datasets.synthetic`` session (``meta.synthetic`` / ``meta.generator``), ``None`` for recorded data —
    copied into the episode's ``meta.preprocessing.synthetic``."""
    meta = man.meta or {}
    if meta.get("fake"):
        return "fake_recorder"
    if meta.get("synthetic") or str(meta.get("generator") or "").startswith("robot_skin.datasets.synthetic"):
        return "generator"
    return None


def _qc_failed(session_dir: Path) -> bool:
    p = session_dir / "qc.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text()).get("passed") is False
    except (OSError, ValueError):
        return False


def build_all(raw_roots: str | Path | Sequence[str | Path], out_root: str | Path,
              cfg: Mapping[str, Any] | None = None, *, force: bool = False, fail_fast: bool = False,
              pressure_loader: Callable | None = None) -> list[dict]:
    """Preprocess every session under ``raw_roots`` into ``out_root``. Idempotent: sessions whose
    episode already exists are ``skipped`` (``force`` rebuilds them); a skipped episode built with
    another config / preprocessing version / raw file set is reported ``stale``. Two sessions that map
    to the same ``<dataset>/<session_id>`` in one run are an error for the second (never a silent
    overwrite / skip). ``record --dry-run`` plans (``meta.dry_run``: nothing recorded) are reported
    ``plan`` and skipped; synthetic sessions (``record --fake``, ``datasets.synthetic``) mixed with
    recorded ones are warned about. Returns one report per session ``{session, episode, status:
    built|skipped|qc_failed|plan|failed, stale?, synthetic?, error?}``."""
    cfg = _merge_cfg(cfg)
    roots = [raw_roots] if isinstance(raw_roots, (str, Path)) else list(raw_roots)
    sessions = sorted({s.resolve() for r in roots for s in find_sessions(r)})
    reports = []
    claimed: dict[Path, Path] = {}                     # episode dir → the session that owns it in this run
    for s in sessions:
        rep: dict[str, Any] = {"session": str(s)}
        try:
            man = SessionManifest.load(s)
            if (man.meta or {}).get("dry_run"):        # a `record --dry-run` plan: no streams were recorded
                rep["status"] = "plan"
                log.info("%s: dry-run plan (nothing recorded) — skipped", s)
                reports.append(rep)
                continue
            origin = synthetic_origin(man)
            if origin:
                rep["synthetic"] = origin
            dest = episode_dir(out_root, man)
            rep["episode"] = str(dest)
            key = dest.resolve()
            if key in claimed:                         # same <dataset>/<session_id>: never overwrite silently
                raise ValueError(f"episode id collision: {dest} is also the episode of {claimed[key]} "
                                 "(give the sessions distinct session_id values)")
            claimed[key] = s
            if (dest / EPISODE_JSON).is_file() and not force:
                old = json.loads((dest / EPISODE_JSON).read_text()).get("preprocessing", {})
                rep["status"] = "skipped"
                if (old.get("config_hash") != _cfg_hash(cfg) or old.get("version") != PREPROCESS_VERSION
                        or old.get("source_fingerprint") != _source_fingerprint(s, man)):
                    rep["stale"] = True
                    log.info("%s: built from another config / version / raw content (use --force to rebuild)",
                             dest)
            elif cfg["qc"]["skip_failed"] and _qc_failed(s):
                rep["status"] = "qc_failed"
            else:
                preprocess_session(s, out_root, cfg, pressure_loader=pressure_loader, overwrite=force)
                rep["status"] = "built"
        except Exception as e:  # noqa: BLE001 - reported per session
            if fail_fast:
                raise
            rep.update(status="failed", error=f"{type(e).__name__}: {e}")
            log.error("%s: %s", s, rep["error"])
        reports.append(rep)
    syn = [r for r in reports if r.get("synthetic")]
    real = [r for r in reports if r.get("episode") and not r.get("synthetic")]
    if syn and real:
        log.warning("%d synthetic session(s) (record --fake / datasets.synthetic) are preprocessed together with %d "
                    "recorded one(s) into %s — synthetic data must not be mixed with real data (keep it under "
                    "paths.synthetic_root and give it its own --out), e.g. %s", len(syn), len(real), out_root,
                    syn[0]["session"])
    return reports


def _parse_set(items: Sequence[str]) -> dict:
    out: dict = {}
    for it in items:
        key, sep, val = it.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {it!r}")
        d = out
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: preprocess raw sessions into episodes. Exit code 0 = no failures, 1 = some session failed,
    2 = a --raw root does not exist."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.datasets.build", description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="append", default=None,
                    help="raw root or session dir (repeatable; default: config raw_root)")
    ap.add_argument("--out", default=None,
                    help="processed root, episodes go to <out>/<dataset>/<episode_id> (default: config out_root)")
    ap.add_argument("--config", default=None, help=f"stage YAML (default: built-in defaults = {CONFIG_PATH.name})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. qd.method=savgol_causal")
    ap.add_argument("--force", action="store_true", help="rebuild existing episodes")
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first failing session")
    ap.add_argument("--json", default=None, help="write the per-session report to this file")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_preprocess_config(args.config, _parse_set(args.set))
    raw_roots = args.raw or [cfg["raw_root"]]
    missing = [r for r in raw_roots if not Path(r).exists()]
    if missing:
        print(f"error: raw root(s) not found: {missing}", file=sys.stderr)
        return 2
    reps = build_all(raw_roots, args.out or cfg["out_root"], cfg, force=args.force, fail_fast=args.fail_fast)
    counts: dict[str, int] = {}
    for r in reps:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if not args.quiet:
            extra = f"  ({r['error']})" if "error" in r else (" (stale)" if r.get("stale") else "")
            extra += "  (dry-run plan, nothing recorded)" if r["status"] == "plan" else ""
            print(f"{r['status']:9s} {r['session']}{extra}")
    print("summary: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no sessions found"))
    if args.json:
        Path(args.json).write_text(json.dumps(reps, indent=2, ensure_ascii=False))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
