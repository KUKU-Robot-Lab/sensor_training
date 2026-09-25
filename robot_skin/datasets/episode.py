"""Processed Episode — the single on-disk format every training stage reads.

A raw session (``acquisition``) is preprocessed (``datasets.build``) into one Episode: all
streams resampled onto the master clock (200 Hz) in **layout order**, plus labels. Layout::

    robot_skin/data/processed/<dataset>/<episode_id>/
      episode.json            # EpisodeMeta
      arrays/<key>.npy        # time-major arrays [T, ...]  (np.load(mmap_mode="r") friendly)
      static/<key>.npy        # time-invariant arrays (e.g. baseline_raw[N])
      derived/<key>.npy       # outputs of later stages (baseline_pred, residual, contact_prob …)
      camera_<name>/          # copied/linked frames: frames.npy uint8[F,H,W,3] or 000000.jpg …
                              #   + timestamps.npy [F]; cam_<name>_idx maps master clock → frame

Canonical keys are the ``K_*`` constants below; a stage must not invent a synonym for one of
them. Optional keys may be absent (``episode.has(key)``). See ``docs/DATA_FORMAT.md``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

EPISODE_JSON = "episode.json"
FORMAT_VERSION = 1

# ── canonical time-major array keys ([T, ...] on the master clock) ────────────
K_T = "t"                              # [T] float64 s from session start
K_PRESSURE_RAW = "pressure_raw"        # [T,N] float64, layout order
K_DELTA = "delta_pct"                  # [T,N] float32, ΔS% (SATS sign: press → negative)
K_SATURATED = "saturated"              # [T,N] bool
K_TAXEL_POS = "taxel_pos"              # [T,N,3] float32 m, hand/robot base frame (glove: MANO wrist
                                       #   frame, go = 0 & wrist at origin; robot: URDF root;
                                       #   meta.preprocessing["taxel_frame"])
K_TAXEL_NRM = "taxel_nrm"              # [T,N,3] float32 unit
K_Q = "q"                              # [T,D] float32 joint state (robot joints | glove: finger_pose flat 45)
K_QD = "qd"                            # [T,D] float32 (smoothed derivative)
K_IMU_QUAT = "imu_quat"                # [T,S,4] float32 wxyz (calibrated, continuity-fixed)
K_IMU_GYRO = "imu_gyro"                # [T,S,3] float32 rad/s
K_IMU_ACC = "imu_acc"                  # [T,S,3] float32 m/s²
K_IMU_VALID = "imu_valid"              # [T] bool IMU measured there (inside its span, no gap > clock.max_gap_s)
K_JOINT_VALID = "joint_state_valid"    # [T] bool joint_state measured there (same rule; robot q/qd)
K_HAND_GLOBAL = "hand_global_orient"   # [T,3] float32 MANO axis-angle (wrist)
K_HAND_FINGERS = "hand_finger_pose"    # [T,15,3] float32 MANO axis-angle, MANO joint order
K_HAND_WRIST = "hand_wrist_pos"        # [T,3] float32 m
K_HAND_VALID = "hand_pose_valid"       # [T] bool (label present & confident)
K_OBJECT_POS = "object_pos"            # [T,3] float32 (optional)
K_OBJECT_QUAT = "object_quat"          # [T,4] float32 wxyz (optional)
K_PHASE = "phase_id"                   # [T] int16, index into meta.phase_names, -1 = none
K_SELF_TOUCH = "self_touch"            # [T,N] bool, geometric self-touch label
K_CONTACT_LABEL = "contact_label"      # [T,N] int8: -1 unknown, 0 no contact, 1 contact


def cam_idx_key(camera: str) -> str:
    """``cam_<name>_idx`` [T] int32: latest frame index with timestamp ≤ t, -1 before first."""
    return f"cam_{camera}_idx"


# ── static keys ────────────────────────────────────────────────────────────────
S_BASELINE_RAW = "baseline_raw"        # [N] float64
S_CHANNELS = "taxel_channels"          # [N] int64 raw channel per layout taxel

# ── derived keys (written by later stages) ─────────────────────────────────────
D_BASELINE_PRED = "baseline_pred"      # [T,N] float32 predicted no-contact ΔS%
D_BASELINE_LOGVAR = "baseline_logvar"  # [T,N] float32
D_RESIDUAL = "residual"                # [T,N] float32 = delta_pct − baseline_pred
D_RESIDUAL_Z = "residual_z"            # [T,N] float32 calibrated z-score (press-positive)
D_CONTACT_PROB = "contact_prob"        # [T,N] float32
D_LEVEL = "contact_level"              # [T,N] int8 ContactLevel
D_HAND_POSE_IMU = "hand_finger_pose_imu"  # [T,15,3] float32 IMU-estimated finger pose
D_CONTACT_LABEL_PSEUDO = "contact_label_pseudo"  # [T,N] int8 D2 pseudo contact labels (−1/0/1, contact stage)


@dataclass
class EpisodeMeta:
    episode_id: str
    dataset: str                        # motion | task | other
    kind: str                           # glove | robot | bench
    layout: str
    n_taxels: int
    hz: float = 200.0
    subject: str = ""
    joint_names: list[str] = field(default_factory=list)
    imu_sites: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)       # {name, t0, t1} (s, episode clock)
    phase_names: list[str] = field(default_factory=list)   # vocabulary for phase_id
    task: dict | None = None                               # {task_id, instruction, object, success}
    source_session: str = ""
    preprocessing: dict = field(default_factory=dict)      # params + code version
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = FORMAT_VERSION

    @property
    def instruction(self) -> str | None:
        return None if not self.task else self.task.get("instruction")


class Episode:
    """In-memory or lazily loaded (mmap) episode."""

    def __init__(self, meta: EpisodeMeta, arrays: dict[str, np.ndarray],
                 static: dict[str, np.ndarray] | None = None, root: Path | None = None):
        self.meta = meta
        self.arrays = dict(arrays)
        self.static = dict(static or {})
        self.root = Path(root) if root is not None else None
        self._derived: dict[str, np.ndarray] = {}
        self.validate()

    # ── basic views ───────────────────────────────────────────────────────
    @property
    def T(self) -> int:
        return int(self.arrays[K_T].shape[0])

    @property
    def t(self) -> np.ndarray:
        return self.arrays[K_T]

    def __len__(self) -> int:
        return self.T

    def has(self, key: str) -> bool:
        return key in self.arrays

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def validate(self) -> None:
        if K_T not in self.arrays:
            raise ValueError("episode needs a 't' array")
        T = self.arrays[K_T].shape[0]
        for k, v in self.arrays.items():
            if v.shape[0] != T:
                raise ValueError(f"array {k!r} has {v.shape[0]} rows, expected {T}")
        for k in (K_PRESSURE_RAW, K_DELTA, K_SATURATED):
            if k in self.arrays and self.arrays[k].shape[1] != self.meta.n_taxels:
                raise ValueError(f"{k!r} has {self.arrays[k].shape[1]} taxels, meta says {self.meta.n_taxels}")

    def phase_mask(self, name: str) -> np.ndarray:
        """Bool [T] for frames inside phase ``name``."""
        if K_PHASE not in self.arrays or name not in self.meta.phase_names:
            return np.zeros(self.T, dtype=bool)
        return self.arrays[K_PHASE] == self.meta.phase_names.index(name)

    # ── derived (stage outputs) ───────────────────────────────────────────
    def has_derived(self, key: str) -> bool:
        return key in self._derived or (self.root is not None and (self.root / "derived" / f"{key}.npy").is_file())

    def derived(self, key: str) -> np.ndarray:
        if key not in self._derived:
            if self.root is None:
                raise KeyError(key)
            self._derived[key] = np.load(self.root / "derived" / f"{key}.npy", mmap_mode="r")
        return self._derived[key]

    def set_derived(self, key: str, value: np.ndarray, *, save: bool = True) -> None:
        value = np.asarray(value)
        if value.shape[0] != self.T:
            raise ValueError(f"derived {key!r} has {value.shape[0]} rows, expected {self.T}")
        self._derived[key] = value
        if save and self.root is not None:
            # atomic: write a temp file, then os.replace — a reader holding a memmap of the old file
            # keeps its (old) inode instead of seeing a truncated / half-written array
            d = self.root / "derived"
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{key}.npy.tmp{os.getpid()}"
            try:
                with open(tmp, "wb") as fh:
                    np.save(fh, value)
                os.replace(tmp, d / f"{key}.npy")
            finally:
                if tmp.exists():
                    tmp.unlink()

    # ── cameras ───────────────────────────────────────────────────────────
    def frame_timestamps(self, camera: str) -> np.ndarray:
        return np.load(self._cam_dir(camera) / "timestamps.npy")

    def get_frame(self, camera: str, frame_idx: int) -> np.ndarray:
        """uint8 [H,W,3]. Reads ``frames.npy`` (mmap) or ``%06d.jpg`` (needs Pillow)."""
        d = self._cam_dir(camera)
        npy = d / "frames.npy"
        if npy.is_file():
            return np.asarray(np.load(npy, mmap_mode="r")[int(frame_idx)])
        from PIL import Image  # optional dependency, only for jpg storage
        return np.asarray(Image.open(d / f"{int(frame_idx):06d}.jpg").convert("RGB"))

    def frame_at(self, camera: str, i: int) -> np.ndarray | None:
        """Frame shown at master-clock index ``i`` (None before the first frame)."""
        idx = int(self.arrays[cam_idx_key(camera)][i])
        return None if idx < 0 else self.get_frame(camera, idx)

    def _cam_dir(self, camera: str) -> Path:
        if self.root is None:
            raise ValueError("camera frames need an on-disk episode (root)")
        return self.root / f"camera_{camera}"

    # ── io ────────────────────────────────────────────────────────────────
    def save(self, root: str | Path) -> Path:
        root = Path(root)
        (root / "arrays").mkdir(parents=True, exist_ok=True)
        for k, v in self.arrays.items():
            np.save(root / "arrays" / f"{k}.npy", np.asarray(v))
        if self.static:
            (root / "static").mkdir(exist_ok=True)
            for k, v in self.static.items():
                np.save(root / "static" / f"{k}.npy", np.asarray(v))
        (root / EPISODE_JSON).write_text(json.dumps(asdict(self.meta), indent=2, ensure_ascii=False))
        self.root = root
        for k, v in self._derived.items():
            self.set_derived(k, v, save=True)
        return root

    @classmethod
    def load(cls, root: str | Path, *, mmap: bool = True, keys: Iterable[str] | None = None) -> "Episode":
        root = Path(root)
        d = json.loads((root / EPISODE_JSON).read_text())
        if d.get("format_version", FORMAT_VERSION) > FORMAT_VERSION:
            raise ValueError(f"episode format {d['format_version']} newer than supported {FORMAT_VERSION}")
        meta = EpisodeMeta(**d)
        want = None if keys is None else set(keys) | {K_T}
        mode = "r" if mmap else None
        arrays = {p.stem: np.load(p, mmap_mode=mode) for p in sorted((root / "arrays").glob("*.npy"))
                  if want is None or p.stem in want}
        static = {p.stem: np.load(p) for p in sorted((root / "static").glob("*.npy"))} \
            if (root / "static").is_dir() else {}
        return cls(meta, arrays, static, root=root)


def list_episodes(processed_root: str | Path, dataset: str | None = None) -> list[Path]:
    """All episode dirs under ``processed_root`` (optionally one dataset), sorted."""
    base = Path(processed_root)
    if dataset is not None:
        base = base / dataset
    return sorted(p.parent for p in base.rglob(EPISODE_JSON))
