"""SessionManifest — one JSON per recorded (raw) session describing its streams.

Layout on disk (``robot_skin/data/`` is git-ignored)::

    robot_skin/data/raw/<dataset>/<subject>/<session_id>/
      session.json                  # this manifest
      pressure.npz                  # t[T] (s, host clock), raw[T,C] (channel order)
      imu.npz                       # t, quat[T,S,4] wxyz, gyro[T,S,3] rad/s, acc[T,S,3] m/s², sites[S]
      joint_state.npz               # t, q[T,D], qd[T,D]?, tau[T,D]?, names[D]          (robot)
      hand_pose.npz                 # t, global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3],
                                    #   confidence[T]  — MANO axis-angle labels (vision / mocap)
      object_pose.npz               # t, pos[T,3], quat[T,4] wxyz                        (optional)
      camera_<name>/timestamps.npy  # [F] s, + frames.npy uint8[F,H,W,3] or 000000.jpg …
      events.jsonl                  # {"t", "type": phase_start|phase_end|marker|success|instruction,
                                    #  "name", "value"}

The full contract is ``docs/DATA_FORMAT.md``. ``segments`` marks labelled time spans, most
importantly ``no_contact`` spans that the baseline predictor trains on.

Schema v2 adds ``dataset`` (``motion`` = D1 free-motion / self-touch, ``task`` = D2 object
tasks), ``subject``, ``task`` (task_id / instruction / object / success) and ``calibration``.
v1 manifests load with defaults.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2
MANIFEST_NAME = "session.json"
SESSION_KINDS = ("glove", "robot", "bench")
DATASETS = ("motion", "task", "other")


@dataclass
class StreamInfo:
    file: str                      # relative to the session dir
    rate_hz: float | None = None   # nominal; timestamps inside the file are authoritative
    fields: list[str] = field(default_factory=list)
    clock: str = "host"            # which clock the timestamps are in
    method: str = "linear"         # common.timeline resampling: linear | zoh


@dataclass
class SessionManifest:
    kind: str
    layout: str
    streams: dict[str, StreamInfo] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    master_hz: float = 200.0
    segments: list[dict] = field(default_factory=list)  # {t0, t1, label}
    baseline: list[float] | None = None                   # per-channel raw baseline if known
    notes: str = ""
    meta: dict = field(default_factory=dict)
    dataset: str = "other"                                # motion | task | other
    subject: str = ""
    task: dict | None = None                              # {task_id, instruction, object, success}
    calibration: dict = field(default_factory=dict)       # e.g. imu offsets, channel map
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.kind not in SESSION_KINDS:
            raise ValueError(f"kind must be one of {SESSION_KINDS}, got {self.kind!r}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        for name, s in self.streams.items():
            if Path(s.file).is_absolute():
                raise ValueError(f"stream {name!r}: file must be relative to the session dir")
            if s.method not in ("linear", "zoh"):
                raise ValueError(f"stream {name!r}: method must be linear|zoh")
        for seg in self.segments:
            if not {"t0", "t1", "label"} <= set(seg) or seg["t1"] < seg["t0"]:
                raise ValueError(f"bad segment {seg}")

    # ── segments ───────────────────────────────────────────────────────────
    def add_segment(self, t0: float, t1: float, label: str) -> None:
        self.segments.append({"t0": float(t0), "t1": float(t1), "label": str(label)})
        self.validate()

    def spans(self, label: str) -> list[tuple[float, float]]:
        return [(s["t0"], s["t1"]) for s in self.segments if s["label"] == label]

    # ── io ────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionManifest":
        d = dict(d)
        ver = d.get("schema_version", SCHEMA_VERSION)
        if ver > SCHEMA_VERSION:
            raise ValueError(f"manifest schema {ver} is newer than supported {SCHEMA_VERSION}")
        d["streams"] = {k: StreamInfo(**v) for k, v in d.get("streams", {}).items()}
        d["schema_version"] = SCHEMA_VERSION  # older manifests are upgraded with field defaults
        return cls(**d)

    def save(self, session_dir: str | Path) -> Path:
        p = Path(session_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / MANIFEST_NAME
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SessionManifest":
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_NAME
        return cls.from_dict(json.loads(p.read_text()))

    def stream_path(self, session_dir: str | Path, name: str) -> Path:
        return Path(session_dir) / self.streams[name].file

    @property
    def cameras(self) -> list[str]:
        """Camera names = streams called ``camera_<name>``."""
        return sorted(k[len("camera_"):] for k in self.streams if k.startswith("camera_"))
