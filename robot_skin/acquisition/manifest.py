"""SessionManifest — one JSON per recorded session describing its streams.

Layout on disk (``robot_skin/data/`` is git-ignored)::

    <session_dir>/
      session.json            # this manifest
      pressure.bin | .npz     # stream files (paths relative to session_dir)
      imu.npz, camera/…, joint_state.npz

``segments`` marks labelled time spans, most importantly ``no_contact`` spans that the
baseline predictor trains on.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_NAME = "session.json"
SESSION_KINDS = ("glove", "robot", "bench")


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
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.kind not in SESSION_KINDS:
            raise ValueError(f"kind must be one of {SESSION_KINDS}, got {self.kind!r}")
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
