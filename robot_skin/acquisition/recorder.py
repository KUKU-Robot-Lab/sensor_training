"""Recorder: poll stream sources on one host clock, log events, write a raw session directory.

::

    rec = Recorder(sources, "robot_skin/data/raw/motion/S01/<session_id>", manifest)
    rec.start()                                   # t0 = host monotonic clock; sources started
    with rec.phase("baseline_start", contact="none", labels=["no_contact"]):
        rec.run_for(5.0)                          # real: sleeps while source threads poll
    rec.marker("tap")
    manifest = rec.stop()                         # writes the raw files + events + session.json

Timestamps: every sample is stamped by its source on the **host monotonic clock**; the recorder
stores ``t = t_host − t0`` (seconds since :meth:`start`, float64). Remaining per-device latency
and drift are measured afterwards by the 3-tap sync (:mod:`robot_skin.acquisition.sync`).

Output (formats of :mod:`robot_skin.acquisition.manifest`, read by ``datasets.build``)::

    pressure.npz      t[T], raw[T,C]                      (channel order)
    imu.npz           t, quat[T,S,4] wxyz, gyro[T,S,3], acc[T,S,3], sites[S]
    joint_state.npz   t, q[T,D] (+ qd, tau), names[D]
    hand_pose.npz     t, global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3], confidence[T]
    object_pose.npz   t, pos[T,3], quat[T,4]
    camera_<name>/    timestamps.npy[F] + frames.npy uint8[F,H,W,3]  (or 000000.jpg …)
    events.jsonl      {"t", "type", "name", "value"} — appended + flushed live (crash-safe)
    session.json      SessionManifest: streams, segments (from phase events), meta, calibration

Events: ``phase_start``/``phase_end`` (name = step id; the start ``value`` carries ``contact`` and
``labels``), ``marker``, ``instruction``, ``success``. At :meth:`stop` every phase becomes one
manifest segment per label (``labels`` or the contact default: ``none`` → ``no_contact``,
``self`` → ``self_touch``); phases still open are closed at the stop time (``auto_closed``).

Two drive modes: **threaded** (real devices; one polling thread per source, :meth:`run_for`
sleeps) and **synchronous** (:class:`~robot_skin.acquisition.sources.SimClock`; :meth:`step` polls
once, :meth:`run_until` advances the simulated clock — deterministic and faster than real time).
The layout of a session (per-stream files + manifest + event log) follows ActionSense
(DelPreto et al., NeurIPS 2022 Datasets & Benchmarks).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import platform
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .manifest import SessionManifest, StreamInfo
from .protocol import DEFAULT_CONTACT_LABELS
from .sources import Clock, MonotonicClock, SimClock, _check_sources

__all__ = [
    "EVENTS_NAME", "EVENT_TYPES", "EventLog", "Recorder", "STREAM_FILES", "load_events", "phases_from_events",
    "segments_from_events", "stream_file", "TIME_EPS",
]

log = logging.getLogger(__name__)

EVENTS_NAME = "events.jsonl"
EVENT_TYPES = ("phase_start", "phase_end", "marker", "success", "instruction")
#: tolerance (s) under which :meth:`Recorder.run_until` treats a target time as reached
TIME_EPS = 1e-9
#: canonical file per stream kind (camera: directory ``<stream name>``)
STREAM_FILES = {"pressure": "pressure.npz", "imu": "imu.npz", "joint_state": "joint_state.npz",
                "hand_pose": "hand_pose.npz", "object_pose": "object_pose.npz"}
_METHOD = {"camera": "zoh"}


def stream_file(name: str, kind: str) -> str:
    """Relative file for a stream: canonical names for the canonical stream names, ``<name>.npz``
    for extra streams of a kind (e.g. a second pressure board), ``<name>/`` for cameras."""
    if kind == "camera":
        return name
    if STREAM_FILES.get(kind) == f"{name}.npz":
        return STREAM_FILES[kind]
    return f"{name}.npz"


# ── events ───────────────────────────────────────────────────────────────────
def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class EventLog:
    """Append-only event list, mirrored line-by-line to ``events.jsonl`` (flushed per event)."""

    def __init__(self, path: str | Path | None = None):
        self.events: list[dict] = []
        self.path = None if path is None else Path(path)
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, t: float, type: str, name: str = "", value: Any = None) -> dict:
        if type not in EVENT_TYPES:
            raise ValueError(f"event type {type!r} not in {EVENT_TYPES}")
        if not math.isfinite(float(t)):
            raise ValueError("event time must be finite")
        ev = {"t": float(t), "type": type, "name": str(name), "value": _jsonable(value)}
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._fh.flush()
        return ev

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load_events(path: str | Path) -> list[dict]:
    """Read ``events.jsonl`` (a session dir is accepted); missing file → ``[]``. Sorted by time
    (stable), so manual edits appended out of order are fine."""
    p = Path(path)
    if p.is_dir():
        p = p / EVENTS_NAME
    if not p.is_file():
        return []
    out = []
    for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not ln.strip():
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i + 1}: bad JSON ({e})") from None
        if not {"t", "type"} <= set(ev):
            raise ValueError(f"{p}:{i + 1}: event needs 't' and 'type'")
        ev.setdefault("name", "")
        ev.setdefault("value", None)
        out.append(ev)
    return sorted(out, key=lambda e: float(e["t"]))


def phases_from_events(events: Sequence[Mapping], t_end: float | None = None) -> list[dict]:
    """Pair ``phase_start``/``phase_end`` by name (innermost first) → ``[{name, t0, t1, value,
    closed}]`` sorted by start. Unclosed phases end at ``t_end`` (or the last event time)."""
    open_: dict[str, list[tuple[float, Any]]] = {}
    out: list[dict] = []
    last = max((float(e["t"]) for e in events), default=0.0)
    for e in events:
        if e["type"] == "phase_start":
            open_.setdefault(e["name"], []).append((float(e["t"]), e.get("value")))
        elif e["type"] == "phase_end":
            stack = open_.get(e["name"])
            if not stack:
                log.warning("phase_end %r without a matching phase_start (ignored)", e["name"])
                continue
            t0, v = stack.pop()
            out.append({"name": e["name"], "t0": t0, "t1": float(e["t"]), "value": v, "closed": True})
    end = last if t_end is None else float(t_end)
    for name, stack in open_.items():
        for t0, v in stack:
            out.append({"name": name, "t0": t0, "t1": max(end, t0), "value": v, "closed": False})
    return sorted(out, key=lambda p: (p["t0"], p["t1"]))


def _labels_of(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    if value.get("labels") is not None:
        return tuple(str(x) for x in value["labels"])
    return DEFAULT_CONTACT_LABELS.get(str(value.get("contact")), ())


def segments_from_events(events: Sequence[Mapping], t_end: float | None = None) -> list[dict]:
    """Manifest segments ``{t0, t1, label}`` from phase events (one per phase and label)."""
    segs = []
    for p in phases_from_events(events, t_end):
        for lab in _labels_of(p["value"]):
            segs.append({"t0": p["t0"], "t1": p["t1"], "label": lab})
    return segs


# ── stream buffers ───────────────────────────────────────────────────────────
class _Buffer:
    """Accumulates samples of one non-camera stream; writes its npz at the end."""

    def __init__(self, name: str, kind: str):
        self.name, self.kind = name, kind
        self.t: list[float] = []
        self.cols: dict[str, list] = {}

    def add(self, t: float, sample: Mapping[str, Any]) -> None:
        if not self.cols:
            self.cols = {k: [] for k in sample}          # one assignment: snapshot() may iterate concurrently
        elif set(sample) != set(self.cols):
            raise ValueError(f"{self.name}: sample keys changed from {sorted(self.cols)} to {sorted(sample)}")
        for k, v in sample.items():                      # columns first, then t: snapshot() reads len(t)
            self.cols[k].append(np.asarray(v))
        self.t.append(float(t))

    def __len__(self) -> int:
        return len(self.t)

    def arrays(self) -> dict[str, np.ndarray]:
        t = np.asarray(self.t, dtype=np.float64)
        out = {"t": t}
        for k, v in self.cols.items():
            try:
                out[k] = np.stack(v)
            except ValueError as e:
                raise ValueError(f"{self.name}.{k}: inconsistent sample shapes ({e})") from None
        return out

    def write(self, session_dir: Path, info: Mapping) -> tuple[str, list[str]]:
        a = self.arrays()
        rel = stream_file(self.name, self.kind)
        path = session_dir / rel
        if self.kind == "hand_pose":
            from robot_skin.pose.vision_hand import save_hand_labels
            save_hand_labels(path, a["t"], a["global_orient"], a["finger_pose"], a["wrist_pos"], a.get("confidence"))
            return rel, ["t", "global_orient[3]", "finger_pose[15,3]", "wrist_pos[3]", "confidence"]
        if self.kind == "pressure":
            a["raw"] = a["raw"].astype(np.float64)
            fields = ["raw[C]"]
        elif self.kind == "imu":
            S = a["quat"].shape[1]
            sites = list(info.get("sites") or [f"site{i}" for i in range(S)])
            if len(sites) != S:
                raise ValueError(f"{self.name}: {len(sites)} site names for {S} IMU sites")
            a["sites"] = np.array(sites)
            fields = [f"{s}.{k}" for s in sites for k in ("quat", "gyro", "acc")]
        elif self.kind == "joint_state":
            D = a["q"].shape[1]
            names = list(info.get("names") or [f"j{i}" for i in range(D)])
            if len(names) != D:
                raise ValueError(f"{self.name}: {len(names)} joint names for D={D}")
            a["names"] = np.array(names)
            fields = [f"{k}[D]" for k in ("q", "qd", "tau") if k in a]
        elif self.kind == "object_pose":
            fields = ["pos[3]", "quat[4]"]
        else:  # pragma: no cover - validated earlier
            fields = sorted(k for k in a if k != "t")
        np.savez(path, **a)
        return rel, fields


class _CameraBuffer:
    """Camera frames: ``npy`` (kept in memory, one ``frames.npy``) or ``jpg`` (written as they
    arrive — for real, long sessions). ``auto`` picks npy for tiny frames (≤ 128×128)."""

    def __init__(self, name: str, session_dir: Path, fmt: str = "auto", jpeg_quality: int = 90):
        if fmt not in ("auto", "npy", "jpg"):
            raise ValueError(f"camera_format must be auto|npy|jpg, got {fmt!r}")
        self.name, self.kind = name, "camera"
        self.dir = session_dir / name
        self.fmt = fmt
        self.quality = int(jpeg_quality)
        self.t: list[float] = []
        self.frames: list[np.ndarray] = []
        self.shape: tuple | None = None

    def add(self, t: float, sample: Mapping[str, Any]) -> None:
        f = np.asarray(sample["frame"])
        if f.ndim != 3 or f.shape[-1] != 3 or f.dtype != np.uint8:
            raise ValueError(f"{self.name}: frames must be uint8 [H,W,3], got {f.dtype} {f.shape}")
        if self.shape is None:
            self.shape = f.shape
            if self.fmt == "auto":
                self.fmt = "npy" if f.shape[0] * f.shape[1] <= 128 * 128 else "jpg"
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.fmt == "jpg":
                try:
                    from PIL import Image  # noqa: F401
                except ImportError as e:
                    raise ImportError("camera_format='jpg' needs Pillow: pip install pillow") from e
        elif f.shape != self.shape:
            raise ValueError(f"{self.name}: frame shape changed {self.shape} → {f.shape}")
        if self.fmt == "jpg":
            from PIL import Image
            Image.fromarray(f).save(self.dir / f"{len(self.t):06d}.jpg", quality=self.quality)
        else:
            self.frames.append(f)
        self.t.append(float(t))

    def __len__(self) -> int:
        return len(self.t)

    def write(self, session_dir: Path, info: Mapping) -> tuple[str, list[str]]:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "timestamps.npy", np.asarray(self.t, dtype=np.float64))
        if self.fmt == "npy":
            np.save(self.dir / "frames.npy", np.stack(self.frames) if self.frames
                    else np.zeros((0, 0, 0, 3), dtype=np.uint8))
        return self.name, ["frame"]


# ── recorder ─────────────────────────────────────────────────────────────────
class Recorder:
    """Record ``sources`` into ``session_dir`` (see module docstring).

    ``clock`` — host clock (default ``time.monotonic``; a :class:`SimClock` selects synchronous
    mode); ``camera_format`` — ``auto`` | ``npy`` | ``jpg``; ``overwrite`` — allow a non-empty
    ``session_dir`` (the files this recording writes — stream files, camera directories,
    ``events.jsonl``, ``qc.json`` — are deleted first); ``sim_dt`` — step of :meth:`run_until` under a SimClock;
    ``poll_interval_s`` — idle sleep of the polling threads.
    """

    def __init__(self, sources: Sequence[Any], session_dir: str | Path, manifest: SessionManifest, *,
                 clock: Clock | None = None, camera_format: str = "auto", jpeg_quality: int = 90,
                 overwrite: bool = False, sim_dt: float = 0.01, poll_interval_s: float = 0.002):
        _check_sources(sources)
        self.sources = list(sources)
        self.session_dir = Path(session_dir)
        self.manifest = manifest
        self.clock: Clock = clock or MonotonicClock()
        self.sim = isinstance(self.clock, SimClock)
        self.camera_format = camera_format
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite
        self.sim_dt = float(sim_dt)
        self.poll_interval_s = float(poll_interval_s)
        self.events: EventLog | None = None
        self.t0: float | None = None
        self.t_stop: float | None = None
        self._buffers: dict[str, Any] = {}
        self._threads: list[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._errors: list[str] = []
        self._segments: list[dict] = []
        self._started_utc = ""
        self._threaded = False

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self.t0 is not None and self.t_stop is None

    def start(self, *, threaded: bool | None = None) -> "Recorder":
        if self.t0 is not None:
            raise RuntimeError("recorder already started")
        d = self.session_dir
        if d.exists() and any(d.iterdir()) and not self.overwrite:
            raise FileExistsError(f"{d} is not empty (pass overwrite=True to reuse it)")
        d.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            self._clear_outputs(d)
        self.events = EventLog(d / EVENTS_NAME)
        for s in self.sources:
            self._buffers[s.name] = (_CameraBuffer(s.name, d, self.camera_format, self.jpeg_quality)
                                     if s.kind == "camera" else _Buffer(s.name, s.kind))
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self.t0 = self.clock()
        started = []
        try:
            for s in self.sources:
                s.start(self.clock)
                started.append(s)
        except BaseException:
            for s in started:
                with contextlib.suppress(Exception):
                    s.stop()
            self.events.close()
            raise
        self._threaded = (not self.sim) if threaded is None else bool(threaded)
        if self._threaded:
            for s in self.sources:
                th = threading.Thread(target=self._poll_loop, args=(s,), name=f"poll-{s.name}", daemon=True)
                th.start()
                self._threads.append(th)
        return self

    def _clear_outputs(self, d: Path) -> None:
        """``overwrite=True``: delete what this recording will write, so nothing of an earlier take
        survives (leftover ``%06d.jpg`` frames, or a ``timestamps_host.npy`` that sync would read as
        the new take's original stamps). Other files (e.g. offline ``hand_pose.npz``) are kept."""
        for s in self.sources:
            p = d / stream_file(s.name, s.kind)
            if s.kind == "camera":
                if p.is_dir():
                    shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
        for name in (EVENTS_NAME, "qc.json"):      # qc.json: acquisition.qc.QC_NAME (stale report)
            (d / name).unlink(missing_ok=True)

    def _ingest(self, src, samples) -> int:
        buf = self._buffers[src.name]
        for t_host, sample in samples:
            buf.add(float(t_host) - self.t0, sample)
        return len(samples)

    def _poll_loop(self, src) -> None:
        while not self._stop_evt.is_set():
            try:
                n = self._ingest(src, src.poll())
            except Exception as e:  # keep the other streams recording
                self._errors.append(f"{src.name}: {type(e).__name__}: {e}")
                log.exception("source %s failed; its stream stops here", src.name)
                return
            if n == 0:
                self._stop_evt.wait(self.poll_interval_s)

    def step(self) -> int:
        """Poll every source once (synchronous mode). Returns the number of new samples."""
        self._require_running()
        if self._threaded:
            raise RuntimeError("step() is for synchronous mode; this recorder runs polling threads")
        return sum(self._ingest(s, s.poll()) for s in self.sources)

    def now(self) -> float:
        """Session time (s since :meth:`start`)."""
        if self.t0 is None:
            raise RuntimeError("recorder not started")
        return self.clock() - self.t0

    def run_until(self, t_session: float, *, sleep=time.sleep) -> None:
        """Keep recording until session time ``t_session`` (SimClock: advance in ``sim_dt`` steps).

        Times within :data:`TIME_EPS` of the target count as reached (float accumulation of a
        simulated clock); callers looping on ``now()`` must use the same tolerance.
        """
        self._require_running()
        while True:
            remaining = float(t_session) - self.now()
            if remaining <= TIME_EPS:
                break
            if self.sim:
                self.clock.advance(min(self.sim_dt, remaining))
                self.step()
            elif self._threaded:
                sleep(min(0.05, remaining))
            else:
                sleep(min(self.poll_interval_s, remaining))
                self.step()

    def run_for(self, seconds: float, **kw) -> None:
        self.run_until(self.now() + float(seconds), **kw)

    def _require_running(self) -> None:
        if not self.running:
            raise RuntimeError("recorder is not running (call start())")

    # ── events ───────────────────────────────────────────────────────────
    def event(self, type: str, name: str = "", value: Any = None, *, t: float | None = None) -> dict:
        self._require_running()
        return self.events.log(self.now() if t is None else float(t), type, name, value)

    def phase_start(self, name: str, value: Mapping | None = None, **kw) -> dict:
        return self.event("phase_start", name, {**(value or {}), **kw})

    def phase_end(self, name: str, value: Mapping | None = None, **kw) -> dict:
        v = {**(value or {}), **kw}
        return self.event("phase_end", name, v or None)

    @contextlib.contextmanager
    def phase(self, name: str, value: Mapping | None = None, **kw) -> Iterator[dict]:
        ev = self.phase_start(name, value, **kw)
        try:
            yield ev
        finally:
            if self.running:
                self.phase_end(name)

    def marker(self, name: str, value: Any = None) -> dict:
        return self.event("marker", name, value)

    def instruction(self, text: str) -> dict:
        return self.event("instruction", "instruction", str(text))

    def success(self, ok: bool | None) -> dict:
        """Operator verdict of the episode (``value`` = true / false / null = not judged)."""
        return self.event("success", "success", None if ok is None else bool(ok))

    def add_segment(self, t0: float, t1: float, label: str) -> None:
        """Explicit manifest segment (in addition to the ones derived from phases)."""
        if t1 < t0:
            raise ValueError("segment t1 < t0")
        self._segments.append({"t0": float(t0), "t1": float(t1), "label": str(label)})

    def open_phases(self) -> list[str]:
        return [p["name"] for p in phases_from_events(self.events.events) if not p["closed"]] if self.events else []

    # ── in-memory access (live checks, e.g. IMU calibration after its block) ─
    def snapshot(self, stream: str, t0: float | None = None, t1: float | None = None) -> dict[str, np.ndarray]:
        """Copy of the samples recorded so far for ``stream`` (optionally within [t0, t1])."""
        buf = self._buffers[stream]
        if isinstance(buf, _CameraBuffer):
            raise ValueError("snapshot() is for non-camera streams")
        n = len(buf)
        t = np.asarray(buf.t[:n], dtype=np.float64)
        m = np.ones(n, dtype=bool)
        if t0 is not None:
            m &= t >= t0
        if t1 is not None:
            m &= t <= t1
        idx = np.flatnonzero(m)
        out = {"t": t[idx]}
        for k, v in buf.cols.items():
            out[k] = np.stack([v[i] for i in idx]) if idx.size else np.zeros((0,))
        return out

    def source_info(self, stream: str) -> dict:
        for s in self.sources:
            if s.name == stream:
                return s.info() if hasattr(s, "info") else {}
        raise KeyError(stream)

    def counts(self) -> dict[str, int]:
        return {k: len(b) for k, b in self._buffers.items()}

    # ── stop & write ─────────────────────────────────────────────────────
    def stop(self) -> SessionManifest:
        """Stop sources, write every stream, events and ``session.json``; return the manifest."""
        self._require_running()
        if self._threaded:
            self._stop_evt.set()
            for th in self._threads:
                th.join(timeout=5.0)
        for s in self.sources:                           # drain what arrived meanwhile
            try:
                self._ingest(s, s.poll())
            except Exception as e:
                self._errors.append(f"{s.name}: {type(e).__name__}: {e}")
        t_end = self.now()
        for s in self.sources:
            try:
                s.stop()
            except Exception as e:
                self._errors.append(f"{s.name}.stop: {type(e).__name__}: {e}")
        for name in self.open_phases():
            self.events.log(t_end, "phase_end", name, {"auto_closed": True})
            log.warning("phase %r was still open at stop(); closed at t=%.3f", name, t_end)
        self.t_stop = t_end
        self.events.close()
        return self._finalize(t_end)

    def _finalize(self, t_end: float) -> SessionManifest:
        m, d = self.manifest, self.session_dir
        stats, empty = {}, []
        infos = {s.name: (s.info() if hasattr(s, "info") else {}) for s in self.sources}
        for s in self.sources:
            buf = self._buffers[s.name]
            if len(buf) == 0:
                empty.append(s.name)
                log.warning("stream %s recorded no samples (not written)", s.name)
                continue
            try:
                rel, fields = buf.write(d, infos[s.name])
            except Exception as e:                       # keep the other streams + manifest
                self._errors.append(f"{s.name}.write: {type(e).__name__}: {e}")
                log.exception("writing stream %s failed", s.name)
                continue
            m.streams[s.name] = StreamInfo(file=rel, rate_hz=s.rate_hz, fields=fields, clock="host",
                                           method=_METHOD.get(s.kind, "linear"))
            t = np.asarray(buf.t, dtype=np.float64)
            span = float(t[-1] - t[0]) if t.size > 1 else 0.0
            stats[s.name] = {"kind": s.kind, "n": int(t.size), "t_first": float(t[0]), "t_last": float(t[-1]),
                             "rate_hz_measured": (t.size - 1) / span if span > 0 else None}
        segs = segments_from_events(self.events.events, t_end) + self._segments
        known = {(s["t0"], s["t1"], s["label"]) for s in m.segments}
        for s in segs:
            if (s["t0"], s["t1"], s["label"]) not in known:
                m.add_segment(s["t0"], s["t1"], s["label"])
        m.meta.setdefault("recorder", {})
        m.meta["recorder"].update({
            "clock": "sim" if self.sim else "host_monotonic", "started_utc": self._started_utc,
            "duration_s": float(t_end), "events_file": EVENTS_NAME, "n_events": len(self.events.events),
            "streams": stats, "sources": {k: _jsonable(v) for k, v in infos.items()},
            "empty_streams": empty, "errors": list(self._errors), "host": platform.node() or "",
            "python": platform.python_version(),
        })
        m.save(d)
        if self._errors:
            log.error("recording finished with source errors: %s", self._errors)
        return m

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "Recorder":
        if self.t0 is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.running:
            if exc_type is not None and self.events is not None:
                with contextlib.suppress(Exception):
                    self.events.log(self.now(), "marker", "aborted", f"{exc_type.__name__}: {exc}")
            self.stop()
