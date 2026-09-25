"""Stream sources for the :class:`~robot_skin.acquisition.recorder.Recorder`.

A source is anything with ``name``, ``kind``, ``rate_hz``, ``start(clock)``, ``poll()``,
``stop()`` and ``info()``. ``poll()`` is **non-blocking** and returns the samples that arrived
since the last call as ``[(t_host, sample_dict), ...]`` where ``t_host`` is on the host
monotonic clock handed to ``start`` (the recorder subtracts the session ``t0``). Sample dicts per
``kind`` (the recorder turns them into the raw files of ``acquisition.manifest``):

=============  =====================================================  ==========================
kind           sample keys                                            file
=============  =====================================================  ==========================
pressure       ``raw[C]`` (channel order)                             ``pressure.npz``
imu            ``quat[S,4]`` wxyz, ``gyro[S,3]`` rad/s, ``acc[S,3]``   ``imu.npz`` (+ ``sites``)
joint_state    ``q[D]`` (+ ``qd``, ``tau``)                            ``joint_state.npz`` (+names)
hand_pose      ``global_orient[3]``, ``finger_pose[15,3]``,            ``hand_pose.npz``
               ``wrist_pos[3]``, ``confidence``
object_pose    ``pos[3]``, ``quat[4]``                                 ``object_pose.npz``
camera         ``frame`` uint8 ``[H,W,3]`` RGB                         ``camera_<name>/``
=============  =====================================================  ==========================

Implementations here:

- :class:`PlaybackSource` — replays precomputed arrays against the clock (deterministic; used
  by the fakes and for replaying recorded sessions through the recorder).
- ``Fake*Source`` — playback of a :class:`~robot_skin.acquisition.fake.FakeScene` (synthetic,
  seeded, physically consistent across streams; ``--fake`` and tests).
- :class:`SerialPressureSource` — serial taxel board (pyserial, optional). The frame format is
  injectable: :class:`CsvLineParser` (ASCII lines) is built in; for the mk555 binary protocol
  inject a parser built on ``deformable_sats/sats/preprocessing/bin_merge.py`` (canonical —
  robot_skin must not copy or import it).
- :class:`CameraSource` — OpenCV capture (cv2, optional; background grab thread).
- :class:`ImuSource`, :class:`RosJointStateSource` — documented stubs until the devices exist.

Clocks: :class:`MonotonicClock` (``time.monotonic``) for real runs, :class:`SimClock` (manually
advanced) for deterministic synchronous runs (``Recorder.step``).
"""
from __future__ import annotations

import collections
import re
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "CameraSource", "Clock", "CsvLineParser", "FakeCameraSource", "FakeHandPoseSource", "FakeImuSource",
    "FakeJointSource", "FakeObjectPoseSource", "FakePressureSource", "FrameParser", "ImuSource",
    "MonotonicClock", "PlaybackSource", "RosJointStateSource", "SAMPLE_KEYS", "STREAM_KINDS", "SerialPressureSource",
    "SimClock", "StreamSource", "fake_sources", "validate_source",
]

Clock = Callable[[], float]
Sample = tuple[float, dict]

STREAM_KINDS = ("pressure", "imu", "joint_state", "hand_pose", "object_pose", "camera")
#: required keys of a sample dict per kind (optional extras are kept if present)
SAMPLE_KEYS: dict[str, tuple[str, ...]] = {
    "pressure": ("raw",), "imu": ("quat", "gyro", "acc"), "joint_state": ("q",),
    "hand_pose": ("global_orient", "finger_pose", "wrist_pos"), "object_pose": ("pos", "quat"),
    "camera": ("frame",),
}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


# ── clocks ───────────────────────────────────────────────────────────────────
class MonotonicClock:
    """Host monotonic clock in seconds (never jumps with NTP / wall-clock changes)."""

    def __call__(self) -> float:
        return time.monotonic()


class SimClock:
    """Manually advanced clock for deterministic, faster-than-real-time recording."""

    def __init__(self, t0: float = 1000.0):
        self._t = float(t0)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("SimClock cannot go backwards")
        with self._lock:
            self._t += float(dt)
            return self._t

    def set(self, t: float) -> None:
        with self._lock:
            if t < self._t:
                raise ValueError("SimClock cannot go backwards")
            self._t = float(t)


# ── protocol ─────────────────────────────────────────────────────────────────
@runtime_checkable
class StreamSource(Protocol):
    name: str
    kind: str
    rate_hz: float | None

    def start(self, clock: Clock | None = None) -> None: ...
    def poll(self) -> list[Sample]: ...
    def stop(self) -> None: ...
    def info(self) -> dict: ...


def validate_source(src: Any) -> None:
    """Check the attributes the recorder relies on (clear error instead of a late KeyError)."""
    for attr in ("name", "kind", "start", "poll", "stop"):
        if not hasattr(src, attr):
            raise TypeError(f"{src!r} is not a StreamSource (missing {attr!r})")
    if src.kind not in STREAM_KINDS:
        raise ValueError(f"source {src.name!r}: kind {src.kind!r} not in {STREAM_KINDS}")
    if not _NAME.match(str(src.name)):
        raise ValueError(f"source name {src.name!r} must match {_NAME.pattern}")
    if src.kind == "camera" and not str(src.name).startswith("camera_"):
        raise ValueError(f"camera source names must be 'camera_<name>', got {src.name!r}")
    if src.kind != "camera" and str(src.name).startswith("camera_"):
        raise ValueError(f"'camera_' names are reserved for camera sources, got {src.name!r}")


class BaseSource:
    """Common state: name/kind/rate, clock handling, start/stop flags."""

    kind = ""

    def __init__(self, name: str, *, rate_hz: float | None = None):
        self.name = str(name)
        self.rate_hz = None if rate_hz is None else float(rate_hz)
        self._clock: Clock = MonotonicClock()
        self._running = False

    def start(self, clock: Clock | None = None) -> None:
        self._clock = clock or MonotonicClock()
        self._running = True

    def poll(self) -> list[Sample]:  # pragma: no cover - abstract
        raise NotImplementedError

    def stop(self) -> None:
        self._running = False

    def info(self) -> dict:
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind!r}, rate_hz={self.rate_hz})"


class PlaybackSource(BaseSource):
    """Replays ``data[key][k]`` at ``t[k]`` seconds after :meth:`start` (host clock).

    ``t`` must be non-decreasing. Samples are released once the clock passes their stamp, so a
    :class:`SimClock` gives exactly reproducible recordings.
    """

    def __init__(self, name: str, kind: str, t: np.ndarray, data: Mapping[str, np.ndarray], *,
                 rate_hz: float | None = None, info: Mapping | None = None):
        super().__init__(name, rate_hz=rate_hz)
        self.kind = kind
        self._t = np.asarray(t, dtype=np.float64).reshape(-1)
        if np.any(np.diff(self._t) < 0):
            raise ValueError(f"{name}: playback timestamps must be non-decreasing")
        self._data = {k: np.asarray(v) for k, v in data.items()}
        for k, v in self._data.items():
            if v.shape[0] != self._t.shape[0]:
                raise ValueError(f"{name}: data[{k!r}] has {v.shape[0]} rows for {self._t.shape[0]} stamps")
        missing = [k for k in SAMPLE_KEYS.get(kind, ()) if k not in self._data]
        if missing:
            raise ValueError(f"{name}: {kind} samples need keys {missing}")
        self._info = dict(info or {})
        self._i = 0
        self._start = 0.0

    def start(self, clock: Clock | None = None) -> None:
        super().start(clock)
        self._start = self._clock()
        self._i = 0

    def poll(self) -> list[Sample]:
        if not self._running:
            return []
        now = self._clock() - self._start
        j = int(np.searchsorted(self._t, now, side="right"))
        out = [(self._start + float(self._t[k]), {key: v[k] for key, v in self._data.items()})
               for k in range(self._i, j)]
        self._i = j
        return out

    @property
    def exhausted(self) -> bool:
        return self._i >= self._t.shape[0]

    def info(self) -> dict:
        return dict(self._info)


# ── fakes (FakeScene playback) ───────────────────────────────────────────────
def _scene(scene, kw: dict):
    if scene is None:
        from .fake import FakeScene
        scene = FakeScene.default(**kw)
    elif kw:
        raise TypeError(f"scene given — unexpected FakeScene kwargs {sorted(kw)}")
    return scene


class FakePressureSource(PlaybackSource):
    """Synthetic taxel pressure ``raw[C]`` (baseline ~1e6 counts, motion artefact, contacts)."""

    def __init__(self, scene=None, *, name: str = "pressure", **scene_kw):
        self.scene = _scene(scene, scene_kw)
        s = self.scene.pressure_stream()
        super().__init__(name, "pressure", s["stamp"], {"raw": s["raw"]}, rate_hz=self.scene.cfg.rate("pressure"),
                         info={"n_channels": int(s["raw"].shape[1]), "units": "raw counts", "fake": True})


class FakeImuSource(PlaybackSource):
    """Synthetic glove IMUs (7 sites; mounting offsets + IMU-world rotation applied)."""

    def __init__(self, scene=None, *, name: str = "imu", **scene_kw):
        self.scene = _scene(scene, scene_kw)
        s = self.scene.imu_stream()
        super().__init__(name, "imu", s["stamp"], {k: s[k] for k in ("quat", "gyro", "acc")},
                         rate_hz=self.scene.cfg.rate("imu"), info={"sites": [str(x) for x in s["sites"]], "fake": True})


class FakeCameraSource(PlaybackSource):
    """Synthetic RGB camera ``camera_<camera>`` (blobs follow wrist / fingers / object)."""

    def __init__(self, scene=None, camera: str = "ego", *, name: str | None = None, **scene_kw):
        self.scene = _scene(scene, scene_kw)
        s = self.scene.camera_stream(camera)
        super().__init__(name or f"camera_{camera}", "camera", s["stamp"], {"frame": s["frame"]},
                         rate_hz=self.scene.cfg.rate(f"camera_{camera}"),
                         info={"camera": camera, "shape": list(s["frame"].shape[1:]), "fake": True})


class FakeHandPoseSource(PlaybackSource):
    """Synthetic MANO labels (what offline HaMeR/WiLoR would write to ``hand_pose.npz``)."""

    def __init__(self, scene=None, *, name: str = "hand_pose", **scene_kw):
        self.scene = _scene(scene, scene_kw)
        s = self.scene.hand_pose_stream()
        keys = ("global_orient", "finger_pose", "wrist_pos", "confidence")
        super().__init__(name, "hand_pose", s["stamp"], {k: s[k] for k in keys},
                         rate_hz=self.scene.cfg.rate("hand_pose"), info={"fake": True, "source": "fake_vision"})


class FakeObjectPoseSource(PlaybackSource):
    """Synthetic object pose (D2): rests, follows the hand while grasped, stays where released."""

    def __init__(self, scene=None, *, name: str = "object_pose", **scene_kw):
        self.scene = _scene(scene, scene_kw)
        s = self.scene.object_pose_stream()
        super().__init__(name, "object_pose", s["stamp"], {"pos": s["pos"], "quat": s["quat"]},
                         rate_hz=self.scene.cfg.rate("object_pose"), info={"fake": True})


class FakeJointSource(PlaybackSource):
    """Synthetic robot joint state (15 finger joints) for the ``robot`` kind."""

    def __init__(self, scene=None, *, name: str = "joint_state", **scene_kw):
        if scene is None:
            scene_kw.setdefault("kind", "robot")
            scene_kw.setdefault("layout", "robot_hand_template")
        self.scene = _scene(scene, scene_kw)
        s = self.scene.joint_stream()
        super().__init__(name, "joint_state", s["stamp"], {"q": s["q"], "qd": s["qd"], "tau": s["tau"]},
                         rate_hz=self.scene.cfg.rate("joint_state"),
                         info={"names": [str(x) for x in s["names"]], "fake": True})


def fake_sources(scene, *, hand_pose: bool = True, object_pose: bool = True) -> list[PlaybackSource]:
    """All sources of a :class:`~robot_skin.acquisition.fake.FakeScene` (by its ``stream_names``)."""
    out: list[PlaybackSource] = []
    for name in scene.stream_names:
        if name == "pressure":
            out.append(FakePressureSource(scene))
        elif name == "imu":
            out.append(FakeImuSource(scene))
        elif name.startswith("camera_"):
            out.append(FakeCameraSource(scene, name[len("camera_"):]))
        elif name == "hand_pose" and hand_pose:
            out.append(FakeHandPoseSource(scene))
        elif name == "object_pose" and object_pose:
            out.append(FakeObjectPoseSource(scene))
        elif name == "joint_state":
            out.append(FakeJointSource(scene))
    return out


# ── serial taxel board ───────────────────────────────────────────────────────
@runtime_checkable
class FrameParser(Protocol):
    """Stateful byte-stream → frames parser: ``feed(chunk) -> [(device_t_s | None, raw[C]), ...]``."""

    def feed(self, data: bytes) -> list[tuple[float | None, np.ndarray]]: ...


class CsvLineParser:
    """ASCII frames, one per line: ``v0,v1,...,v{C-1}`` or ``t_us,v0,...`` (``time_column=True``,
    device microseconds). Malformed lines are counted in :attr:`n_bad` and skipped."""

    def __init__(self, n_channels: int, *, sep: str = ",", time_column: bool = False, time_scale: float = 1e-6):
        if n_channels < 1:
            raise ValueError("n_channels must be ≥ 1")
        self.n_channels = int(n_channels)
        self.sep = sep
        self.time_column = bool(time_column)
        self.time_scale = float(time_scale)
        self._buf = b""
        self.n_bad = 0

    def feed(self, data: bytes) -> list[tuple[float | None, np.ndarray]]:
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        out = []
        want = self.n_channels + (1 if self.time_column else 0)
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                vals = [float(x) for x in ln.decode("ascii").split(self.sep)]
            except (UnicodeDecodeError, ValueError):
                self.n_bad += 1
                continue
            if len(vals) != want:
                self.n_bad += 1
                continue
            if self.time_column:
                out.append((vals[0] * self.time_scale, np.asarray(vals[1:], dtype=np.float64)))
            else:
                out.append((None, np.asarray(vals, dtype=np.float64)))
        return out


class SerialPressureSource(BaseSource):
    """Taxel board over a serial port (pyserial; ``pip install pyserial``).

    Timestamps (always strictly increasing):

    - parser yields device times → host stamps are ``t_dev + (host − dev)`` with the offset fixed
      at the first frame (device-regular; residual drift is removed by the 3-tap sync). A device
      clock that jumps back (reset / counter wrap) or runs more than
      :attr:`REANCHOR_AHEAD_S` ahead of the arrival time is re-anchored (counted in ``info()``).
    - no device times → the frames of one read are back-filled from the arrival time at
      ``1/rate_hz`` spacing so a burst read does not collapse onto one stamp; the spacing shrinks
      when the burst would otherwise reach back before the previous stamp (USB serial delivers
      in bursts), so stamps stay in ``(previous stamp, arrival time]``.

    Frames whose channel count differs from ``n_channels`` are dropped (``n_bad_frames``).

    mk555 binary boards: pass ``parser=`` wrapping the canonical parser in
    ``deformable_sats/sats/preprocessing/bin_merge.py`` (do not copy it into robot_skin).
    """

    REANCHOR_AHEAD_S = 0.5

    kind = "pressure"

    def __init__(self, port: str, n_channels: int, *, baudrate: int = 115200, parser: FrameParser | None = None,
                 name: str = "pressure", rate_hz: float = 200.0, read_timeout_s: float = 0.0):
        super().__init__(name, rate_hz=rate_hz)
        self.port = str(port)
        self.n_channels = int(n_channels)
        self.baudrate = int(baudrate)
        self.parser = parser or CsvLineParser(n_channels)
        self.read_timeout_s = float(read_timeout_s)
        self._ser = None
        self._dev_offset: float | None = None
        self._last_t: float | None = None
        self.n_frames = 0
        self.n_bad_frames = 0
        self.n_reanchors = 0

    def start(self, clock: Clock | None = None) -> None:
        try:
            import serial  # type: ignore
        except ImportError as e:
            raise ImportError("SerialPressureSource needs pyserial: pip install pyserial") from e
        super().start(clock)
        self._dev_offset, self._last_t = None, None
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout_s)
        self._ser.reset_input_buffer()

    _MIN_STEP_S = 1e-6      # stamp increment when no room is left (clock did not advance)

    def poll(self) -> list[Sample]:
        if self._ser is None:
            return []
        n = int(getattr(self._ser, "in_waiting", 0) or 0)
        if n <= 0:
            return []
        chunk = self._ser.read(n)
        t_arr = self._clock()
        good = []
        for t_dev, raw in self.parser.feed(chunk):
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            if raw.shape[0] != self.n_channels:
                self.n_bad_frames += 1
                continue
            good.append((t_dev, raw))
        nf = len(good)
        dt = 1.0 / self.rate_hz if self.rate_hz else 0.0
        if self._last_t is not None and nf and dt > 0:
            room = t_arr - self._last_t               # back-filled stamps must stay after the last one
            dt = min(dt, room / nf) if room > 0 else 0.0
        out: list[Sample] = []
        for i, (t_dev, raw) in enumerate(good):
            if t_dev is not None:
                if self._dev_offset is not None:
                    t = t_dev + self._dev_offset
                    if (self._last_t is not None and t <= self._last_t) or t > t_arr + self.REANCHOR_AHEAD_S:
                        self._dev_offset = None       # device clock reset / wrapped / jumped
                        self.n_reanchors += 1
                if self._dev_offset is None:
                    self._dev_offset = t_arr - t_dev
                t = t_dev + self._dev_offset
            else:
                t = t_arr - (nf - 1 - i) * dt
            if self._last_t is not None and t <= self._last_t:
                t = self._last_t + self._MIN_STEP_S
            self._last_t = t
            out.append((t, {"raw": raw}))
        self.n_frames += len(out)
        return out

    def stop(self) -> None:
        super().stop()
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def info(self) -> dict:
        return {"port": self.port, "baudrate": self.baudrate, "n_channels": self.n_channels,
                "parser": type(self.parser).__name__, "n_bad": int(getattr(self.parser, "n_bad", 0)),
                "n_bad_frames": self.n_bad_frames, "n_reanchors": self.n_reanchors}


# ── camera (OpenCV) ──────────────────────────────────────────────────────────
class CameraSource(BaseSource):
    """OpenCV camera ``camera_<camera>`` (``pip install opencv-python``); frames are grabbed on a
    background thread, stamped on arrival (host clock) and returned RGB by :meth:`poll`.

    Arrival stamps include the exposure→USB latency (tens of ms); the 3-tap sync measures and
    removes it. Not exercised in CI (no cv2 / camera there).
    """

    kind = "camera"

    def __init__(self, camera: str, device: int | str = 0, *, rate_hz: float = 30.0,
                 size: tuple[int, int] | None = None, max_queue: int = 256):
        super().__init__(f"camera_{camera}", rate_hz=rate_hz)
        try:
            import cv2  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError("CameraSource needs OpenCV: pip install opencv-python") from e
        self.camera = str(camera)
        self.device = device
        self.size = size
        self._q: collections.deque = collections.deque(maxlen=max_queue)
        self._thread: threading.Thread | None = None
        self._cap = None
        self.n_dropped = 0

    def start(self, clock: Clock | None = None) -> None:
        import cv2  # type: ignore
        super().start(clock)
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device!r}")
        if self.rate_hz:
            self._cap.set(cv2.CAP_PROP_FPS, float(self.rate_hz))
        if self.size:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.size[0]))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.size[1]))
        self._thread = threading.Thread(target=self._grab, name=f"grab-{self.name}", daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            t = self._clock()
            if not ok:
                time.sleep(0.005)
                continue
            if len(self._q) == self._q.maxlen:
                self.n_dropped += 1
            self._q.append((t, {"frame": np.ascontiguousarray(frame[..., ::-1])}))

    def poll(self) -> list[Sample]:
        out = []
        while self._q:
            out.append(self._q.popleft())
        return out

    def stop(self) -> None:
        super().stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def info(self) -> dict:
        return {"camera": self.camera, "device": str(self.device), "n_dropped_queue": self.n_dropped}


# ── device stubs ─────────────────────────────────────────────────────────────
class ImuSource(BaseSource):
    """Glove IMU board (7 sites) — **not implemented until the device exists**.

    TODO (device bring-up): open the IMU hub (serial / BLE / vendor SDK), and in :meth:`poll`
    return ``(t_host, {"quat": [S,4] wxyz, "gyro": [S,3] rad/s, "acc": [S,3] m/s²})`` in each
    sensor's own frame (specific force incl. gravity), site order = ``sites`` (layout
    ``imu_sites``: wrist, palm, thumb, index, middle, ring, pinky). Prefer device timestamps
    mapped to host time like :class:`SerialPressureSource`. Use :class:`FakeImuSource` meanwhile.
    """

    kind = "imu"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ImuSource: glove IMU device IO is not implemented yet — implement ImuSource.poll for your IMU hub "
            "(see its docstring) or record with --fake / --dry-run (or --no-imu for pressure + cameras only)")


class RosJointStateSource(BaseSource):
    """ROS 2 ``sensor_msgs/JointState`` subscriber — **stub until a robot hand is connected**.

    TODO: rclpy node on ``topic`` (default ``/joint_states``) with a callback that appends
    ``(clock(), {"q": position, "qd": velocity, "tau": effort})`` to a deque (message header stamps
    converted to the host monotonic clock), ``names`` from the first message; :meth:`poll` drains
    the deque. Use :class:`FakeJointSource` meanwhile.
    """

    kind = "joint_state"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RosJointStateSource: robot joint-state IO is not implemented yet (rclpy subscriber, see docstring) — "
            "use --fake or --dry-run")


def _check_sources(sources: Sequence[Any]) -> None:
    names = set()
    for s in sources:
        validate_source(s)
        if s.name in names:
            raise ValueError(f"duplicate source name {s.name!r}")
        names.add(s.name)
