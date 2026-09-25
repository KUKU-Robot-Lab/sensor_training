"""Deterministic synthetic acquisition scene: what the ``--fake`` loggers and the Fake*Sources play.

A :class:`FakeScene` turns a protocol timeline (``EpisodePlan.timeline``) into physically
consistent raw streams, so the whole acquisition chain (recorder → sync → IMU calibration → QC →
``datasets.build``) can run end-to-end without hardware:

- **hand state** per step type (rest, flat-hand calibration, 3-tap sync, per-finger flexion,
  open/close, wrist rotation, free motion, contact-free "air grasp" shapes, pinch / fist / finger
  crossing / finger-to-palm self-touch, D2 reach → grasp → manipulate → release → retreat) as
  analytic functions of time,
  blended across step transitions;
- **pressure** ``raw[T,C]`` (channel order): per-taxel baseline ~1e6 counts (mk555 scale) with
  ΔS% = motion artefact (per-taxel gain on the parent finger's flexion + lagged velocity term)
  + slow drift + noise − press depth (**press → negative ΔS**, SATS sign; deep presses reach the
  saturation cap);
- **IMU** via ``pose.imu_model.synthesize_imu`` on the MANO skeleton (gravity along −z), then
  per-site mounting ``M`` and an IMU-world heading ``G`` (rotation about the gravity axis, as an
  AHRS reference frame): ``q_raw = G ⊗ q ⊗ M``, ``v_raw = R_Mᵀ v`` (the recorded calibration
  offsets should come back as ``conj(M)``; the wrist mounting is identity because one static
  pose cannot separate it from ``G``);
- **cameras** (tiny RGB frames whose blobs follow the wrist/fingers/object — the 3 sync taps are
  visible as motion), **hand_pose** labels (what offline HaMeR/WiLoR would give, with confidence
  dropouts), **object_pose** (D2) and **joint_state** (robot kind: 15 finger joints).

Each stream has its own nominal rate, host-clock latency, clock drift (ppm), jitter and drops, so
``sync.sync_session`` has something real to recover. Ground truth is in :attr:`FakeScene.truth`.
Everything is seeded; no hidden global RNG.

Limitation: the press is **scripted per contact site** (pinch → thumb + finger tip, fist, palm touch …)
and is not derived from the MANO capsule geometry that ``datasets.build`` turns into ``self_touch`` /
``contact_label`` — on a fake D1 session most pressed samples are labelled unknown (−1) and a few
geometric contacts carry no press. A ``--fake`` session checks the software chain (record → sync →
calibration → QC → preprocessing), not contact-label quality: train / evaluate the contact detector on
``datasets.synthetic`` sessions, whose press comes from the same geometry as the labels.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from common.layouts import Layout, load_layout
from common.signal import ADC_MAX
from robot_skin.geometry.rotations import (
    aa_to_matrix, matrix_to_aa, matrix_to_quat, quat_fix_continuity, quat_mul, quat_normalize, quat_to_matrix,
)

from .protocol import EpisodePlan, Step, SyncSpec, TimedStep, Timing

__all__ = ["FakeConfig", "FakeScene", "default_fake_timeline", "ROBOT_JOINT_NAMES"]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_FI = {f: i for i, f in enumerate(FINGERS)}
#: robot kind joint names (FINGERS order × 3 joints)
ROBOT_JOINT_NAMES = tuple(f"{f}_j{k}" for f in FINGERS for k in (1, 2, 3))

_REST = np.array([0.25, 0.30, 0.35, 0.40, 0.45])
_TAP_FLEX = np.array([0.5, 0.05, 1.2, 1.3, 1.3])
_PRESHAPE = {"power": [0.3, 0.15, 0.15, 0.15, 0.15], "precision": [0.35, 0.2, 0.2, 0.6, 0.7],
             "pinch": [0.35, 0.2, 0.6, 0.8, 0.9], "press": [0.8, 0.05, 1.2, 1.3, 1.3],
             "lateral": [0.2, 0.6, 0.7, 0.8, 0.9], "tripod": [0.35, 0.2, 0.2, 0.7, 0.8]}
_GRIP = {"power": [0.7, 0.9, 0.9, 0.9, 0.9], "precision": [0.6, 0.6, 0.6, 0.7, 0.8],
         "pinch": [0.6, 0.6, 0.7, 0.8, 0.9], "press": [0.8, 0.05, 1.2, 1.3, 1.3],
         "lateral": [0.5, 0.9, 1.0, 1.0, 1.0], "tripod": [0.6, 0.6, 0.6, 0.9, 1.0]}
_GRIP_SITES = {"power": {"thumb": 25, "index": 25, "middle": 25, "ring": 20, "pinky": 15, "palm": 20},
               "precision": {"thumb": 30, "index": 30, "middle": 25},
               "pinch": {"thumb": 30, "index": 30}, "press": {"index": 40},
               "lateral": {"thumb": 30, "index": 20}, "tripod": {"thumb": 30, "index": 25, "middle": 25}}
#: "air grasp" (D1, no object): the grip shape stops this fraction of the way from pre-shape to grip
_AIR_CLOSURE = 0.8
#: joint weights (MCP, PIP, DIP) when one flexion angle drives a finger
_JOINT_W = {"thumb": (0.6, 0.8, 0.8)}
_DEFAULT_JW = (1.0, 1.1, 0.8)
_WRIST_AXES = {"pronation_supination": (1.0, 0.0, 0.0), "flexion_extension": (0.0, 0.0, 1.0),
               "radial_ulnar": (0.0, 1.0, 0.0)}
_WRIST_AMP = {"pronation_supination": 1.0, "flexion_extension": 0.8, "radial_ulnar": 0.35}
#: palm-down base orientation in the pose world (MANO canonical: palm faces −y → world −z)
_GO_BASE = np.array([math.pi / 2, 0.0, 0.0])
_HOME = np.array([0.0, 0.0, 0.12])
_TABLE_Z = 0.0
_TAP_HEIGHT = 0.03       # m fingertip travel of a sync tap
_TAP_TRAVEL_S = 0.15     # s descent (and rebound) time
_TAP_CONTACT_S = 0.07    # s fingertip on the pad
_TAP_DEPTH = 35.0        # % press depth (−ΔS) of a tap


@dataclass
class FakeConfig:
    """Sensor-realism knobs of the synthetic scene (all per stream; camera keys ``camera_<name>``)."""

    rates: dict = field(default_factory=lambda: {"pressure": 200.0, "imu": 100.0, "camera": 30.0,
                                                 "hand_pose": 30.0, "object_pose": 30.0, "joint_state": 100.0})
    latency_s: dict = field(default_factory=lambda: {"pressure": 0.0, "imu": 0.012, "camera_ego": 0.045,
                                                     "camera_third": 0.030, "camera": 0.040, "joint_state": 0.004})
    clock_ppm: dict = field(default_factory=lambda: {"imu": 20.0, "camera_ego": -30.0, "camera_third": 40.0})
    jitter_s: dict = field(default_factory=lambda: {"pressure": 0.0003, "imu": 0.0006, "camera": 0.0015,
                                                    "joint_state": 0.0004})
    drop_prob: dict = field(default_factory=lambda: {"camera": 0.003})
    image_hw: tuple = (24, 32)
    imu_mount_deg: float = 20.0          # random mounting of the non-wrist IMU sites
    baseline_raw: tuple = (0.8e6, 1.6e6)
    noise_pct: float = 0.03
    drift_pct_per_min: float = 0.15
    artifact_pct_per_rad: float = 1.5
    velocity_pct_per_rad_s: float = 0.3
    artifact_lag_s: float = 0.15
    hand_label_noise_rad: float = 0.01
    hand_conf_dropout: float = 0.02
    gyro_noise: float = 0.01
    acc_noise: float = 0.05

    def rate(self, stream: str) -> float:
        base = "camera" if stream.startswith("camera_") else stream
        return float(self.rates.get(stream, self.rates.get(base)))

    def get(self, table: str, stream: str, default: float = 0.0) -> float:
        d = getattr(self, table)
        base = "camera" if stream.startswith("camera_") else stream
        return float(d.get(stream, d.get(base, default)))


def default_fake_timeline(*, rest_s: float = 1.0, free_s: float = 2.0, pinch_s: float = 2.0,
                          sync: SyncSpec | None = None) -> tuple[list[TimedStep], float]:
    """A short standalone timeline (rest → 3-tap sync → flat hand → free motion → pinch → rest) for
    Fake*Sources built without a protocol. Returns ``(timeline, duration_s)``."""
    sync = sync or SyncSpec()
    steps = [
        Step("baseline_start", "baseline_start", "static", rest_s, "none", ("no_contact",), motion={"type": "rest"}),
        Step("sync_start", "sync_start", "sync", sync.duration_s, "any", ("sync",),
             motion={"type": "sync_taps", "tap_times": sync.tap_offsets(), "finger": sync.finger}, scalable=False),
        Step("imu_calibration", "imu_calibration", "calibration", 1.5, "none", ("calibration", "no_contact"),
             motion={"type": "flat_hand"}),
        Step("free_motion", "free_motion", "motion", free_s, "none", ("no_contact",), motion={"type": "free"}),
        Step("pinch_index", "pinch", "self_touch", pinch_s, "self", ("self_touch",),
             motion={"type": "pinch", "finger": "index", "cycle_s": 1.0}),
        Step("baseline_end", "baseline_end", "static", rest_s, "none", ("no_contact",), motion={"type": "rest"}),
    ]
    ep = EpisodePlan(0, steps)
    timing = Timing(lead_in_s=0.5, transition_s=0.3, lead_out_s=0.5)
    return ep.timeline(timing), ep.duration_s(timing)


# ── small math helpers ───────────────────────────────────────────────────────
def _cyc(tau: np.ndarray, cycle: float) -> np.ndarray:
    """0 → 1 → 0 once per ``cycle`` (starts at 0)."""
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * tau / cycle))


def _mj(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk profile on [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def _smooth(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp(a, b, w):
    return np.asarray(a)[None] * (1.0 - w[:, None]) + np.asarray(b)[None] * w[:, None]


def _contact_profile(s: np.ndarray, on: float, depth: np.ndarray | float) -> np.ndarray:
    """Press depth when a 0..1 closure signal exceeds ``on`` (smooth onset)."""
    x = np.clip((s - on) / max(1.0 - on, 1e-6), 0.0, 1.0)
    return np.asarray(depth) * x ** 0.7


def _sum_sines(tau: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Normalised sum of sinusoids; ``p[k] = (freq, phase, amp)``."""
    out = np.zeros_like(tau)
    for f, ph, a in p:
        out += a * np.sin(2.0 * math.pi * f * tau + ph)
    return out / max(float(np.sum(np.abs(p[:, 2]))), 1e-9)


def _rand_quat(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = math.radians(rng.uniform(0.3, 1.0) * max_deg)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * ax]


def _stamps(tau: np.ndarray, latency: float, ppm: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Host stamps of samples taken at true times ``tau``: drift + latency + jitter, strictly increasing."""
    s = tau * (1.0 + ppm * 1e-6) + latency + (rng.normal(0.0, jitter, tau.shape) if jitter > 0 else 0.0)
    s = np.maximum.accumulate(s)
    bump = np.r_[0.0, np.diff(s)] <= 0
    if bump.any():
        s = s + np.cumsum(bump) * 1e-7
    return s


def _lowpass(x: np.ndarray, t: np.ndarray, tau_c: float) -> np.ndarray:
    """First-order lag along axis 0 for (near-)uniform sampling (scipy if available)."""
    if x.shape[0] < 2 or tau_c <= 0:
        return x.copy()
    dt = float(np.median(np.diff(t)))
    a = 1.0 - math.exp(-dt / tau_c)
    try:
        from scipy.signal import lfilter
        zi = (1.0 - a) * x[:1]
        y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=0, zi=zi)
        return y
    except ImportError:  # pragma: no cover - scipy is optional
        y = np.empty_like(x)
        y[0] = x[0]
        for k in range(1, x.shape[0]):
            y[k] = y[k - 1] + a * (x[k] - y[k - 1])
        return y


# ── the scene ────────────────────────────────────────────────────────────────
class FakeScene:
    """Synthetic world for one session (see module docstring).

    ``timeline`` — ``TimedStep`` list on the session clock (t = 0 at recorder start);
    ``duration_s`` — session length; ``kind`` — ``glove`` (IMU + hand pose + cameras) or ``robot``
    (joint_state); ``task`` — the episode's ``manifest.task`` dict (D2: object/target/grasp).
    Stream getters return ``{"stamp": host stamps (s, relative to recorder start), <arrays>}`` and
    are cached.
    """

    def __init__(self, timeline: Sequence[TimedStep], duration_s: float, *, kind: str = "glove",
                 layout: str | Layout = "glove_template", cameras: Sequence[str] = ("ego", "third"),
                 task: Mapping | None = None, seed: int = 0, config: FakeConfig | None = None,
                 skeleton=None):
        if kind not in ("glove", "robot"):
            raise ValueError(f"kind must be glove|robot, got {kind!r}")
        if not duration_s > 0:
            raise ValueError("duration_s must be > 0")
        self.timeline = sorted(timeline, key=lambda ts: ts.t0)
        self.duration_s = float(duration_s)
        self.kind = kind
        self.layout = layout if isinstance(layout, Layout) else load_layout(layout)
        self.cameras = tuple(cameras) if kind == "glove" else ()
        self.task = dict(task) if task else None
        self.seed = int(seed)
        self.cfg = config or FakeConfig()
        self._skeleton = skeleton
        self._cache: dict[str, dict] = {}
        rng = np.random.default_rng([self.seed, 7])
        self._params = [self._draw_step_params(ts.step, np.random.default_rng([self.seed, 100 + i]))
                        for i, ts in enumerate(self.timeline)]
        self._t0 = np.array([ts.t0 for ts in self.timeline])
        self._t1 = np.array([ts.t1 for ts in self.timeline])
        self._sites = self._contact_sites()
        C = int(self.layout.channels.max()) + 1
        self.n_channels = C
        lo, hi = self.cfg.baseline_raw
        self.baseline_raw = np.round(rng.uniform(lo, hi, C))
        self._gain = rng.uniform(-1.0, 1.0, self.layout.n) * self.cfg.artifact_pct_per_rad
        self._vgain = rng.uniform(-1.0, 1.0, self.layout.n) * self.cfg.velocity_pct_per_rad_s
        self._drift = rng.uniform(-1.0, 1.0, self.layout.n) * self.cfg.drift_pct_per_min
        self._ep = self._draw_episode(rng)
        S = len(self.layout.imu_sites)
        self.imu_mount = np.tile([1.0, 0.0, 0.0, 0.0], (S, 1))
        for i, s in enumerate(self.layout.imu_sites):
            if s.parent != "wrist":
                self.imu_mount[i] = _rand_quat(rng, self.cfg.imu_mount_deg)
        # IMU (AHRS) world = pose world rotated about the gravity axis (z) only: fusion filters
        # align their z with gravity, only the heading is arbitrary — keeps quat and acc consistent
        yaw = rng.uniform(-math.pi, math.pi)
        self.imu_world = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        self.truth: dict[str, Any] = {
            "latency_s": {k: self.cfg.get("latency_s", k) for k in self.stream_names},
            "clock_ppm": {k: self.cfg.get("clock_ppm", k) for k in self.stream_names},
            "imu_mount": self.imu_mount.tolist(), "imu_world": self.imu_world.tolist(),
            "tap_times": self.tap_times().tolist(), "seed": self.seed,
        }

    # ── construction helpers ─────────────────────────────────────────────
    @classmethod
    def from_episode(cls, episode: EpisodePlan, timing: Timing, **kw) -> "FakeScene":
        return cls(episode.timeline(timing), episode.duration_s(timing), task=episode.task, **kw)

    @classmethod
    def default(cls, **kw) -> "FakeScene":
        """Short standalone scene (:func:`default_fake_timeline`)."""
        tl, dur = default_fake_timeline()
        return cls(tl, dur, **kw)

    @property
    def stream_names(self) -> list[str]:
        if self.kind == "robot":
            names = ["pressure", "joint_state"]
        else:
            names = ["pressure", "imu"] + [f"camera_{c}" for c in self.cameras]
            if self.cameras:
                names.append("hand_pose")
        if self.task:
            names.append("object_pose")
        return names

    @property
    def skeleton(self):
        if self._skeleton is None:
            from robot_skin.pose.mano import ManoSkeleton
            self._skeleton = ManoSkeleton()
        return self._skeleton

    def tap_times(self) -> np.ndarray:
        """Absolute times of every sync tap in the timeline."""
        out = []
        for ts in self.timeline:
            if ts.step.motion.get("type") == "sync_taps":
                out.extend(ts.t0 + float(x) for x in ts.step.motion.get("tap_times", ()))
        return np.asarray(out, dtype=np.float64)

    def _contact_sites(self) -> dict[str, np.ndarray]:
        """Contact site name → taxel indices (layout order): fingertips, palm and palm halves."""
        L, g = self.layout, self.layout.groups
        sites: dict[str, np.ndarray] = {}
        for f in FINGERS:
            idx = [i for i in g.get(f"finger_{f}", []) if i in set(g.get("fingertip", g.get(f"finger_{f}", [])))]
            if not idx:
                idx = [i for i, t in enumerate(L.taxels) if t.parent.startswith(f)]
            sites[f] = np.asarray(idx, dtype=np.int64)
        palm = [i for i in g.get("palm", [])] or [i for i, t in enumerate(L.taxels) if t.parent in ("palm", "palm_link")]
        palm = np.asarray(palm, dtype=np.int64)
        sites["palm"] = palm
        if palm.size:
            z = L.positions[palm, 2]
            med = float(np.median(z))
            sites["palm_distal"] = palm[z >= med]
            sites["palm_proximal"] = palm[z < med] if (z < med).any() else palm
        else:
            sites["palm_distal"] = sites["palm_proximal"] = palm
        return sites

    def _draw_step_params(self, step: Step, rng: np.random.Generator) -> dict:
        m = step.motion
        p: dict[str, Any] = {"tremor_phase": rng.uniform(0, 2 * math.pi, 5)}
        typ = m.get("type")
        if typ == "free":
            p["flex"] = [np.c_[rng.uniform(0.15, 1.2, 3), rng.uniform(0, 2 * math.pi, 3), rng.uniform(0.3, 1.0, 3)]
                         for _ in range(5)]
            p["abd"] = [np.c_[rng.uniform(0.1, 0.8, 2), rng.uniform(0, 2 * math.pi, 2), rng.uniform(0.3, 1.0, 2)]
                        for _ in range(5)]
            p["rot"] = [np.c_[rng.uniform(0.1, 0.7, 2), rng.uniform(0, 2 * math.pi, 2), rng.uniform(0.3, 1.0, 2)]
                        for _ in range(3)]
            p["pos"] = [np.c_[rng.uniform(0.1, 0.6, 2), rng.uniform(0, 2 * math.pi, 2), rng.uniform(0.3, 1.0, 2)]
                        for _ in range(3)]
        cycle = float(m.get("cycle_s", 2.0) or 2.0)
        n_cyc = max(1, int(math.ceil(step.duration_s / cycle)) + 1)
        p["depth"] = rng.uniform(15.0, 40.0, n_cyc)
        p["deep_cycle"] = int(rng.integers(0, n_cyc))
        return p

    def _draw_episode(self, rng: np.random.Generator) -> dict:
        o = np.array([-0.25, rng.uniform(-0.08, 0.08), 0.05])
        side = 1.0 if rng.random() < 0.5 else -1.0
        g = o + np.array([rng.uniform(-0.05, 0.05), side * 0.2, 0.0])
        return {"obj": o, "target": g, "pregrasp": o + np.array([0.07, 0.0, 0.03])}

    # ── hand state ───────────────────────────────────────────────────────
    def hand_state(self, t) -> dict[str, np.ndarray]:
        """State at times ``t`` [n]: flex5, abd5, wrot (local wrist rotation aa), wp (wrist position),
        press [n, N] (press depth %, layout order, press-positive)."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        n, N = t.shape[0], self.layout.n
        out = {"flex": np.tile(_REST, (n, 1)), "abd": np.zeros((n, 5)), "wrot": np.zeros((n, 3)),
               "wp": np.tile(_HOME, (n, 1)), "press": np.zeros((n, N))}
        if not self.timeline:
            return out
        K = len(self.timeline)
        for k, ts in enumerate(self.timeline):
            m = (t >= ts.t0) & (t <= ts.t1)
            if m.any():
                self._assign(out, m, self._gen(k, t[m] - ts.t0))
            nxt = self.timeline[k + 1].t0 if k + 1 < K else math.inf
            g = (t > ts.t1) & (t < nxt)
            if g.any():
                end = self._gen(k, np.full(int(g.sum()), ts.step.duration_s))
                if k + 1 < K:
                    start = self._gen(k + 1, np.zeros(int(g.sum())))
                    w = _smooth((t[g] - ts.t1) / max(nxt - ts.t1, 1e-9))
                    blend = {key: _lerp(end[key][0], start[key][0], w) for key in ("flex", "abd", "wrot", "wp")}
                    blend["press"] = np.zeros((int(g.sum()), N))
                    self._assign(out, g, blend)
                else:
                    end["press"] = np.zeros_like(end["press"])
                    self._assign(out, g, end)
        before = t < self.timeline[0].t0
        if before.any():
            st = self._gen(0, np.zeros(int(before.sum())))
            st["press"] = np.zeros_like(st["press"])
            self._assign(out, before, st)
        return out

    @staticmethod
    def _assign(out: dict, mask: np.ndarray, vals: dict) -> None:
        for key in ("flex", "abd", "wrot", "wp", "press"):
            out[key][mask] = vals[key]

    def _press(self, n: int, sites: Mapping[str, np.ndarray]) -> np.ndarray:
        P = np.zeros((n, self.layout.n))
        for name, depth in sites.items():
            idx = self._sites.get(name)
            if idx is not None and idx.size:
                P[:, idx] = np.maximum(P[:, idx], np.asarray(depth, dtype=np.float64).reshape(n, 1))
        return P

    def _gen(self, k: int, tau: np.ndarray) -> dict[str, np.ndarray]:
        step, P = self.timeline[k].step, self._params[k]
        m = step.motion
        typ = str(m.get("type", "rest"))
        n = tau.shape[0]
        D = float(step.duration_s)
        cyc_len = float(m.get("cycle_s", 2.0) or 2.0)
        trem = 0.01 * np.sin(2 * math.pi * 7.0 * tau[:, None] + P["tremor_phase"][None])
        flex = np.tile(_REST, (n, 1)) + trem
        abd = np.zeros((n, 5))
        wrot = np.zeros((n, 3))
        wp = np.tile(_HOME, (n, 1))
        sites: dict[str, np.ndarray] = {}
        cyc_idx = np.clip((tau // cyc_len).astype(int), 0, len(P["depth"]) - 1)

        if typ in ("rest", "home", "static"):
            pass
        elif typ in ("flat_hand", "calibration"):
            flex = 0.1 * trem                                        # ~0.001 rad tremor only (still)
        elif typ == "sync_taps":
            # ballistic tap at each onset t_k: fast descent (max speed at impact), contact plateau
            # of TAP_CONTACT_S, rebound (max speed at lift-off) — impact and lift-off are both
            # visible to pressure (edges), IMU (acceleration spikes) and cameras (motion)
            flex = np.tile(_TAP_FLEX, (n, 1)) + 0.3 * trem
            height = np.zeros(n)
            press = np.zeros(n)
            w, c = _TAP_TRAVEL_S, _TAP_CONTACT_S
            for tk in m.get("tap_times", ()):
                r = tau - float(tk)
                down = (r > -w) & (r < 0.0)
                up = (r > c) & (r < c + w)
                height += np.where(down, _TAP_HEIGHT * np.sin(0.5 * math.pi * (-r) / w), 0.0)
                height += np.where(up, _TAP_HEIGHT * np.sin(0.5 * math.pi * (r - c) / w), 0.0)
                press += _TAP_DEPTH * _smooth(r / 0.008) * _smooth((c - r) / 0.008) * (1.0 - 0.15 * np.clip(r / c, 0, 1))
            taps = np.asarray(m.get("tap_times", ()), dtype=np.float64)
            if taps.size:                                   # hand hovers TAP_HEIGHT above the pad between taps
                near = np.zeros(n, dtype=bool)
                for tk in taps:
                    near |= (tau > tk - w) & (tau < tk + c + w)
                height = np.where(near, height, _TAP_HEIGHT)
            wp[:, 2] += height - _TAP_HEIGHT
            sites[str(m.get("finger", "index"))] = press
        elif typ == "finger_flex":
            f = _FI[m.get("finger", "index")]
            flex = 0.5 * flex
            amp = (0.9 if f == 0 else 1.3) * float(m.get("amplitude", 1.0))
            flex[:, f] = 0.05 + amp * _cyc(tau, cyc_len)
        elif typ == "open_close":
            amp = 0.9 if step.speed == "fast" else 1.05
            flex = 0.05 + amp * _cyc(tau, cyc_len)[:, None] + trem
        elif typ == "wrist_rotation":
            ax = m.get("axis", "pronation_supination")
            wrot = np.asarray(_WRIST_AXES[ax])[None] * (_WRIST_AMP[ax] * np.sin(2 * math.pi * tau / cyc_len))[:, None]
        elif typ == "free":
            flex = np.stack([np.clip(0.5 + 0.4 * _sum_sines(tau, P["flex"][i]), 0.0, 1.0) for i in range(5)], 1)
            abd = np.stack([0.12 * _sum_sines(tau, P["abd"][i]) for i in range(5)], 1)
            wrot = np.stack([a * _sum_sines(tau, P["rot"][i]) for i, a in enumerate((0.5, 0.35, 0.2))], 1)
            wp = wp + np.stack([0.05 * _sum_sines(tau, P["pos"][i]) for i in range(3)], 1)
        elif typ == "pinch":
            f = _FI[m.get("finger", "index")]
            s = _cyc(tau, cyc_len)
            flex = 0.6 * flex
            flex[:, 0] = 0.3 + 0.5 * s
            flex[:, f] = 0.2 + (1.0 if f >= 3 else 0.9) * s
            abd[:, 0] = 0.3 * s
            n_c = len(P["depth"])
            ramp = 15.0 + 25.0 * cyc_idx / max(n_c - 1, 1)             # light → firm across cycles
            d = _contact_profile(s, 0.8, ramp)
            sites = {"thumb": d, FINGERS[f]: d}
        elif typ == "fist":
            s = _cyc(tau, cyc_len)
            flex = 0.1 + 1.45 * s[:, None] + trem
            d = _contact_profile(s, 0.75, 30.0)
            deep = np.where(cyc_idx == P["deep_cycle"], 96.0, 25.0)
            sites = {"thumb": 0.7 * d, "index": d, "middle": d, "ring": d, "pinky": d,
                     "palm_distal": _contact_profile(s, 0.75, deep), "palm_proximal": 0.6 * d}
        elif typ == "air_grasp":
            # D2 grasp shape around an imaginary object, formed and released each cycle; no press
            g = str(m.get("grasp", "power"))
            pre = np.asarray(_PRESHAPE.get(g, _PRESHAPE["power"]))
            air = pre + _AIR_CLOSURE * (np.asarray(_GRIP.get(g, _GRIP["power"])) - pre)
            flex = _REST + _cyc(tau, cyc_len)[:, None] * (air - _REST) + trem
        elif typ == "finger_crossing":
            s = _cyc(tau, cyc_len)
            flex[:, 1] = 0.3 + 0.2 * s
            flex[:, 2] = 0.3 + 0.1 * s
            abd[:, 1], abd[:, 2] = 0.35 * s, -0.35 * s
            sites = {"index": _contact_profile(s, 0.7, 10.0)}
        elif typ == "palm_touch":
            f = _FI[m.get("finger", "index")]
            s = _cyc(tau, cyc_len)
            flex = 0.5 * flex
            if f == 0:
                flex[:, 0] = 0.1 + 1.0 * s
                abd[:, 0] = 0.5 * s
            else:
                flex[:, f] = 0.1 + 1.5 * s
            d = _contact_profile(s, 0.8, P["depth"][cyc_idx])
            sites = {FINGERS[f]: d, ("palm_proximal" if f in (0, 4) else "palm_distal"): 0.8 * d}
        elif typ in ("reach", "grasp", "manipulate", "release", "retreat"):
            flex, abd, wrot, wp, sites = self._task_phase(typ, m, tau, D, flex, abd, wrot, wp)
        else:
            raise ValueError(f"fake scene: unknown motion type {typ!r} in step {step.id!r}")
        return {"flex": flex, "abd": abd, "wrot": wrot, "wp": wp, "press": self._press(n, sites)}

    def _task_phase(self, typ, m, tau, D, flex, abd, wrot, wp):
        grasp = str(m.get("grasp", "power"))
        manip = str(m.get("manipulate", "transport"))
        pre = np.asarray(_PRESHAPE.get(grasp, _PRESHAPE["power"]))
        grip = np.asarray(_GRIP.get(grasp, _GRIP["power"]))
        E = self._ep
        pg = E["pregrasp"] + (np.array([0.0, 0.0, 0.03]) if grasp == "press" else 0.0)
        pl = self._place_point(manip, pg)
        u = np.clip(tau / max(D, 1e-9), 0.0, 1.0)
        n = tau.shape[0]
        site_depth = _GRIP_SITES.get(grasp, _GRIP_SITES["power"])
        level = np.zeros(n)
        if typ == "reach":
            wp = _lerp(_HOME, pg, _mj(u))
            flex = _lerp(_REST, pre, _mj(u))
        elif typ == "grasp":
            wp = _lerp(pg, pg - np.array([0.0, 0.0, 0.01]), _mj(u))
            flex = _lerp(pre, grip, _mj(u / 0.7))
            level = _smooth((u - 0.45) / 0.35)
        elif typ == "manipulate":
            start = pg if grasp == "press" else pg - np.array([0.0, 0.0, 0.01])
            flex = np.tile(grip, (n, 1))
            level = np.ones(n)
            if manip == "transport":
                lift = start + np.array([0.0, 0.0, 0.10])
                over = pl + np.array([0.0, 0.0, 0.10])
                wp = np.where((u < 0.25)[:, None], _lerp(start, lift, _mj(u / 0.25)),
                              np.where((u < 0.75)[:, None], _lerp(lift, over, _mj((u - 0.25) / 0.5)),
                                       _lerp(over, pl, _mj((u - 0.75) / 0.25))))
                level = 1.0 + 0.2 * _smooth(u / 0.25)
            elif manip == "pour":
                pour = E["target"] + np.array([0.07, 0.0, 0.12])
                wp = np.where((u < 0.3)[:, None], _lerp(start, pour, _mj(u / 0.3)),
                              np.where((u < 0.85)[:, None], np.tile(pour, (n, 1)), _lerp(pour, pl, _mj((u - 0.85) / 0.15))))
                wrot[:, 0] = 1.5 * np.sin(math.pi * np.clip((u - 0.3) / 0.55, 0.0, 1.0))
                level = 1.0 + 0.15 * np.sin(2 * math.pi * 1.5 * tau)
            elif manip == "insert":
                above = pl + np.array([0.0, 0.0, 0.05])
                wp = np.where((u < 0.6)[:, None], _lerp(start, above, _mj(u / 0.6)), _lerp(above, pl, _mj((u - 0.6) / 0.4)))
                wp[:, 1] += np.where(u > 0.6, 0.003 * np.sin(2 * math.pi * 3.0 * tau), 0.0)
                level = 1.0 + 0.5 * _smooth((u - 0.8) / 0.2)
            elif manip == "twist":
                wp = np.tile(start, (n, 1))
                wrot[:, 1] = 0.6 * np.sin(2 * math.pi * 2.0 * u)
                level = 0.8 + 0.2 * np.cos(2 * math.pi * 2.0 * u)
            elif manip == "wipe":
                down = start + np.array([0.0, 0.0, -0.02])
                wp = np.tile(down, (n, 1))
                wp[:, 1] += 0.08 * np.sin(2 * math.pi * 1.0 * tau) * _smooth(u / 0.1) * _smooth((1 - u) / 0.1)
                level = 1.0 + 0.3 * np.sin(2 * math.pi * 2.0 * tau)
            elif manip == "handover":
                wp = _lerp(start, pl, _mj(u / 0.7))
                level = 1.0 - 0.3 * _smooth((u - 0.7) / 0.3)
            elif manip == "press":
                wp = np.tile(start, (n, 1))
                wp[:, 2] -= 0.03 * np.sin(math.pi * u)
                level = np.clip(np.sin(math.pi * u) * 1.3 - 0.3, 0.0, None)
            elif manip == "rotate":
                wp = _lerp(start, start + np.array([0.0, 0.0, 0.05]), _mj(u / 0.2))
                wrot[:, 2] = 0.3 * u
                for i, f in enumerate((0, 1, 2)):
                    flex[:, f] = grip[f] + 0.2 * np.sin(2 * math.pi * 1.0 * tau + i * 2.1)
                level = 0.7 + 0.3 * np.cos(2 * math.pi * 1.0 * tau)
            else:
                wp = np.tile(start, (n, 1))
        elif typ == "release":
            wp = _lerp(pl, pl + np.array([0.0, 0.0, 0.02]), _mj(u))
            flex = _lerp(grip, pre, _mj(u))
            level = 1.0 - _smooth(u / 0.5)
            if grasp == "press":
                level = np.zeros(n)
        elif typ == "retreat":
            p0 = pl + np.array([0.0, 0.0, 0.02])
            wp = _lerp(p0, _HOME, _mj(u))
            flex = _lerp(pre, _REST, _mj(u))
        sites = {k: v * level for k, v in site_depth.items()} if level.any() else {}
        return flex, abd, wrot, wp, sites

    def _place_point(self, manip: str, pg: np.ndarray) -> np.ndarray:
        E = self._ep
        if manip == "transport":
            return E["target"] + np.array([0.07, 0.0, 0.02])
        if manip == "pour":
            return E["target"] + np.array([0.07, 0.08, 0.02])
        if manip == "insert":
            return E["target"] + np.array([0.07, 0.0, 0.0])
        if manip == "handover":
            return E["obj"] + np.array([-0.25, 0.0, 0.15])
        if manip == "rotate":
            return pg + np.array([0.0, 0.0, 0.04])
        return pg - np.array([0.0, 0.0, 0.01])

    # ── derived kinematics ───────────────────────────────────────────────
    @staticmethod
    def _flex15(flex5: np.ndarray) -> np.ndarray:
        """Per-finger flexion → per-joint flexion in MANO joint order (1..15)."""
        from robot_skin.pose.mano import FINGER_JOINTS
        out = np.zeros((flex5.shape[0], 15))
        for i, f in enumerate(FINGERS):
            for j, w in zip(FINGER_JOINTS[f], _JOINT_W.get(f, _DEFAULT_JW)):
                out[:, j - 1] = w * flex5[:, i]
        return out

    def hand_pose(self, t) -> dict[str, np.ndarray]:
        """MANO labels at ``t``: global_orient[n,3], finger_pose[n,15,3], wrist_pos[n,3] (+ state)."""
        st = self.hand_state(t)
        with torch.no_grad():
            R = aa_to_matrix(torch.as_tensor(_GO_BASE))[None] @ aa_to_matrix(torch.as_tensor(st["wrot"]))
            go = matrix_to_aa(R).numpy()
            fp = self.skeleton.flexion_pose(self._flex15(st["flex"]), st["abd"]).numpy()
        return {"global_orient": go, "finger_pose": fp, "wrist_pos": st["wp"], "R": R.numpy(), "state": st}

    def robot_q(self, t) -> np.ndarray:
        """Robot kind: 15 finger joints (FINGERS order × 3) = weighted per-finger flexion."""
        st = self.hand_state(t)
        q = np.zeros((st["flex"].shape[0], 15))
        for i, f in enumerate(FINGERS):
            for k, w in enumerate(_JOINT_W.get(f, _DEFAULT_JW)):
                q[:, 3 * i + k] = w * st["flex"][:, i]
        return q

    def _true_times(self, stream: str, rng: np.random.Generator) -> np.ndarray:
        rate = self.cfg.rate(stream)
        phase = rng.uniform(0.0, 1.0 / rate)
        return phase + np.arange(int(math.floor((self.duration_s - phase) * rate)) + 1) / rate

    def _stream_rng(self, stream: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, sum(map(ord, stream)), len(stream)])

    def _host(self, stream: str, tau: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return _stamps(tau, self.cfg.get("latency_s", stream), self.cfg.get("clock_ppm", stream),
                       self.cfg.get("jitter_s", stream), rng)

    # ── streams ──────────────────────────────────────────────────────────
    def pressure_stream(self) -> dict[str, np.ndarray]:
        if "pressure" in self._cache:
            return self._cache["pressure"]
        rng = self._stream_rng("pressure")
        tau = self._true_times("pressure", rng)
        st = self.hand_state(tau)
        L = self.layout
        drv5 = st["flex"]                                 # glove: finger flexion; robot: same drive
        driver = np.zeros((tau.shape[0], L.n))
        for f, idx in ((f, self._sites[f]) for f in FINGERS):
            driver[:, idx] = drv5[:, _FI[f]][:, None]
        palm = self._sites["palm"]
        if palm.size:
            driver[:, palm] = drv5[:, 1:].mean(1, keepdims=True)
        lag = _lowpass(driver, tau, self.cfg.artifact_lag_s)
        vel = _lowpass(np.gradient(driver, tau, axis=0), tau, self.cfg.artifact_lag_s)
        delta = (self._gain * lag + self._vgain * vel + self._drift * (tau[:, None] / 60.0)
                 + rng.normal(0.0, self.cfg.noise_pct, lag.shape) - st["press"])
        delta = np.maximum(delta, -100.0)
        raw = np.tile(self.baseline_raw, (tau.shape[0], 1))
        noise_other = rng.normal(0.0, self.cfg.noise_pct, raw.shape) / 100.0
        raw = raw * (1.0 + noise_other)
        raw[:, L.channels] = self.baseline_raw[L.channels] * (1.0 + delta / 100.0)
        raw = np.clip(np.round(raw), 0.0, ADC_MAX)
        out = {"stamp": self._host("pressure", tau, rng), "raw": raw, "tau": tau}
        self._cache["pressure"] = out
        return out

    def imu_stream(self) -> dict[str, np.ndarray]:
        if self.kind != "glove":
            raise ValueError("IMU stream exists for the glove kind only")
        if "imu" in self._cache:
            return self._cache["imu"]
        from robot_skin.pose.imu_model import synthesize_imu
        rng = self._stream_rng("imu")
        tau = self._true_times("imu", rng)
        hp = self.hand_pose(tau)
        d = synthesize_imu(self.layout, self.skeleton, tau, hp["global_orient"], hp["finger_pose"], hp["wrist_pos"])
        q = torch.as_tensor(d["quat"])
        M = torch.as_tensor(self.imu_mount)[None].expand_as(q)
        G = torch.as_tensor(self.imu_world).expand_as(q)
        noise = torch.as_tensor(np.concatenate([np.ones(q.shape[:-1] + (1,)),
                                                rng.normal(0.0, 0.002, q.shape[:-1] + (3,))], -1))
        q_raw = quat_fix_continuity(quat_normalize(quat_mul(quat_mul(G, q), quat_mul(M, noise))).numpy(), axis=0)
        RM = quat_to_matrix(torch.as_tensor(self.imu_mount)).numpy()             # [S,3,3]
        gyro = np.einsum("sji,tsj->tsi", RM, d["gyro"]) + rng.normal(0.0, self.cfg.gyro_noise, d["gyro"].shape)
        acc = np.einsum("sji,tsj->tsi", RM, d["acc"]) + rng.normal(0.0, self.cfg.acc_noise, d["acc"].shape)
        out = {"stamp": self._host("imu", tau, rng), "quat": q_raw, "gyro": gyro, "acc": acc,
               "sites": np.array([s.name for s in self.layout.imu_sites]), "tau": tau}
        self._cache["imu"] = out
        return out

    def hand_pose_stream(self) -> dict[str, np.ndarray]:
        """Vision-style MANO labels at the (first) camera's true frame times (already on the
        reference clock, as labels produced offline after sync)."""
        if "hand_pose" in self._cache:
            return self._cache["hand_pose"]
        rng = self._stream_rng("hand_pose")
        tau = self._true_times("hand_pose", rng)
        hp = self.hand_pose(tau)
        s = self.cfg.hand_label_noise_rad
        n = tau.shape[0]
        conf = np.clip(0.92 + rng.normal(0.0, 0.03, n), 0.0, 1.0)
        k = 0
        while k < n:
            if rng.random() < self.cfg.hand_conf_dropout:
                ln = int(rng.integers(3, 9))
                conf[k:k + ln] = rng.uniform(0.05, 0.3)
                k += ln
            k += 1
        out = {"stamp": tau.copy(), "global_orient": hp["global_orient"] + rng.normal(0, s, (n, 3)),
               "finger_pose": hp["finger_pose"] + rng.normal(0, s, (n, 15, 3)),
               "wrist_pos": hp["wrist_pos"] + rng.normal(0, 0.002, (n, 3)), "confidence": conf, "tau": tau}
        self._cache["hand_pose"] = out
        return out

    def _object_track(self, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        E = self._ep
        n = tau.shape[0]
        pos = np.tile(E["obj"], (n, 1))
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        spans = {ts.step.id: ts for ts in self.timeline}
        grasp = str((self.task or {}).get("grasp", "power"))
        if grasp == "press" or "manipulate" not in spans:
            return pos, quat
        g = spans.get("grasp", spans["manipulate"])
        t_att = g.t0 + (0.6 * (g.t1 - g.t0) if g.step.id == "grasp" else 0.0)
        r = spans.get("release")
        t_det = r.t0 + 0.3 * (r.t1 - r.t0) if r else spans["manipulate"].t1
        hp_ref = self.hand_pose(np.array([t_att, t_det]))
        w_att, R_att = hp_ref["wrist_pos"][0], hp_ref["R"][0]
        rel = E["obj"] - w_att                           # object offset from the wrist at attachment
        att = (tau >= t_att) & (tau <= t_det)
        after = tau > t_det
        if att.any():                                    # rigidly attached: rotates with the hand
            hp = self.hand_pose(tau[att])
            Rrel = hp["R"] @ R_att.T[None]
            pos[att] = hp["wrist_pos"] + np.einsum("nij,j->ni", Rrel, rel)
            quat[att] = matrix_to_quat(torch.as_tensor(Rrel)).numpy()
        if after.any():                                  # left where it was released
            Rd = hp_ref["R"][1] @ R_att.T
            pos[after] = hp_ref["wrist_pos"][1] + Rd @ rel
            if str((self.task or {}).get("manipulate")) == "handover":   # partner takes it away
                pos[after] += np.array([-0.15, 0.0, 0.0]) * _smooth((tau[after] - t_det) / 0.5)[:, None]
            quat[after] = matrix_to_quat(torch.as_tensor(Rd)).numpy()
        return pos, np.where(quat[:, :1] < 0, -quat, quat)

    def object_pose_stream(self) -> dict[str, np.ndarray]:
        if "object_pose" in self._cache:
            return self._cache["object_pose"]
        rng = self._stream_rng("object_pose")
        tau = self._true_times("object_pose", rng)
        pos, quat = self._object_track(tau)
        out = {"stamp": tau.copy(), "pos": pos + rng.normal(0, 0.001, pos.shape), "quat": quat, "tau": tau}
        self._cache["object_pose"] = out
        return out

    def joint_stream(self) -> dict[str, np.ndarray]:
        if "joint_state" in self._cache:
            return self._cache["joint_state"]
        rng = self._stream_rng("joint_state")
        tau = self._true_times("joint_state", rng)
        q = self.robot_q(tau)
        qd = np.gradient(q, tau, axis=0)
        press = self.hand_state(tau)["press"]
        tip_load = np.zeros_like(q)
        for i, f in enumerate(FINGERS):
            idx = self._sites[f]
            if idx.size:
                tip_load[:, 3 * i + 2] = press[:, idx].max(1) * 0.004
        tau_j = 0.05 * q + tip_load + rng.normal(0.0, 0.002, q.shape)
        out = {"stamp": self._host("joint_state", tau, rng), "q": q + rng.normal(0, 1e-4, q.shape), "qd": qd,
               "tau": tau_j, "names": np.array(ROBOT_JOINT_NAMES), "tau_true": tau}
        self._cache["joint_state"] = out
        return out

    def camera_stream(self, camera: str) -> dict[str, np.ndarray]:
        name = f"camera_{camera}"
        if name in self._cache:
            return self._cache[name]
        if camera not in self.cameras:
            raise KeyError(f"camera {camera!r} not in scene cameras {self.cameras}")
        rng = self._stream_rng(name)
        tau = self._true_times(name, rng)
        keep = rng.random(tau.shape[0]) >= self.cfg.get("drop_prob", name)
        keep[0] = keep[-1] = True
        tau = tau[keep]
        frames = self._render(camera, tau, rng)
        out = {"stamp": self._host(name, tau, rng), "frame": frames, "tau": tau}
        self._cache[name] = out
        return out

    def _render(self, camera: str, tau: np.ndarray, rng: np.random.Generator, chunk: int = 256) -> np.ndarray:
        """Tiny RGB frames: background gradient + palm / finger / object blobs (+ ego head sway)."""
        H, W = (int(x) for x in self.cfg.image_hw)
        hp = self.hand_pose(tau)
        wp, R = hp["wrist_pos"], hp["R"]
        mflex = hp["state"]["flex"][:, 1:].mean(1)
        blobs = []                                       # (points [F,3], sigma_m, colour)
        if self.task:
            opos, _ = self._object_track(tau)
            ocol = np.random.default_rng(sum(map(ord, str(self.task.get("object", "")))) + 3).uniform(60, 255, 3)
            blobs.append((opos, 0.025, ocol))
        blobs.append((wp + np.einsum("nij,j->ni", R, np.array([-0.045, 0.0, 0.0])), 0.03, np.array([200.0, 160.0, 140.0])))
        tips = wp + np.einsum("nij,nj->ni", R, np.c_[-0.09 * (1.0 - 0.45 * mflex), np.zeros((len(tau), 2))])
        blobs.append((tips, 0.02, np.array([215.0, 175.0, 150.0])))
        ppm = W / 0.35
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        crng = np.random.default_rng([self.seed, sum(map(ord, camera))])
        bg = crng.uniform(40, 90, 3)[None, None, :] + 25.0 * (yy / H)[:, :, None]         # [H,W,3]
        out = np.empty((len(tau), H, W, 3), dtype=np.uint8)
        for a in range(0, len(tau), chunk):
            b = min(a + chunk, len(tau))
            tt = tau[a:b]
            img = np.broadcast_to(bg, (b - a, H, W, 3)).copy()
            for pts, sig, col in blobs:
                p = pts[a:b]
                if camera == "ego":                      # head-mounted, oblique down-forward view, slight sway
                    sway = 0.01 * np.sin(2 * math.pi * 0.3 * tt)
                    u = W / 2 + (p[:, 1] + sway) * ppm
                    v = H / 2 + (0.6 * (p[:, 0] + 0.15) - 0.8 * (p[:, 2] - 0.1) + sway) * ppm
                else:                                    # fixed side view
                    u = W / 2 + (-p[:, 0] - 0.12) * ppm
                    v = 0.75 * H - (p[:, 2] - _TABLE_Z) * ppm
                s = sig * ppm
                al = np.exp(-((xx[None] - u[:, None, None]) ** 2 + (yy[None] - v[:, None, None]) ** 2) / (2 * s * s))
                img = img * (1.0 - al[..., None]) + col[None, None, None, :] * al[..., None]
            img += rng.normal(0.0, 2.0, img.shape)
            out[a:b] = np.clip(np.round(img), 0, 255).astype(np.uint8)
        return out

    def stream(self, name: str) -> dict[str, np.ndarray]:
        """Stream by name (``pressure``, ``imu``, ``camera_<c>``, ``hand_pose``, ``object_pose``,
        ``joint_state``)."""
        if name == "pressure":
            return self.pressure_stream()
        if name == "imu":
            return self.imu_stream()
        if name == "hand_pose":
            return self.hand_pose_stream()
        if name == "object_pose":
            return self.object_pose_stream()
        if name == "joint_state":
            return self.joint_stream()
        if name.startswith("camera_"):
            return self.camera_stream(name[len("camera_"):])
        raise KeyError(f"unknown fake stream {name!r}")

    def replace_config(self, **changes) -> "FakeScene":
        """Copy of this scene with some :class:`FakeConfig` fields replaced."""
        return FakeScene(self.timeline, self.duration_s, kind=self.kind, layout=self.layout, cameras=self.cameras,
                         task=self.task, seed=self.seed, config=dataclasses.replace(self.cfg, **changes),
                         skeleton=self._skeleton)
