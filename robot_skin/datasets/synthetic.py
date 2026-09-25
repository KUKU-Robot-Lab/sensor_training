"""Synthetic raw sessions (D1 ``motion`` / D2 ``task``, glove / robot) in the exact acquisition format.

Purpose: tests, smoke runs and pipeline development without hardware. :func:`generate_session`
writes one complete *raw* session directory — the same files a real recording produces
(``acquisition.manifest`` docstring / ``docs/DATA_FORMAT.md``)::

    session.json          SessionManifest v2 (segments, task, calibration, meta)
    pressure.npz          t[T] float64 s, raw[T,C] float64 ADC counts, **channel-major**
    imu.npz               t, quat[T,S,4] wxyz (w ≥ 0), gyro[T,S,3] rad/s, acc[T,S,3] m/s², sites  (glove)
    hand_pose.npz         t, global_orient, finger_pose[T,15,3], wrist_pos, confidence         (glove)
    joint_state.npz       t, q[T,D], qd[T,D], names[D]                                          (robot)
    robot.urdf            synthetic 16-DoF hand whose links match robot_hand_template          (robot)
    object_pose.npz       t, pos[T,3], quat[T,4] wxyz                                           (task)
    camera_<name>/        timestamps.npy [F] + frames.npy uint8[F,H,W,3]
    events.jsonl          phase_start / phase_end (+ instruction / success for tasks)
    gt_synthetic.npz      ground truth of the generator (not a stream; see load_ground_truth)
    layout.yaml           only when a Layout *object* was passed (manifest.layout points to it)

Kinematics (ground truth, evaluated at every stream's own timestamps)
- Glove: MANO hand from anatomical flexion/abduction angles (``pose.mano.ManoSkeleton
  .flexion_pose``; MANO: Romero et al., SIGGRAPH Asia 2017), global orientation and wrist
  position. D1 follows the protocol blocks of ``acquisition/protocols/d1_motion.yaml``
  (``imu_calibration`` flat hand = calibration + no_contact, ``baseline_start``, slow/fast
  open–close sweeps, wrist rotation, free motion, thumb–finger pinches, fist, ``baseline_end``);
  D2 runs baseline (flat hand, then the relaxed start pose) → reach (travelling above the object,
  descending onto it) → grasp → manipulate → release (fingers open, palm lifts off) → retreat →
  baseline_end around a spherical object; on failure (``task.success = False``) the object slips
  out during manipulate.
- Robot: the synthetic URDF (:func:`robot_hand_urdf`, links ``palm_link``, ``<finger>_distal_link``
  … as in ``common/layouts/robot_hand_template.yaml``) driven by joint trajectories with the same
  block structure (no wrist / IMU; the static start holds q = 0 before moving to the relaxed
  pose). Its thumb only opposes index and middle, so robot pinches use those two (pads pressed
  ``robot_pinch_depth`` together) and robot grasps are thumb–index(–middle) pinch grasps. Object
  poses of robot sessions are in the hand base frame.

Contact (labels are geometric, so they agree with preprocessing)
- Self-touch (glove) is decided by ``pose.mano.self_touch_from_hand`` itself — the function
  ``datasets.build`` uses — and its magnitude is the penetration into the same margin-inflated
  capsules (``r + margin − d``). Robot self-touch uses ``contact.self_touch_labels`` on link capsules
  (radius ``robot_link_radius``; fingertip taxels sit on the link axis, so their skin surface is
  ``robot_pad_inset`` further out, palm taxels are on the surface).
- Object contact: penetration of a taxel into the object sphere (``R + margin − d``). In D2 power
  grasps the thumb pad may also come within the self-touch margin of the index finger; that is
  labelled (and pressed) as self-touch, exactly as preprocessing would label it.
- Phase events carry ``{"contact": none|self|object, "labels": [...]}`` like the acquisition
  recorder; segments are one per block label (+ one ``task`` span in D2), and ``no_contact``
  spans are trimmed around any geometric contact or press lag tail (|press| > ``PRESS_EPS_PCT``),
  so the label is always truthful. By design nothing is trimmed in sessions of ≥ 6 s; very short
  sessions (fast blocks) can lose a few hundred ms (``meta.synthetic.no_contact_trimmed_s``).

Tactile model (per taxel i, per pressure sample; ΔS in % with the SATS sign, press → negative)::

    u_i = Σ_j W_ij θ_j              local joint angle (glove: anatomical flexion, robot: q)
    x_i = g_i u_i + c_i u_i² + h_i Σ_j W_ij θ̇_j                    angle AND velocity
    τ_i ẏ_i = x_i − y_i                                   first-order lag → motion artefact y_i
    ΔS_i = y_i − P_i (1 − exp(−k_i pen_i / P_i)) (lagged) + drift_i(t) + noise
    raw_i = b_i (1 + ΔS_i / 100),  clipped to the ADC rails, rounded to counts; some contact
                                   runs drop out to ADC_MIN for 50–300 ms around their peak

``W`` couples a taxel to the joints of its own finger (distal-weighted) or, for palm taxels, to
nearby knuckles. The artefact is zero for a flat hand / ``q = 0`` (the calibration pose), and every
session (D1 and D2, glove and robot) starts its first ``no_contact`` segment there, so that segment
gives the true baseline and all sessions share one ΔS reference. Raw columns are in *channel* order: layout
taxel i is written to column ``layout.channels[i]`` (``layout.by_channel(raw)`` undoes it); unused
channels carry baseline + noise.

Other streams: IMUs from ``pose.imu_model.synthesize_imu`` (sensor frame = segment frame) with a
known per-site mounting ``M`` (wrist = identity) and IMU-world yaw ``G``: ``q_raw = G ⊗ q ⊗ M``,
``v_raw = R_Mᵀ v`` + bias + noise, stored in ``manifest.calibration`` via
``imu_calibration_to_dict(conj(M), G, sites)``. hand_pose.npz ≈ 30 Hz with label noise and
occasional low-confidence dropouts. Cameras render tiny frames whose finger bands shorten and
darken with flexion, shift with the wrist and show the object; their timestamps carry a small
constant clock offset (``meta.synthetic.camera_offset_s``). Every stream has its own rate error and
timestamp jitter.

Ground truth for later stages: ``gt_synthetic.npz`` (:func:`load_ground_truth`) holds, at the
pressure timestamps and in layout order, the no-contact artefact ``artefact_pct``, the press term,
drift, contact masks, penetration, saturation, true baselines and the artefact parameters, so a
test can check that a baseline stage recovers the artefact.

Everything is deterministic in ``seed`` (numpy ``SeedSequence`` streams per component; torch is
used only for deterministic CPU kinematics).
"""
from __future__ import annotations

import json
import math
import numbers
import shutil
import zlib
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from common.layouts import MANO_SEGMENTS, Layout, load_layout
from common.signal import ADC_MAX, ADC_MIN
from robot_skin.acquisition.manifest import SessionManifest, StreamInfo
from robot_skin.contact.self_touch import finger_of, point_segment_distance, self_touch_labels
from robot_skin.geometry.rotations import aa_to_matrix, matrix_to_aa, matrix_to_quat, quat_mul
from robot_skin.pose.imu_model import imu_calibration_to_dict, synthesize_imu
from robot_skin.pose.mano import (
    DEFAULT_SELF_TOUCH_EXCLUDE, FINGER_JOINTS, FINGERS, MANO_JOINTS, ManoSkeleton, self_touch_from_hand,
    taxel_poses_from_hand,
)
from robot_skin.pose.robot_fk import taxel_poses_from_joints
from robot_skin.pose.urdf import URDFModel
from robot_skin.pose.vision_hand import save_hand_labels

__all__ = [
    "GENERATOR_VERSION", "GT_FILE", "URDF_FILE", "EVENTS_FILE", "LAYOUT_FILE", "TASK_PHASES",
    "CONTACT_PHASES", "CONTACT_LABELS", "MOTION_BLOCKS", "SYNTH_TASKS", "OBJECT_RADIUS", "ROBOT_JOINT_NAMES",
    "PRESS_EPS_PCT",
    "SynthParams", "Block", "plan_motion_session", "plan_task_session", "robot_hand_urdf",
    "generate_session", "generate_dataset", "load_ground_truth",
]

GENERATOR_VERSION = 2                  # bump when the generated data change for the same seed
GT_FILE = "gt_synthetic.npz"
URDF_FILE = "robot.urdf"
EVENTS_FILE = "events.jsonl"
LAYOUT_FILE = "layout.yaml"
DEFAULT_LAYOUTS = {"glove": "glove_template", "robot": "robot_hand_template"}
TASK_PHASES: tuple[str, ...] = ("reach", "grasp", "manipulate", "release", "retreat")
CONTACT_PHASES: tuple[str, ...] = ("grasp", "manipulate")
GRAVITY = (0.0, 0.0, -9.81)
MIN_DURATION_S = {"motion": 2.5, "task": 3.0}
#: |press ΔS| (%) above which a sample counts as pressed for segment truthfulness (well below the
#: sensor noise floor): ``no_contact`` spans are trimmed around contact *and* the press lag tail.
PRESS_EPS_PCT = 0.02
REACH_LIFT_M = 0.05                    # glove D2 reach: travel height above the straight-line path

#: D1 blocks between the static start and end, cycled to fill the session:
#: (name, generator, contact expectation, nominal duration s, info). Names follow the step ids of
#: ``acquisition/protocols/d1_motion.yaml``. Robot sessions skip ``wrist`` blocks and pinches the
#: synthetic thumb cannot reach (ring, pinky).
MOTION_BLOCKS: tuple[tuple[str, str, str, float, dict], ...] = (
    ("open_close_slow", "sweep", "none", 1.2, {"speed": "slow"}),
    ("pinch_index", "pinch", "self", 1.0, {"finger": "index"}),
    ("open_close_fast", "sweep", "none", 0.8, {"speed": "fast"}),
    ("pinch_middle", "pinch", "self", 1.0, {"finger": "middle"}),
    ("wrist_rotation", "wrist", "none", 1.2, {}),
    ("pinch_ring", "pinch", "self", 1.0, {"finger": "ring"}),
    ("free_motion", "free", "none", 1.5, {}),
    ("fist", "fist", "self", 1.0, {}),
    ("pinch_pinky", "pinch", "self", 1.0, {"finger": "pinky"}),
)
#: segment labels implied by a contact expectation (= ``acquisition.protocol.DEFAULT_CONTACT_LABELS``).
CONTACT_LABELS: dict[str, tuple[str, ...]] = {"none": ("no_contact",), "self": ("self_touch",), "object": ()}

#: Minimal D2 task catalog mirrored for synthetic data (the real catalog lives in
#: ``acquisition/protocols/d2_task.yaml``). ``manip`` selects the manipulate-phase motion,
#: ``grasp`` the grasp synthesis (power: palm-anchored wrap; precision: thumb–index pinch).
SYNTH_TASKS: dict[str, dict] = {
    "grasp_lift_place": {"objects": ("cup", "block", "ball"), "targets": ("plate", "tray", "box"),
                         "template": "pick up the {object} and place it on the {target}",
                         "manip": "lift_move", "grasp": "power"},
    "pour": {"objects": ("cup", "bottle"), "targets": ("bowl", "glass"),
             "template": "pour the {object} into the {target}", "manip": "rotate", "grasp": "power"},
    "wipe": {"objects": ("sponge",), "targets": ("table", "tray"),
             "template": "wipe the {target} with the {object}", "manip": "oscillate", "grasp": "power"},
    "open_jar": {"objects": ("jar",), "targets": (), "template": "open the {object}",
                 "manip": "twist", "grasp": "precision"},
    "handover": {"objects": ("block", "ball"), "targets": ("person",),
                 "template": "hand the {object} to the {target}", "manip": "lift_move", "grasp": "precision"},
    "in_hand_rotate": {"objects": ("ball", "block"), "targets": (),
                       "template": "rotate the {object} in your hand", "manip": "finger_gait",
                       "grasp": "precision"},
}
#: object sphere radius (m) used for contact geometry.
OBJECT_RADIUS: dict[str, float] = {"cup": 0.034, "block": 0.030, "ball": 0.030, "bottle": 0.032,
                                   "sponge": 0.032, "jar": 0.036}


# ── parameters ──────────────────────────────────────────────────────────────
@dataclass
class SynthParams:
    """Knobs of the generator (defaults = realistic-but-tiny). Ranges ``(lo, hi)`` are sampled per
    taxel / site / session with the session seed."""

    pressure_hz: float = 200.0
    imu_hz: float = 100.0
    joint_hz: float = 100.0
    hand_pose_hz: float = 30.0
    object_hz: float = 30.0
    camera_fps: float = 30.0
    rate_error: float = 0.002                 # ± relative device clock-rate error
    jitter: float = 0.1                       # timestamp jitter std, fraction of the period (clipped ±0.3)
    camera_offset_s: tuple[float, float] = (-0.02, 0.03)   # constant camera clock offset
    baseline_raw: tuple[float, float] = (4.0e6, 9.0e6)     # true per-taxel baseline (ADC counts)
    noise_pct: tuple[float, float] = (0.03, 0.10)          # white noise std (% of baseline)
    drift_pct: float = 0.4                    # max |linear drift| over the session (%)
    drift_wave_pct: float = 0.1               # max slow (20–60 s) drift oscillation (%)
    artefact_gain: tuple[float, float] = (1.0, 3.0)        # |g| %/rad (sign random)
    artefact_vel_gain: tuple[float, float] = (0.2, 0.6)    # |h| %/(rad/s)
    artefact_quad: float = 0.6                # max |c| %/rad²
    artefact_tau_s: tuple[float, float] = (0.04, 0.20)     # lag time constant
    press_gain: tuple[float, float] = (2.5, 5.0)           # k: % per mm penetration (small press)
    press_max_pct: tuple[float, float] = (55.0, 80.0)      # P: soft saturation of the press response
    press_tau_s: tuple[float, float] = (0.01, 0.03)        # fast press dynamics
    sat_prob: float = 0.15                    # P(a contact run of a taxel drops out to the ADC rail)
    sat_duration_s: tuple[float, float] = (0.05, 0.3)      # length of such a dropout (around peak press)
    self_touch_margin: float = 0.004          # = pose.mano.self_touch_from_hand default
    object_margin: float = 0.002
    robot_link_radius: float = 0.008          # capsule radius of the synthetic robot's finger links
    robot_self_margin: float = 0.002
    robot_pad_inset: float = 0.008            # robot template fingertip taxels sit on the link axis
    robot_pinch_depth: float = 0.003          # pad-surface interference at the robot pinch key
    hand_label_noise: tuple[float, float, float] = (0.01, 0.02, 0.002)  # go rad, finger rad, wrist m
    hand_drop_rate: float = 0.5               # low-confidence dropouts per second
    imu_mount_deg: float = 20.0               # max mounting rotation (non-wrist sites)
    imu_quat_noise_deg: float = 0.2
    imu_gyro_noise: float = 0.01              # rad/s
    imu_gyro_bias: float = 0.005              # rad/s
    imu_acc_noise: float = 0.05               # m/s²
    joint_noise: float = 5e-4                 # rad
    success_prob: float = 0.85
    image_noise: float = 3.0                  # uint8 levels

    _RATES = ("pressure_hz", "imu_hz", "joint_hz", "hand_pose_hz", "object_hz", "camera_fps")
    _POSITIVE = ("baseline_raw", "artefact_tau_s", "press_gain", "press_max_pct", "press_tau_s",
                 "sat_duration_s", "robot_link_radius")
    _PROBS = ("sat_prob", "success_prob")

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            vals = tuple(v) if isinstance(v, (tuple, list)) else (v,)
            if not all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)
                       for x in vals):
                raise ValueError(f"SynthParams.{f.name} must be finite number(s), got {v!r}")
            if isinstance(f.default, tuple) and len(vals) != len(f.default):
                raise ValueError(f"SynthParams.{f.name} needs {len(f.default)} values, got {v!r}")
            if isinstance(f.default, tuple) and len(vals) == 2 and vals[0] > vals[1]:
                raise ValueError(f"SynthParams.{f.name} range must be (lo, hi) with lo ≤ hi, got {v!r}")
            if f.name != "camera_offset_s" and min(vals) < 0:
                raise ValueError(f"SynthParams.{f.name} must be ≥ 0, got {v!r}")
            if f.name in self._RATES + self._POSITIVE and min(vals) <= 0:
                raise ValueError(f"SynthParams.{f.name} must be > 0, got {v!r}")
            if f.name in self._PROBS and not 0.0 <= v <= 1.0:
                raise ValueError(f"SynthParams.{f.name} must be in [0, 1], got {v!r}")
        if self.rate_error >= 0.5:
            raise ValueError(f"SynthParams.rate_error must be < 0.5, got {self.rate_error!r}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "SynthParams":
        if d is None:
            return cls()
        if isinstance(d, SynthParams):
            return d
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"unknown SynthParams keys {unknown}; valid: {sorted(names)}")
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        return cls(**kw)


# ── small numeric helpers ───────────────────────────────────────────────────
def _rng(seed: int, *keys: str) -> np.random.Generator:
    """Independent, reproducible stream per component (adding a camera never changes tactile)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [zlib.crc32(k.encode()) for k in keys]))


def _mj(x):
    """Min-jerk ramp 0→1 on x∈[0,1] (C² at both ends), clipped outside."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def _bump(x):
    """C² bump 64·x³(1−x)³ on [0,1] (peak 1 at 0.5), 0 outside."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return 64.0 * (x * (1.0 - x)) ** 3


def _window(tau, dur: float, ramp: float = 0.25):
    """Min-jerk fade-in/fade-out envelope over a block (0 with zero slope at both ends)."""
    r = max(1e-6, min(ramp, 0.3 * dur))
    return _mj(tau / r) * _mj((dur - tau) / r)


def _R(aa) -> np.ndarray:
    return aa_to_matrix(torch.as_tensor(np.asarray(aa, dtype=np.float64))).numpy()


def _aa(R) -> np.ndarray:
    return matrix_to_aa(torch.as_tensor(np.asarray(R, dtype=np.float64))).numpy()


def _compose(base_aa, delta_aa) -> np.ndarray:
    """axis-angle of R(base)·R(delta) (broadcast)."""
    return _aa(_R(base_aa) @ _R(delta_aa))


def _rel(a_aa, b_aa) -> np.ndarray:
    """axis-angle δ with R(a)·R(δ) = R(b)."""
    return _aa(_R(a_aa).T @ _R(b_aa))


def _quat(R) -> np.ndarray:
    return matrix_to_quat(torch.as_tensor(np.asarray(R, dtype=np.float64))).numpy()


def _qmul(a, b) -> np.ndarray:
    return quat_mul(torch.as_tensor(np.array(a, dtype=np.float64)),
                    torch.as_tensor(np.array(b, dtype=np.float64))).numpy()


def _hemisphere(q: np.ndarray) -> np.ndarray:
    """Device convention w ≥ 0 (introduces the sign flips preprocessing must continuity-fix)."""
    return np.where(q[..., :1] < 0, -q, q)


def _random_rotation_aa(rng: np.random.Generator, max_angle: float, min_angle: float = 0.0) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(min_angle, max_angle)


def _lag(x: np.ndarray, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """First-order lag ``τ ẏ = x − y`` on irregular timestamps (exact ZOH discretisation), y₀ = x₀."""
    y = np.empty_like(x)
    y[0] = x[0]
    dt = np.diff(t)
    alpha = 1.0 - np.exp(-dt[:, None] / np.asarray(tau)[None, :])
    for k in range(1, x.shape[0]):
        y[k] = y[k - 1] + alpha[k - 1] * (x[k] - y[k - 1])
    return y


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a 1-D bool array as [start, end) index pairs."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def _stream_times(rng: np.random.Generator, rate: float, t_end: float, p: SynthParams) -> np.ndarray:
    """Host timestamps of one device: own clock-rate error, start offset and bounded jitter
    (strictly increasing, within [0, t_end])."""
    true_rate = rate * (1.0 + rng.uniform(-p.rate_error, p.rate_error))
    t0 = rng.uniform(0.0, min(0.5 / rate, 0.01))
    n = int(math.floor((t_end - t0) * true_rate)) + 1
    jit = np.clip(rng.normal(0.0, p.jitter, n), -0.3, 0.3) / true_rate
    t = t0 + np.arange(n) / true_rate + jit
    return t[(t >= 0.0) & (t <= t_end)]


def _safe_name(s: str) -> bool:
    """Usable as one path component (camera / subject / session names)."""
    return bool(s) and s not in (".", "..") and "/" not in s and "\\" not in s


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ── session plans ───────────────────────────────────────────────────────────
@dataclass
class Block:
    """One protocol block / task phase on the session clock (s).

    ``labels`` become manifest segments over the block (``no_contact`` spans are trimmed around
    any geometric contact or press tail); ``contact`` is the expectation (none | self | object).
    Both are written into the ``phase_start`` event value like the acquisition recorder does, so
    ``acquisition.recorder.segments_from_events`` reproduces the block segments (unless trimmed).
    """

    name: str                     # phase name written to events.jsonl
    gen: str                      # motion generator key
    t0: float
    t1: float
    labels: tuple[str, ...] = ()
    contact: str = "none"
    info: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def _check_duration(duration_s: float, dataset: str) -> float:
    d = float(duration_s)
    if not math.isfinite(d) or d < MIN_DURATION_S[dataset]:
        raise ValueError(f"duration_s must be ≥ {MIN_DURATION_S[dataset]} s for {dataset} sessions, got {duration_s}")
    return d


def plan_motion_session(duration_s: float = 6.0, kind: str = "glove") -> list[Block]:
    """D1 block plan: static start → :data:`MOTION_BLOCKS` cycled (at least three) and time-scaled
    to fill → ``baseline_end``. Glove: ``imu_calibration`` (flat hand, palm down; labels
    calibration + no_contact) then ``baseline_start`` (move to the relaxed pose); robot:
    ``baseline_start`` (hold q = 0, then move to the relaxed pose). The flat hand / q = 0 has zero
    motion artefact, so the first no_contact segment yields the true baseline."""
    if kind not in DEFAULT_LAYOUTS:
        raise ValueError(f"kind must be 'glove' or 'robot', got {kind!r}")
    D = _check_duration(duration_s, "motion")
    cycle = [b for b in MOTION_BLOCKS
             if kind == "glove" or (b[1] != "wrist" and b[4].get("finger") not in ("ring", "pinky"))]
    s = float(np.clip(0.18 * D, 0.8, 5.0))
    e = float(np.clip(0.12 * D, 0.5, 5.0))
    rest = D - s - e
    chosen: list[tuple] = []
    total = 0.0
    while len(chosen) < 3 or total < 0.9 * rest:
        b = cycle[len(chosen) % len(cycle)]
        chosen.append(b)
        total += b[3]
    scale = rest / total
    if kind == "glove":
        h = 0.65 * s
        blocks = [Block("imu_calibration", "calib_hold", 0.0, h, ("calibration", "no_contact"), "none",
                        {"pose": "flat_hand"}),
                  Block("baseline_start", "to_neutral", h, s, ("no_contact",), "none", {"pose": "rest"})]
    else:
        blocks = [Block("baseline_start", "static_start", 0.0, s, ("no_contact",), "none", {"hold": 0.65 * s})]
    t = s
    for name, gen, contact, nominal, info in chosen:
        blocks.append(Block(name, gen, t, t + nominal * scale, CONTACT_LABELS[contact], contact, dict(info)))
        t += nominal * scale
    blocks[-1].t1 = D - e
    blocks.append(Block("baseline_end", "static_end", D - e, D, ("no_contact",), "none", {}))
    return blocks


def plan_task_session(duration_s: float = 6.0) -> list[Block]:
    """D2 phase plan (``acquisition/protocols/d2_task.yaml`` vocabulary): ``baseline`` (flat hand /
    q = 0 held, then the relaxed start pose; no_contact) → reach → grasp → manipulate → release →
    retreat → ``baseline_end`` (no_contact). Contact expectation: object for grasp / manipulate /
    release, none otherwise; the task phases carry no segment labels (the ``task`` segment
    spanning them is added separately)."""
    D = _check_duration(duration_s, "task")
    frac = (("baseline", 0.12), ("reach", 0.16), ("grasp", 0.14), ("manipulate", 0.28),
            ("release", 0.12), ("retreat", 0.10), ("baseline_end", 0.08))
    gens = {"baseline": "task_start", "baseline_end": "task_end"}
    blocks, t = [], 0.0
    for name, f in frac:
        static = name in gens
        contact = "object" if name in ("grasp", "manipulate", "release") else "none"
        blocks.append(Block(name, gens.get(name, name), t, t + f * D, ("no_contact",) if static else (), contact,
                            {"contact_expected": name in CONTACT_PHASES}))
        t += f * D
    blocks[-1].t1 = D
    return blocks


def _dispatch(blocks: Sequence[Block], t: np.ndarray, handlers: Mapping[int, Callable], out_shapes: dict):
    """Evaluate per-block generators at arbitrary times (times outside the plan clamp to the ends)."""
    t = np.asarray(t, dtype=np.float64)
    t0 = np.array([b.t0 for b in blocks])
    idx = np.clip(np.searchsorted(t0, t, side="right") - 1, 0, len(blocks) - 1)
    out = {k: np.zeros((t.shape[0],) + shp) for k, shp in out_shapes.items()}
    for i, b in enumerate(blocks):
        m = idx == i
        if not m.any():
            continue
        tau = np.clip(t[m] - b.t0, 0.0, b.dur)
        res = handlers[i](b, tau)
        for k in out:
            out[k][m] = res[k]
    return out


# ── glove: pad points, pinch keys, grasp synthesis ─────────────────────────
_PAD_DEPTH = 0.006   # fingertip pad: 6 mm palmar of the distal bone axis …
_PAD_FRAC = 0.5      # … halfway along the distal phalanx (glove_template pads sit there)
_SEG3 = [MANO_SEGMENTS.index(f + "3") for f in FINGERS]

#: Pad-to-pad pinch keys for the default ManoSkeleton, found offline with :func:`_solve_glove_pinch`
#: (300 steps): (thumb flex j1..j3, thumb abduction, finger flex j1..j3, finger abduction), pad
#: distance (m). Verified at run time (pad distance must match); re-optimised if the geometry changed.
_GLOVE_PINCH_DEFAULT: dict[str, tuple[tuple[float, ...], float]] = {
    "index": ((0.7173, -0.1043, -0.6139, -0.0068, 0.9867, 1.2236, 0.6985, -0.0692), 0.00060),
    "middle": ((0.9207, -0.0229, 0.0388, -0.7704, 1.4143, 1.2582, 1.1836, 0.7419), 0.00068),
    "ring": ((1.0775, 0.1502, -0.2248, -1.2177, 1.4085, 1.1675, 0.9826, 0.2680), 0.00046),
    "pinky": ((1.1110, 0.1711, -0.5376, -1.5180, 1.5256, 0.9791, 0.5951, -0.2790), 0.00070),
}
_KEY_TOL = 0.0005   # m, allowed deviation from the stored pad distance


@dataclass
class _Cache:
    """Process-wide caches of pure functions of the (fixed) default geometry only."""

    glove_pinch: dict = field(default_factory=dict)
    robot_pinch: dict = field(default_factory=dict)
    robot_model: Any = None


_CACHE = _Cache()


@lru_cache(maxsize=1)
def _skeleton() -> ManoSkeleton:
    """The default skeleton — preprocessing uses ``ManoSkeleton()`` too, so labels agree."""
    return ManoSkeleton()


def _glove_pad_local(sk: ManoSkeleton) -> np.ndarray:
    tl = np.linalg.norm(sk.tip_offsets, axis=1)
    return np.stack([np.array([0.0, -_PAD_DEPTH, _PAD_FRAC * tl[i]]) for i in range(5)])   # [5,3] FINGERS


def _glove_pads(sk: ManoSkeleton, flex, abd):
    """Fingertip pad points / pad normals ``[...,5,3]`` in the hand (joint-0) frame (torch)."""
    fp = sk.flexion_pose(flex, abd)
    ST = sk.segment_transform_tensor(sk.forward(None, fp))[..., _SEG3, :, :]          # [...,5,4,4]
    pl = torch.as_tensor(_glove_pad_local(sk), dtype=ST.dtype)
    pads = (ST[..., :3, :3] @ pl[..., None])[..., 0] + ST[..., :3, 3]
    return pads, -ST[..., :3, 1]


def _pinch_flex(finger: str, params):
    """(flex15, abd5) torch from pinch params [8] (differentiable)."""
    params = torch.as_tensor(params, dtype=torch.float64)
    tj = torch.tensor([j - 1 for j in FINGER_JOINTS["thumb"]])
    fj = torch.tensor([j - 1 for j in FINGER_JOINTS[finger]])
    flex = torch.zeros(15, dtype=torch.float64).index_put((tj,), params[:3]).index_put((fj,), params[4:7])
    abd = torch.zeros(5, dtype=torch.float64).index_put((torch.tensor([0]),), params[3:4])
    abd = abd.index_put((torch.tensor([FINGERS.index(finger)]),), params[7:8])
    return flex, abd


def _glove_pinch_distance(sk: ManoSkeleton, finger: str, params) -> tuple[torch.Tensor, torch.Tensor]:
    flex, abd = _pinch_flex(finger, params)
    pads, nrm = _glove_pads(sk, flex, abd)
    fi = FINGERS.index(finger)
    return (pads[0] - pads[fi]).norm(), (nrm[0] * nrm[fi]).sum()


def _solve_glove_pinch(sk: ManoSkeleton, finger: str, steps: int = 300) -> np.ndarray:
    """Thumb + finger flexion/abduction bringing the two fingertip pads together, pad-to-pad
    (Adam on pad distance + opposing normals + a small angle prior)."""
    init = torch.tensor([0.3, 0.3, 0.3, 0.0, 0.5, 0.5, 0.5, 0.0], dtype=torch.float64)
    p = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(steps):
        d, nn = _glove_pinch_distance(sk, finger, p)
        loss = d + 0.002 * nn + 1e-4 * p.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p.detach().numpy().copy()


def _glove_pinch_keys(sk: ManoSkeleton) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """finger → (flex15, abd5) of the pinch key (only thumb + finger entries non-zero)."""
    key = id(sk)
    if key not in _CACHE.glove_pinch:
        out = {}
        for f in ("index", "middle", "ring", "pinky"):
            params, d_ref = _GLOVE_PINCH_DEFAULT[f]
            params = np.asarray(params, dtype=np.float64)
            with torch.no_grad():
                d, _ = _glove_pinch_distance(sk, f, params)
            ok = abs(float(d) - d_ref) < _KEY_TOL
            if not ok:
                params = _solve_glove_pinch(sk, f)
            flex, abd = _pinch_flex(f, params)
            out[f] = (flex.numpy(), abd.numpy())
        _CACHE.glove_pinch[key] = out
    return _CACHE.glove_pinch[key]


def _glove_open_pose(neutral_flex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-shape for grasping: fingers nearly straight and spread, thumb abducted."""
    flex = 0.4 * neutral_flex
    abd = np.array([0.45, 0.12, 0.0, -0.12, -0.22])
    return flex, abd


def _glove_grasp(sk: ManoSkeleton, grasp: str, R: float, flex_open: np.ndarray, abd_open: np.ndarray,
                 pen: np.ndarray) -> dict:
    """Grasp synthesis around a sphere of radius ``R`` in the hand (joint-0) frame.

    power: centre in front of the palm; every finger closes along its flexion path until its pad
    penetrates the sphere by ``pen`` (1-D search; fingers that cannot reach curl to their closest
    approach). precision: thumb + index follow the pinch key until the pad gap equals the object
    diameter; the sphere sits between the pads and the other fingers close onto it if they reach.
    Returns ``flex[15]``, ``abd[5]``, ``center[3]``, per-finger closure ``s[5]``.
    """
    s = np.linspace(0.0, 1.0, 81)
    flex_full = np.zeros(15)
    abd_full = np.array([0.0, 0.02, 0.0, -0.02, -0.05])
    for f in FINGERS:
        js = [j - 1 for j in FINGER_JOINTS[f]]
        flex_full[js] = (1.1, 0.35, 0.5) if f == "thumb" else (1.2, 1.35, 1.0)
    if grasp == "precision":
        kf, ka = _glove_pinch_keys(sk)["index"]
        tj = [j - 1 for j in FINGER_JOINTS["thumb"] + FINGER_JOINTS["index"]]
        flex_full[tj] = kf[tj]
        abd_full[[0, 1]] = ka[[0, 1]]
    flex = flex_open + s[:, None] * (flex_full - flex_open)
    abd = abd_open + s[:, None] * (abd_full - abd_open)
    with torch.no_grad():
        pads, _ = _glove_pads(sk, torch.as_tensor(flex), torch.as_tensor(abd))
    pads = pads.numpy()                                                         # [S,5,3]
    if grasp == "precision":
        gap = np.linalg.norm(pads[:, 0] - pads[:, 1], axis=1)
        target = 2.0 * R - pen[0] - pen[1]
        k = int(np.argmax(gap <= target)) if (gap <= target).any() else int(np.argmin(gap))
        u = pads[k, 1] - pads[k, 0]
        u = u / max(np.linalg.norm(u), 1e-9)
        center = pads[k, 0] + (R - pen[0]) * u
        closure = np.full(5, np.nan)
        closure[[0, 1]] = s[k]
        search = (2, 3, 4)
        candidates = [center]
    else:
        base = np.array([-0.074, -(R + 0.003), 0.0])
        candidates = [base + np.array([dx, dy, dz]) for dy in (0.0, 0.004, 0.008)
                      for dx, dz in ((0.0, 0.0), (0.004, 0.006), (-0.004, 0.006))]
        search = (0, 1, 2, 3, 4)
        closure = np.full(5, np.nan)
    best = None
    for c in candidates:
        d = np.linalg.norm(pads - c, axis=-1)                                      # [S,5]
        cl = closure.copy()
        for fi in search:
            hit = d[:, fi] <= R - pen[fi]
            cl[fi] = s[int(np.argmax(hit))] if hit.any() else -s[int(np.argmin(d[:, fi]))]
        ok = cl[0] >= 0 and cl[1] >= 0
        if best is None or ok:
            best = (c, cl)
        if ok:
            break
    center, cl = best
    reach = np.abs(cl)
    reach = np.where(cl < 0, np.minimum(reach, 0.8), reach)                  # no contact: curl, stop short
    fl, ab = flex_open.copy(), abd_open.copy()
    for fi, f in enumerate(FINGERS):
        js = [j - 1 for j in FINGER_JOINTS[f]]
        fl[js] = flex_open[js] + reach[fi] * (flex_full[js] - flex_open[js])
        ab[fi] = abd_open[fi] + reach[fi] * (abd_full[fi] - abd_open[fi])
    return {"flex": fl, "abd": ab, "center": np.asarray(center, dtype=np.float64), "s": cl}


# ── robot hand ─────────────────────────────────────────────────────────────
_ROBOT_FINGER_BASE = {"index": (0.0, 0.033, 0.095), "middle": (0.0, 0.011, 0.098),
                      "ring": (0.0, -0.011, 0.095), "pinky": (0.0, -0.033, 0.088)}
_ROBOT_FINGER_LEN = {"index": (0.045, 0.027, 0.024), "middle": (0.048, 0.030, 0.025),
                     "ring": (0.045, 0.028, 0.024), "pinky": (0.038, 0.022, 0.021)}
_ROBOT_THUMB_BASE = (0.012, 0.042, 0.030)
_ROBOT_THUMB_RPY = (-0.7, 0.0, 0.0)     # thumb points outward (+y) and up (+z); pads face +x (palm side)
_ROBOT_THUMB_LEN = (0.040, 0.032, 0.026)
_ROBOT_JOINTS: dict[str, tuple[str, ...]] = {
    "thumb": ("thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip"),
    **{f: (f"{f}_mcp", f"{f}_pip", f"{f}_dip") for f in ("index", "middle", "ring", "pinky")},
}
#: actuated joints of :func:`robot_hand_urdf` (file order = ``URDFModel.joint_names``).
ROBOT_JOINT_NAMES: tuple[str, ...] = tuple(j for f in FINGERS for j in _ROBOT_JOINTS[f])
#: capsules for robot self-contact: (link, length along local +z).
_ROBOT_CAPSULES: tuple[tuple[str, float], ...] = (
    ("thumb_metacarpal_link", _ROBOT_THUMB_LEN[0]), ("thumb_proximal_link", _ROBOT_THUMB_LEN[1]),
    ("thumb_distal_link", _ROBOT_THUMB_LEN[2]),
    *[(f"{f}_{seg}_link", _ROBOT_FINGER_LEN[f][i]) for f in ("index", "middle", "ring", "pinky")
      for i, seg in enumerate(("proximal", "middle", "distal"))],
)
_ROBOT_DISTAL = {"thumb": _ROBOT_THUMB_LEN[2], **{f: v[2] for f, v in _ROBOT_FINGER_LEN.items()}}

#: Pinch keys for the synthetic robot, found offline with :func:`_solve_robot_pinch` (300 steps):
#: (thumb 4 + finger 3 joint angles), pad distance (m; the middle pinch is limit-bound but its pads
#: still lie within the self-contact radius). Verified at run time like the glove keys.
_ROBOT_PINCH_DEFAULT: dict[str, tuple[tuple[float, ...], float]] = {
    "index": ((0.8172, 0.4503, 0.1517, -0.2000, 0.7639, 1.0899, 1.4973), 0.00055),
    "middle": ((1.0000, 0.2478, 0.0014, 0.0882, 0.4219, 1.8000, 1.4928), 0.00605),
}


def robot_hand_urdf() -> str:
    """URDF of the synthetic 16-DoF robot hand (``base_link`` → fixed → ``palm_link``; per finger
    ``<f>_mcp/pip/dip`` → ``<f>_proximal/middle/distal_link``; thumb ``thumb_cmc_abd/cmc_flex/mcp/ip``
    → ``thumb_base/metacarpal/proximal/distal_link``). Palm plane x = 0, fingers along +z, pads and
    palm face +x, flexion is a positive rotation about the local +y axis — matching
    ``common/layouts/robot_hand_template.yaml``."""
    def joint(name, parent, child, xyz, rpy=(0.0, 0.0, 0.0), axis=(0, 1, 0), lim=(-0.3, 1.6),
              typ="revolute"):
        s = (f'  <joint name="{name}" type="{typ}">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
             f'    <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" '
             f'rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>\n')
        if typ != "fixed":
            s += (f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n'
                  f'    <limit lower="{lim[0]}" upper="{lim[1]}" effort="2.0" velocity="5.0"/>\n')
        return s + "  </joint>\n"

    links = ["base_link", "palm_link", "thumb_base_link", "thumb_metacarpal_link", "thumb_proximal_link",
             "thumb_distal_link"]
    js = joint("palm_mount", "base_link", "palm_link", (0, 0, 0), typ="fixed")
    js += joint("thumb_cmc_abd", "palm_link", "thumb_base_link", _ROBOT_THUMB_BASE, _ROBOT_THUMB_RPY,
                (1, 0, 0), (-0.4, 1.0))
    js += joint("thumb_cmc_flex", "thumb_base_link", "thumb_metacarpal_link", (0, 0, 0), lim=(-0.3, 1.4))
    js += joint("thumb_mcp", "thumb_metacarpal_link", "thumb_proximal_link", (0, 0, _ROBOT_THUMB_LEN[0]),
                lim=(-0.2, 1.2))
    js += joint("thumb_ip", "thumb_proximal_link", "thumb_distal_link", (0, 0, _ROBOT_THUMB_LEN[1]),
                lim=(-0.2, 1.4))
    for f in ("index", "middle", "ring", "pinky"):
        l1, l2, _ = _ROBOT_FINGER_LEN[f]
        links += [f"{f}_proximal_link", f"{f}_middle_link", f"{f}_distal_link"]
        js += joint(f"{f}_mcp", "palm_link", f"{f}_proximal_link", _ROBOT_FINGER_BASE[f], lim=(-0.3, 1.6))
        js += joint(f"{f}_pip", f"{f}_proximal_link", f"{f}_middle_link", (0, 0, l1), lim=(0.0, 1.8))
        js += joint(f"{f}_dip", f"{f}_middle_link", f"{f}_distal_link", (0, 0, l2), lim=(0.0, 1.5))
    body = "".join(f'  <link name="{ln}"/>\n' for ln in links)
    return ('<?xml version="1.0"?>\n<!-- synthetic 16-DoF hand (robot_skin.datasets.synthetic) -->\n'
            f'<robot name="synthetic_hand">\n{body}{js}</robot>\n')


def _robot_model() -> URDFModel:
    if _CACHE.robot_model is None:
        _CACHE.robot_model = URDFModel.from_string(robot_hand_urdf())
        assert _CACHE.robot_model.joint_names == ROBOT_JOINT_NAMES
    return _CACHE.robot_model


def _robot_q(thumb=(0.0, 0.0, 0.0, 0.0), fingers=(0.0, 0.0, 0.0)) -> np.ndarray:
    q = np.zeros(len(ROBOT_JOINT_NAMES))
    idx = {n: i for i, n in enumerate(ROBOT_JOINT_NAMES)}
    q[[idx[n] for n in _ROBOT_JOINTS["thumb"]]] = thumb
    for f in ("index", "middle", "ring", "pinky"):
        q[[idx[n] for n in _ROBOT_JOINTS[f]]] = fingers
    return q


def _robot_joint_index(finger: str) -> list[int]:
    return [ROBOT_JOINT_NAMES.index(n) for n in _ROBOT_JOINTS[finger]]


def _robot_pads(model: URDFModel, q, fingers: Sequence[str] = FINGERS):
    """Pad points (on the distal-link axis, 60 % along) and pad normals (+x) per finger (torch)."""
    fk = model.fk(q, links=[f"{f}_distal_link" for f in fingers])
    pads, nrm = [], []
    for f in fingers:
        T = fk[f"{f}_distal_link"]
        pads.append(T[..., :3, 3] + T[..., :3, 2] * 0.6 * _ROBOT_DISTAL[f])
        nrm.append(T[..., :3, 0])
    return torch.stack(pads, -2), torch.stack(nrm, -2)


def _solve_robot_pinch(model: URDFModel, finger: str, steps: int = 300) -> np.ndarray:
    """Thumb + finger joint angles bringing the robot fingertip pads together (clamped Adam)."""
    q0 = torch.as_tensor(_robot_q((0.3, 0.6, 0.4, 0.4), (0.5, 0.6, 0.4)))
    mask = torch.zeros(len(ROBOT_JOINT_NAMES), dtype=torch.float64)
    mask[_robot_joint_index("thumb") + _robot_joint_index(finger)] = 1.0
    lo, hi = torch.as_tensor(model.lower), torch.as_tensor(model.upper)
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=0.03)
    for _ in range(steps):
        qq = q0 + mask * (q - q0)
        pads, nrm = _robot_pads(model, qq, ("thumb", finger))
        loss = (pads[0] - pads[1]).norm() + 0.002 * (nrm[0] * nrm[1]).sum() + 1e-4 * ((qq - q0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.copy_(torch.maximum(torch.minimum(q, hi), lo))
    qq = (q0 + mask * (q - q0)).detach().numpy()
    return qq[_robot_joint_index("thumb") + _robot_joint_index(finger)]


def _robot_pinch_keys(model: URDFModel) -> dict[str, np.ndarray]:
    """finger → pinch key (thumb 4 + finger 3 joint angles)."""
    if not _CACHE.robot_pinch:
        for f in ("index", "middle"):
            v, d_ref = _ROBOT_PINCH_DEFAULT[f]
            v = np.asarray(v, dtype=np.float64)
            q = _robot_q()
            q[_robot_joint_index("thumb") + _robot_joint_index(f)] = v
            with torch.no_grad():
                pads, _ = _robot_pads(model, torch.as_tensor(q), ("thumb", f))
            ok = abs(float((pads[0] - pads[1]).norm()) - d_ref) < _KEY_TOL
            _CACHE.robot_pinch[f] = v if ok else _solve_robot_pinch(model, f)
    return _CACHE.robot_pinch


def _robot_grasp(model: URDFModel, grasp: str, R: float, q_open: np.ndarray, pen: np.ndarray, inset: float) -> dict:
    """Robot grasp synthesis (hand base frame): thumb + index follow the pinch key until the pad
    axes are one object diameter (+ pad insets) apart; the sphere sits between them; the other
    fingers (middle only for ``precision``) close onto it by 1-D search along their flexion path."""
    s = np.linspace(0.0, 1.0, 81)
    q_full = _robot_q((0.3, 0.6, 0.4, 0.4), (1.3, 1.4, 1.1))
    ti, ii = _robot_joint_index("thumb"), _robot_joint_index("index")
    q_full[ti + ii] = _robot_pinch_keys(model)["index"]
    Q = q_open + s[:, None] * (q_full - q_open)
    with torch.no_grad():
        pads = _robot_pads(model, torch.as_tensor(Q))[0].numpy()                 # [S,5,3]
    gap = np.linalg.norm(pads[:, 0] - pads[:, 1], axis=1)
    Re = R + inset
    target = 2.0 * Re - pen[0] - pen[1]
    k = int(np.argmax(gap <= target)) if (gap <= target).any() else int(np.argmin(gap))
    u = pads[k, 1] - pads[k, 0]
    u /= max(np.linalg.norm(u), 1e-9)
    center = pads[k, 0] + (Re - pen[0]) * u
    cl = np.full(5, np.nan)
    cl[[0, 1]] = s[k]
    d = np.linalg.norm(pads - center, axis=-1)
    for fi in ((2,) if grasp == "precision" else (2, 3, 4)):
        hit = d[:, fi] <= Re - pen[fi]
        cl[fi] = s[int(np.argmax(hit))] if hit.any() else -min(0.5, s[int(np.argmin(d[:, fi]))])
    reach = np.nan_to_num(np.abs(cl), nan=0.25)
    q = q_open.copy()
    for fi, f in enumerate(FINGERS):
        j = _robot_joint_index(f)
        q[j] = q_open[j] + reach[fi] * (q_full[j] - q_open[j])
    return {"q": q, "center": center, "s": cl}


# ── glove motion (ground-truth kinematics) ─────────────────────────────────
class _GloveMotion:
    """Ground-truth glove kinematics of one session: anatomical controls at arbitrary times."""

    def __init__(self, blocks: list[Block], rng: np.random.Generator, style: dict, task: dict | None,
                 p: SynthParams):
        self.blocks, self.sk, self.task, self.p = blocks, _skeleton(), task, p
        self.amp, self.speed = style["amp"], style["speed"]
        fn = np.zeros(15)
        for f in FINGERS:
            js = [j - 1 for j in FINGER_JOINTS[f]]
            fn[js] = (np.array([0.12, 0.08, 0.08]) if f == "thumb" else np.array([0.18, 0.22, 0.12])) \
                + rng.uniform(-0.03, 0.03, 3)
        self.flex_n = fn
        self.abd_n = np.array([0.15, 0.06, 0.0, -0.06, -0.12]) * rng.uniform(0.8, 1.2)
        self.go_c = np.array([math.pi / 2, 0.0, 0.0])                   # flat hand, palm down on the table
        self.go_n = _compose(self.go_c, _random_rotation_aa(rng, 0.25))
        self.wp_n = np.array([0.0, 0.0, 0.10]) + rng.uniform(-0.02, 0.02, 3)
        self.wp_c = np.array([self.wp_n[0], self.wp_n[1], 0.03])
        self.ref_pos = self.wp_n.copy()
        self._bi = {id(b): i for i, b in enumerate(blocks)}
        self.bp = [self._block_params(b, rng) for b in blocks]
        self.handlers = {i: getattr(self, f"_g_{b.gen}") for i, b in enumerate(blocks)}
        if task is not None:
            self._init_task(rng)

    # per-block random parameters (drawn in block order → deterministic)
    def _block_params(self, b: Block, rng: np.random.Generator) -> dict:
        if b.gen == "sweep":
            fast = b.info.get("speed") == "fast"
            f0 = (1.8 if fast else 0.6) * self.speed
            n = max(1, int(round(b.dur * f0)))
            amp = rng.uniform(0.35, 0.55, 5) if fast else rng.uniform(0.55, 0.85, 5)
            amp[0] = rng.uniform(0.05, 0.15)
            return {"f": n / b.dur, "amp": amp * self.amp, "phase": -rng.uniform(0.0, 0.4, 5),
                    "dist": np.array([0.8, 1.0, 0.7]) + rng.uniform(-0.1, 0.1, 3)}
        if b.gen == "wrist":
            return {"f": max(1, round(b.dur * 0.8 * self.speed)) / b.dur, "amp": rng.uniform(0.3, 0.6, 3),
                    "phase": rng.uniform(0, 2 * np.pi, 3), "wp_amp": rng.uniform(0.01, 0.03, 3)}
        if b.gen == "free":
            return {"f": rng.uniform(0.3, 1.5, (3, 5)) * self.speed, "a": rng.normal(0, 1.0, (3, 5)),
                    "ph": rng.uniform(0, 2 * np.pi, (3, 5)), "go": rng.normal(0, 0.15, (2, 3)),
                    "gof": rng.uniform(0.3, 1.0, 3), "wp": rng.normal(0, 0.015, (2, 3))}
        if b.gen in ("pinch", "fist"):
            n = max(1, int(round(b.dur / 1.0)))
            return {"n": n, "cmax": rng.uniform(1.0, 1.05, n), "squeeze": rng.uniform(0.01, 0.04, n)}
        if b.gen == "static_end":
            return {"ph": rng.uniform(0, 2 * np.pi, 15)}
        return {}

    # ── evaluation ────────────────────────────────────────────────────────
    def controls(self, t) -> dict[str, np.ndarray]:
        return _dispatch(self.blocks, t, self.handlers, {"flex": (15,), "abd": (5,), "go": (3,), "wp": (3,)})

    def state(self, t) -> dict[str, np.ndarray]:
        c = self.controls(t)
        with torch.no_grad():
            c["fp"] = self.sk.flexion_pose(torch.as_tensor(c["flex"]), torch.as_tensor(c["abd"])).numpy()
        return c

    def _out(self, n, flex=None, abd=None, go=None, wp=None):
        return {"flex": np.broadcast_to(self.flex_n if flex is None else flex, (n, 15)),
                "abd": np.broadcast_to(self.abd_n if abd is None else abd, (n, 5)),
                "go": np.broadcast_to(self.go_n if go is None else go, (n, 3)),
                "wp": np.broadcast_to(self.wp_n if wp is None else wp, (n, 3))}

    # ── D1 generators ─────────────────────────────────────────────────────
    def _g_calib_hold(self, b: Block, tau):
        """IMU calibration: flat hand (zero flexion/abduction), palm down, still."""
        n = len(tau)
        return self._out(n, np.zeros(15), np.zeros(5), self.go_c, self.wp_c)

    def _g_to_neutral(self, b: Block, tau):
        """From the calibration pose to the relaxed rest pose (70 %), then hold it."""
        x = _mj(tau / (0.7 * b.dur))[:, None]
        go = _compose(self.go_c, x * _rel(self.go_c, self.go_n))
        return self._out(len(tau), x * self.flex_n, x * self.abd_n, go, self.wp_c + x * (self.wp_n - self.wp_c))

    def _g_static_end(self, b: Block, tau):
        tremor = 0.003 * np.sin(2 * np.pi * 9.0 * tau[:, None] + self.bp[self._bi[id(b)]]["ph"])
        return self._out(len(tau), self.flex_n + tremor * _window(tau, b.dur)[:, None])

    def _g_sweep(self, b: Block, tau):
        q = self.bp[self._bi[id(b)]]
        w = _window(tau, b.dur)[:, None]
        s = 0.5 * (1.0 - np.cos(2 * np.pi * q["f"] * tau[:, None] + 2 * np.pi * q["f"] * q["phase"]))   # [n,5]
        dev = q["amp"] * s * w                                                                         # [n,5]
        flex = np.repeat(self.flex_n[None], len(tau), 0)
        for fi, f in enumerate(FINGERS):
            js = [j - 1 for j in FINGER_JOINTS[f]]
            flex[:, js] += dev[:, fi:fi + 1] * q["dist"]
        abd = self.abd_n * (1.0 - 0.5 * np.clip(dev.mean(1, keepdims=True) / 0.8, 0, 1))
        wp = self.wp_n + 0.005 * np.sin(2 * np.pi * 0.5 * tau)[:, None] * w
        return self._out(len(tau), flex, abd, None, wp)

    def _g_wrist(self, b: Block, tau):
        q = self.bp[self._bi[id(b)]]
        w = _window(tau, b.dur)[:, None]
        fr = np.array([1.0, 1.0, 0.5]) * q["f"]
        delta = q["amp"] * np.sin(2 * np.pi * fr * tau[:, None] + q["phase"]) * w
        delta -= q["amp"] * np.sin(q["phase"]) * w       # start/end exactly at neutral
        flex = self.flex_n + 0.08 * np.sin(2 * np.pi * 0.7 * tau)[:, None] * w
        return self._out(len(tau), flex, None, _compose(self.go_n, delta),
                         self.wp_n + q["wp_amp"] * np.sin(2 * np.pi * q["f"] * tau)[:, None] * w)

    def _g_free(self, b: Block, tau):
        q = self.bp[self._bi[id(b)]]
        w = _window(tau, b.dur, 0.3)[:, None]
        z = (q["a"][None] * np.sin(2 * np.pi * q["f"][None] * tau[:, None, None] + q["ph"][None])).sum(1)  # [n,5]
        lvl = 0.5 + 0.5 * np.tanh(z / 1.5)
        amax = np.array([0.15, 0.7, 0.7, 0.7, 0.7]) * self.amp
        dev = amax * lvl * w
        flex = np.repeat(self.flex_n[None], len(tau), 0)
        for fi, f in enumerate(FINGERS):
            js = [j - 1 for j in FINGER_JOINTS[f]]
            flex[:, js] += dev[:, fi:fi + 1] * np.array([0.8, 1.0, 0.7])
        abd = self.abd_n + 0.08 * np.tanh(z / 2.0) * w
        s1 = np.sin(2 * np.pi * q["gof"][None] * tau[:, None])
        go = _compose(self.go_n, (q["go"][0] * s1 + q["go"][1] * s1 ** 2) * w)
        wp = self.wp_n + (q["wp"][0] * s1 + q["wp"][1] * (1 - np.cos(2 * np.pi * q["gof"] * tau[:, None]))) * w
        return self._out(len(tau), flex, abd, go, wp)

    def _closure(self, b: Block, tau):
        """Repeated approach–hold(+squeeze)–release profile ∈ [0, ~1.05]."""
        q = self.bp[self._bi[id(b)]]
        per = b.dur / q["n"]
        k = np.minimum((tau // per).astype(int), q["n"] - 1)
        x = (tau - k * per) / per
        c = _mj(x / 0.35) - _mj((x - 0.65) / 0.35)
        return (q["cmax"][k] * (c + q["squeeze"][k] * _bump((x - 0.35) / 0.3)))[:, None]

    def _g_pinch(self, b: Block, tau):
        f = b.info["finger"]
        kf, ka = _glove_pinch_keys(self.sk)[f]
        c = self._closure(b, tau)
        js = [j - 1 for j in FINGER_JOINTS["thumb"] + FINGER_JOINTS[f]]
        flex = np.repeat(self.flex_n[None], len(tau), 0)
        flex[:, js] += c * (kf[js] - self.flex_n[js])
        abd = np.repeat(self.abd_n[None], len(tau), 0)
        ai = [0, FINGERS.index(f)]
        abd[:, ai] += c * (ka[ai] - self.abd_n[ai])
        return self._out(len(tau), flex, abd)

    def _g_fist(self, b: Block, tau):
        key = np.zeros(15)
        for f in FINGERS:
            js = [j - 1 for j in FINGER_JOINTS[f]]
            key[js] = (0.7, 0.35, 0.45) if f == "thumb" else (1.25, 1.45, 1.1)
        c = self._closure(b, tau)
        abd_key = np.array([0.35, 0.0, 0.0, 0.0, 0.0])
        return self._out(len(tau), self.flex_n + c * (key - self.flex_n), self.abd_n + c * (abd_key - self.abd_n))

    # ── D2 (task) ─────────────────────────────────────────────────────────
    def _init_task(self, rng: np.random.Generator) -> None:
        T = self.task
        R = T["radius"]
        self.flex_open, self.abd_open = _glove_open_pose(self.flex_n)
        pen = rng.uniform(0.002, 0.004, 5)
        g = _glove_grasp(self.sk, T["grasp"], R, self.flex_open, self.abd_open, pen)
        self.flex_g, self.abd_g, self.c_h = g["flex"], g["abd"], g["center"]
        self.grasp_closure = g["s"]
        yaw = rng.uniform(-0.5, 0.5)
        self.go_g = _aa(_R([0, 0, yaw]) @ _R(self.go_c))                        # palm down, rotated about z
        obj0 = np.array([rng.uniform(0.25, 0.40), rng.uniform(-0.10, 0.10), R])   # resting on the table
        Rg = _R(self.go_g)
        self.wp_g = obj0 - Rg @ self.c_h
        self.wp_pre = self.wp_g + Rg @ np.array([0.0, 0.03, 0.0])               # 3 cm dorsal (above)
        self.go_s = _aa(_R([0, 0, rng.uniform(-0.3, 0.3)]) @ _R(self.go_c))
        self.wp_s = np.array([0.05, 0.0, 0.12]) + rng.uniform(-0.03, 0.03, 3)
        self.go_n, self.wp_n = self.go_s, self.wp_s                             # static phases use the start pose
        self.ref_pos = obj0.copy()
        self.obj0, self.qobj0 = obj0, _quat(_R([0, 0, rng.uniform(-np.pi, np.pi)]))
        m = T["manip"]
        self.manip = {"type": m, "move": np.array([rng.uniform(-0.12, 0.12), rng.uniform(0.10, 0.20), 0.0]),
                      "lift": rng.uniform(0.05, 0.09), "angle": rng.uniform(0.8, 1.3) * rng.choice([-1, 1]),
                      "osc": np.array([rng.uniform(0.04, 0.08), rng.uniform(-0.03, 0.03), 0.0]),
                      "osc_f": rng.uniform(1.2, 2.0), "twist": rng.uniform(0.4, 0.7),
                      "gait_ph": rng.uniform(0, 2 * np.pi, 5), "squeeze": rng.uniform(0.03, 0.08),
                      "spin": rng.uniform(0.5, 1.2)}
        self.retreat = np.array([rng.uniform(-0.12, -0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.08, 0.14)])
        man = next(b for b in self.blocks if b.name == "manipulate")
        self.success = bool(T["success"])
        self.t_slip = None if self.success else man.t0 + rng.uniform(0.3, 0.6) * man.dur
        # a dropped object tumbles ~15 cm sideways (hand radial/ulnar direction) so the hand,
        # returning to the grasp spot, does not touch it again
        side = Rg @ np.array([0.0, 0.0, rng.choice([-1.0, 1.0])])
        self.slip_slide = 0.15 * side / max(np.linalg.norm(side[:2]), 1e-9) * np.array([1.0, 1.0, 0.0])
        self.man_end = self._manip_pose(man, np.array([man.dur]))
        go_e, wp_e = self.man_end
        self.wp_lift = wp_e[0] + _R(go_e[0]) @ np.array([0.0, 0.03, 0.0])      # release: lift the palm off
        self._man_block = man

    def _manip_pose(self, b: Block, tau):
        """Wrist (go, wp) during manipulate; returns to (go_g, wp_g) or ends at the place pose."""
        M, n = self.manip, len(tau)
        x = tau / b.dur
        wp = np.repeat(self.wp_g[None], n, 0)
        go = np.repeat(self.go_g[None], n, 0)
        typ = M["type"]
        if typ == "lift_move":
            wp = wp + _mj(x)[:, None] * M["move"] + (M["lift"] * _bump(x))[:, None] * np.array([0, 0, 1.0])
        elif typ == "rotate":
            wp = wp + (M["lift"] * _bump(x))[:, None] * np.array([0, 0, 1.0])
            ang = M["angle"] * _bump(x)
            go = _aa(_R(ang[:, None] * np.array([1.0, 0.0, 0.0])) @ _R(self.go_g))   # pour: about the fingers' axis
        elif typ == "oscillate":
            w = _window(tau, b.dur, 0.2)[:, None]
            wp = wp + M["osc"] * np.sin(2 * np.pi * M["osc_f"] * tau)[:, None] * w
        elif typ == "twist":
            ang = M["twist"] * np.sin(2 * np.pi * x) * _window(tau, b.dur, 0.2)
            Rz = _R(ang[:, None] * np.array([0.0, 0.0, 1.0]))
            c0 = self.wp_g + _R(self.go_g) @ self.c_h                          # twist about the object's vertical axis
            go = _aa(Rz @ _R(self.go_g))
            wp = c0 - np.einsum("nij,j->ni", _R(go), self.c_h)
        return go, wp

    def _grip(self, tau, dur, extra):
        """Grasp flex/abd with a squeeze modulation (+ per-finger gait for in-hand rotation)."""
        w = _window(tau, dur, 0.2)[:, None]
        sq = self.manip["squeeze"] * (0.5 - 0.5 * np.cos(2 * np.pi * 1.3 * tau))[:, None] * w
        flex = self.flex_g + (sq + extra) * (self.flex_g - self.flex_open)
        return flex

    def _g_reach(self, b: Block, tau):
        """Pre-shape and orient while travelling above the object, then descend vertically to the
        pre-grasp pose (the lift keeps the opening fingers — the abducted thumb in particular —
        from sweeping through the object on a straight-line approach)."""
        xr = tau / b.dur
        x = _mj(xr)[:, None]
        lift = (REACH_LIFT_M * _mj(xr / 0.4) * (1.0 - _mj((xr - 0.6) / 0.4)))[:, None] * np.array([0.0, 0.0, 1.0])
        return self._out(len(tau), self.flex_n + x * (self.flex_open - self.flex_n),
                         self.abd_n + x * (self.abd_open - self.abd_n),
                         _compose(self.go_s, x * _rel(self.go_s, self.go_g)),
                         self.wp_s + x * (self.wp_pre - self.wp_s) + lift)

    def _g_grasp(self, b: Block, tau):
        xw = _mj(tau / (0.6 * b.dur))[:, None]
        xf = _mj((tau - 0.25 * b.dur) / (0.75 * b.dur))[:, None]
        return self._out(len(tau), self.flex_open + xf * (self.flex_g - self.flex_open),
                         self.abd_open + xf * (self.abd_g - self.abd_open), self.go_g,
                         self.wp_pre + xw * (self.wp_g - self.wp_pre))

    def _g_manipulate(self, b: Block, tau):
        go, wp = self._manip_pose(b, tau)
        extra = np.zeros((len(tau), 15))
        if self.manip["type"] == "finger_gait":
            w = _window(tau, b.dur, 0.2)
            for fi, f in enumerate(FINGERS[1:], start=1):
                js = [j - 1 for j in FINGER_JOINTS[f]]
                extra[:, js] = (0.06 * np.sin(2 * np.pi * 1.5 * tau + self.manip["gait_ph"][fi]) * w)[:, None]
        return self._out(len(tau), self._grip(tau, b.dur, extra), self.abd_g, go, wp)

    def _g_release(self, b: Block, tau):
        """Fingers open; in the second half the palm lifts off the placed object (3 cm dorsal)."""
        x = _mj(tau / b.dur)[:, None]
        xl = _mj((tau - 0.4 * b.dur) / (0.6 * b.dur))[:, None]
        go, wp = self.man_end
        return self._out(len(tau), self.flex_g + x * (self.flex_open - self.flex_g),
                         self.abd_g + x * (self.abd_open - self.abd_g), go[0], wp[0] + xl * (self.wp_lift - wp[0]))

    def _g_retreat(self, b: Block, tau):
        x = _mj(tau / b.dur)[:, None]
        go, _ = self.man_end
        return self._out(len(tau), self.flex_open + x * (self.flex_n - self.flex_open),
                         self.abd_open + x * (self.abd_n - self.abd_open),
                         _compose(go[0], x * _rel(go[0], self.go_s)), self.wp_lift + x * self.retreat)

    def _g_task_start(self, b: Block, tau):
        """D2 ``baseline``: flat hand (zero artefact) held for 65 %, then to the relaxed start pose —
        like the robot's ``static_start``. D1 and D2 thus share the same ΔS reference (the first
        no_contact data is artefact-free in both), as the real protocol intends with its common
        rest pose; a baseline model trained on D1 then leaves no constant offset on D2."""
        hold = b.info.get("hold", 0.65 * b.dur)
        x = _mj((tau - hold) / max(b.dur - hold, 1e-6))[:, None]
        return self._out(len(tau), x * self.flex_n, x * self.abd_n, self.go_s, self.wp_s)

    def _g_task_end(self, b: Block, tau):
        return self._out(len(tau), self.flex_n, self.abd_n, self.go_s, self.wp_lift + self.retreat)

    def object_pose(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Object centre ``pos[T,3]`` and ``quat[T,4]`` (world). Static until grasped, rigidly
        attached during manipulate (until a slip on failure → ballistic fall onto the table),
        left at the place pose afterwards."""
        t = np.asarray(t, dtype=np.float64)
        pos = np.repeat(self.obj0[None], len(t), 0)
        quat = np.repeat(self.qobj0[None], len(t), 0)
        man = self._man_block
        t_att = man.t0
        t_det = man.t1 if self.t_slip is None else self.t_slip

        def attached(tt):
            st = self.controls(tt)
            Rh = _R(st["go"])
            c = st["wp"] + np.einsum("nij,j->ni", Rh, self.c_h)
            Rrel = Rh @ _R(self.go_g).T
            if self.manip["type"] == "finger_gait":         # in-hand rotation about the vertical
                spin = self.manip["spin"] * _mj((tt - man.t0) / man.dur)
                Rrel = _R(spin[:, None] * np.array([0.0, 0.0, 1.0])) @ Rrel
            return c, _qmul(_quat(Rrel), np.broadcast_to(self.qobj0, (len(tt), 4)))

        m = (t >= t_att) & (t < t_det)
        if m.any():
            pos[m], quat[m] = attached(t[m])
        c_det, q_det = attached(np.array([t_det]))
        after = t >= t_det
        if after.any():
            pos[after], quat[after] = c_det[0], q_det[0]
            if self.t_slip is not None:
                dt = t[after] - t_det
                z = c_det[0, 2] + 0.5 * GRAVITY[2] * dt ** 2
                pos[after, :2] = c_det[0, :2] + _mj(dt / 0.4)[:, None] * self.slip_slide[:2]
                pos[after, 2] = np.maximum(z, self.task["radius"])
        return pos, _hemisphere(quat)


# ── robot motion ───────────────────────────────────────────────────────────
class _RobotMotion:
    """Ground-truth robot joint trajectories (and object pose in the hand base frame)."""

    def __init__(self, blocks: list[Block], rng: np.random.Generator, style: dict, task: dict | None,
                 p: SynthParams):
        self.blocks, self.model, self.task, self.p = blocks, _robot_model(), task, p
        self.amp, self.speed = style["amp"], style["speed"]
        D = len(ROBOT_JOINT_NAMES)
        self.q_n = _robot_q((0.2, 0.2, 0.1, 0.1), (0.1, 0.15, 0.1)) + rng.uniform(-0.03, 0.03, D)
        self.ref_pos = np.array([0.0, 0.0, 0.10])
        self._bi = {id(b): i for i, b in enumerate(blocks)}
        self.handlers = {i: getattr(self, f"_g_{b.gen}") for i, b in enumerate(blocks)}
        self.bp = [self._block_params(b, rng) for b in blocks]
        self._pinch_cache: dict[str, np.ndarray] = {}
        if task is not None:
            self._init_task(rng)

    def _block_params(self, b: Block, rng: np.random.Generator) -> dict:
        D = len(ROBOT_JOINT_NAMES)
        if b.gen == "sweep":
            fast = b.info.get("speed") == "fast"
            f0 = (1.8 if fast else 0.6) * self.speed
            n = max(1, int(round(b.dur * f0)))
            amp = np.where(np.arange(D) < 4, rng.uniform(0.05, 0.2, D),
                           rng.uniform(0.35, 0.55, D) if fast else rng.uniform(0.55, 0.9, D))
            return {"f": n / b.dur, "amp": amp * self.amp, "phase": -rng.uniform(0.0, 0.4, D)}
        if b.gen == "free":
            return {"f": rng.uniform(0.3, 1.5, (3, D)) * self.speed, "a": rng.normal(0, 1.0, (3, D)),
                    "ph": rng.uniform(0, 2 * np.pi, (3, D))}
        if b.gen in ("pinch", "fist"):
            n = max(1, int(round(b.dur / 1.0)))
            return {"n": n, "cmax": rng.uniform(1.0, 1.05, n), "squeeze": rng.uniform(0.01, 0.04, n)}
        if b.gen in ("static_end", "task_end"):
            return {"ph": rng.uniform(0, 2 * np.pi, D)}
        return {}

    def controls(self, t) -> dict[str, np.ndarray]:
        return _dispatch(self.blocks, t, self.handlers, {"q": (len(ROBOT_JOINT_NAMES),)})

    state = controls

    def _out(self, n, q=None):
        return {"q": self.model.clamp(np.broadcast_to(self.q_n if q is None else q, (n, len(ROBOT_JOINT_NAMES))))}

    def _g_static_start(self, b: Block, tau):
        hold = b.info.get("hold", 0.65 * b.dur)
        x = _mj((tau - hold) / max(b.dur - hold, 1e-6))[:, None]
        return self._out(len(tau), x * self.q_n)

    def _g_static_end(self, b: Block, tau):
        tremor = 0.002 * np.sin(2 * np.pi * 9.0 * tau[:, None] + self.bp[self._bi[id(b)]]["ph"])
        return self._out(len(tau), self.q_n + tremor * _window(tau, b.dur)[:, None])

    def _g_sweep(self, b: Block, tau):
        q = self.bp[self._bi[id(b)]]
        s = 0.5 * (1.0 - np.cos(2 * np.pi * q["f"] * tau[:, None] + 2 * np.pi * q["f"] * q["phase"]))
        return self._out(len(tau), self.q_n + q["amp"] * s * _window(tau, b.dur)[:, None])

    def _g_free(self, b: Block, tau):
        q = self.bp[self._bi[id(b)]]
        z = (q["a"][None] * np.sin(2 * np.pi * q["f"][None] * tau[:, None, None] + q["ph"][None])).sum(1)
        amax = np.where(np.arange(len(ROBOT_JOINT_NAMES)) < 4, 0.2, 0.7) * self.amp
        return self._out(len(tau), self.q_n + amax * (0.5 + 0.5 * np.tanh(z / 1.5)) * _window(tau, b.dur, 0.3)[:, None])

    _closure = _GloveMotion._closure
    _g_task_start = _g_static_start            # D2: home (q = 0) → relaxed pose
    _g_task_end = _g_static_end

    def _pinch_target(self, f: str) -> np.ndarray:
        """Thumb + finger joint angles of the pinch *contact* pose: along the path q_n → pinch key
        (whose on-axis pad points coincide), the first pose where the pads are
        ``2·robot_link_radius − robot_pinch_depth`` apart, i.e. the pad surfaces pressed
        ``robot_pinch_depth`` together instead of the links interpenetrating."""
        if f not in self._pinch_cache:
            j = _robot_joint_index("thumb") + _robot_joint_index(f)
            key = _robot_pinch_keys(self.model)[f]
            s = np.linspace(0.0, 1.0, 101)
            Q = np.repeat(self.q_n[None], len(s), 0)
            Q[:, j] += s[:, None] * (key - self.q_n[j])
            with torch.no_grad():
                pads = _robot_pads(self.model, torch.as_tensor(Q), ("thumb", f))[0].numpy()
            d = np.linalg.norm(pads[:, 0] - pads[:, 1], axis=1)
            hit = d <= 2.0 * self.p.robot_link_radius - self.p.robot_pinch_depth
            k = int(np.argmax(hit)) if hit.any() else len(s) - 1
            self._pinch_cache[f] = self.q_n[j] + s[k] * (key - self.q_n[j])
        return self._pinch_cache[f]

    def _g_pinch(self, b: Block, tau):
        f = b.info["finger"]
        j = _robot_joint_index("thumb") + _robot_joint_index(f)
        key = self._pinch_target(f)
        q = np.repeat(self.q_n[None], len(tau), 0)
        q[:, j] += self._closure(b, tau) * (key - self.q_n[j])
        return self._out(len(tau), q)

    def _g_fist(self, b: Block, tau):
        """Full finger curl (at the joint limits) so the distal links press on the distal palm taxels."""
        key = _robot_q((0.3, 0.9, 0.5, 0.4), (1.6, 1.8, 1.4))
        return self._out(len(tau), self.q_n + self._closure(b, tau) * (key - self.q_n))

    # ── task ──────────────────────────────────────────────────────────────
    def _init_task(self, rng: np.random.Generator) -> None:
        T = self.task
        R = T["radius"]
        self.q_open = _robot_q((0.1, 0.2, 0.05, 0.05), (0.0, 0.05, 0.05))
        g = _robot_grasp(self.model, T["grasp"], R, self.q_open, rng.uniform(0.002, 0.004, 5), self.p.robot_pad_inset)
        self.q_g, self.c_h, self.grasp_closure = g["q"], g["center"], g["s"]
        self.c_pre = self.c_h + np.array([0.03, 0.0, 0.0])
        self.c_far = self.c_h + np.array([rng.uniform(0.12, 0.18), rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.03)])
        self.c_away = self.c_h + np.array([rng.uniform(0.08, 0.14), rng.uniform(-0.03, 0.03), -0.03])
        self.qobj0 = _quat(_R([0, 0, rng.uniform(-np.pi, np.pi)]))
        self.manip = {"type": T["manip"], "squeeze": rng.uniform(0.03, 0.08), "gait_ph": rng.uniform(0, 2 * np.pi, 5),
                      "spin": rng.uniform(0.5, 1.2)}
        man = next(b for b in self.blocks if b.name == "manipulate")
        self._man_block = man
        self.success = bool(T["success"])
        self.t_slip = None if self.success else man.t0 + rng.uniform(0.3, 0.6) * man.dur
        self.slip_dir = np.array([0.6, 0.0, -0.8])

    def _g_reach(self, b, tau):
        x = _mj(tau / b.dur)[:, None]
        return self._out(len(tau), self.q_n + x * (self.q_open - self.q_n))

    def _g_grasp(self, b, tau):
        x = _mj((tau - 0.25 * b.dur) / (0.75 * b.dur))[:, None]
        return self._out(len(tau), self.q_open + x * (self.q_g - self.q_open))

    def _g_manipulate(self, b, tau):
        w = _window(tau, b.dur, 0.2)[:, None]
        sq = self.manip["squeeze"] * (0.5 - 0.5 * np.cos(2 * np.pi * 1.3 * tau))[:, None] * w
        extra = np.zeros((len(tau), len(ROBOT_JOINT_NAMES)))
        if self.manip["type"] == "finger_gait":
            for fi, f in enumerate(FINGERS[1:], start=1):
                extra[:, _robot_joint_index(f)] = (0.06 * np.sin(2 * np.pi * 1.5 * tau + self.manip["gait_ph"][fi])
                                                   * w[:, 0])[:, None]
        return self._out(len(tau), self.q_g + (sq + extra) * (self.q_g - self.q_open))

    def _g_release(self, b, tau):
        x = _mj(tau / b.dur)[:, None]
        return self._out(len(tau), self.q_g + x * (self.q_open - self.q_g))

    def _g_retreat(self, b, tau):
        x = _mj(tau / b.dur)[:, None]
        return self._out(len(tau), self.q_open + x * (self.q_n - self.q_open))

    def object_pose(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Object pose in the hand base (URDF root) frame: approaches during reach/grasp, held during
        manipulate (slips away on failure), left behind while the hand retreats."""
        t = np.asarray(t, dtype=np.float64)
        B = {b.name: b for b in self.blocks}
        pos = np.repeat(self.c_far[None], len(t), 0)
        quat = np.repeat(self.qobj0[None], len(t), 0)
        r, g, man, ret = B["reach"], B["grasp"], B["manipulate"], B["retreat"]
        x = _mj((t - r.t0) / r.dur)[:, None]
        pos = np.where((t >= r.t0)[:, None], self.c_far + x * (self.c_pre - self.c_far), pos)
        x = _mj((t - g.t0) / (0.6 * g.dur))[:, None]
        pos = np.where((t >= g.t0)[:, None], self.c_pre + x * (self.c_h - self.c_pre), pos)
        if self.manip["type"] == "finger_gait":
            ang = self.manip["spin"] * _mj((t - man.t0) / man.dur)
            quat = _qmul(_quat(_R(ang[:, None] * np.array([0.0, 0.0, 1.0]))), quat)
        x = _mj((t - ret.t0) / ret.dur)[:, None]
        pos = np.where((t >= ret.t0)[:, None], self.c_h + x * (self.c_away - self.c_h), pos)
        if self.t_slip is not None:
            m = t >= self.t_slip
            dt = np.minimum(t[m] - self.t_slip, 0.25)                    # slides ~0.2 m out of the hand
            pos[m] = self.c_h + (0.5 * 9.81 * 0.7 * dt ** 2)[:, None] * self.slip_dir
        return pos, _hemisphere(quat)


# ── contact geometry ───────────────────────────────────────────────────────
def _capsule_penetration(pos: np.ndarray, parents: Sequence[str], p0: np.ndarray, p1: np.ndarray,
                         radius: np.ndarray, names: Sequence[str], margin: float,
                         exclude: Mapping[str, Sequence[str]] | None = None, inset=0.0) -> np.ndarray:
    """``max_s (r_s + inset_i + margin − d(taxel_i, capsule_s))`` over capsules not on the taxel's
    own finger (and not excluded) → ``[T,N]`` (≥ 0 ⇔ ``contact.self_touch_labels`` with radius
    ``r_s + inset_i`` says touched). ``inset_i``: depth of taxel i below the skin surface."""
    T, N = pos.shape[:2]
    out = np.full((T, N), -np.inf)
    rad = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(names),))
    ins = np.broadcast_to(np.asarray(inset, dtype=np.float64), (N,))
    for i, par in enumerate(parents):
        drop = set()
        if exclude:
            drop = set(exclude.get(finger_of(par), ())) | set(exclude.get(par, ()))
        keep = [k for k, n in enumerate(names) if finger_of(n) != finger_of(par) and n not in drop]
        if not keep:
            continue
        d = point_segment_distance(pos[:, i, None, :], p0[:, keep], p1[:, keep])        # [T,K]
        out[:, i] = (rad[keep][None] + ins[i] + margin - d).max(1)
    return out


def _glove_contact(layout: Layout, st: dict, p: SynthParams) -> dict[str, np.ndarray]:
    sk = _skeleton()
    go, fp, wp = st["go"], st["fp"], st["wp"]
    pos, nrm = taxel_poses_from_hand(layout, sk, go, fp, wp)
    label = self_touch_from_hand(layout, sk, go, fp, wp, margin=p.self_touch_margin)
    pen = np.zeros(label.shape)
    if label.any():
        rows = np.where(label.any(1))[0]
        with torch.no_grad():
            fk = sk.forward(torch.as_tensor(go[rows]), torch.as_tensor(fp[rows]), torch.as_tensor(wp[rows]))
            p0, p1, rad, names = sk.capsules(fk)
        pen_r = _capsule_penetration(pos[rows], layout.parents, p0.numpy(), p1.numpy(), rad, names,
                                     p.self_touch_margin, DEFAULT_SELF_TOUCH_EXCLUDE)
        pen[rows] = np.maximum(pen_r, 0.0)
    return {"pos": pos, "nrm": nrm, "self": label, "pen_self": np.where(label, pen, 0.0)}


def _robot_capsules(model: URDFModel, q: np.ndarray):
    names = [c[0] for c in _ROBOT_CAPSULES]
    lens = np.array([c[1] for c in _ROBOT_CAPSULES])
    fk = model.fk_numpy(q, links=names)
    p0 = np.stack([fk[n][:, :3, 3] for n in names], 1)
    p1 = p0 + np.stack([fk[n][:, :3, 2] for n in names], 1) * lens[None, :, None]
    return p0, p1, names


def _robot_taxel_inset(layout: Layout, p: SynthParams) -> np.ndarray:
    """Depth of each robot taxel below the skin surface: taxels on finger links sit on the link axis
    (``robot_hand_template`` pads, ``robot_pad_inset``), palm / base taxels on the surface (0)."""
    return np.array([p.robot_pad_inset if finger_of(par) in FINGERS else 0.0 for par in layout.parents])


def _robot_contact(layout: Layout, q: np.ndarray, p: SynthParams) -> dict[str, np.ndarray]:
    """Robot self-contact: link capsules of radius ``robot_link_radius``; a taxel touches when its
    skin surface point (``inset`` above the taxel) is within ``robot_self_margin`` of another finger's
    capsule, i.e. ``d ≤ r + inset + margin`` (on-axis pads: 2r + margin between axes)."""
    model = _robot_model()
    pos, nrm = taxel_poses_from_joints(layout, model, q)
    p0, p1, names = _robot_capsules(model, q)
    inset = _robot_taxel_inset(layout, p)
    label = np.zeros(pos.shape[:2], dtype=bool)
    for v in np.unique(inset):
        idx = np.flatnonzero(inset == v)
        label[:, idx] = self_touch_labels(pos[:, idx], [layout.parents[i] for i in idx], p0, p1,
                                          p.robot_link_radius + v, names, margin=p.robot_self_margin)
    pen = _capsule_penetration(pos, layout.parents, p0, p1, np.full(len(names), p.robot_link_radius), names,
                               p.robot_self_margin, inset=inset)
    return {"pos": pos, "nrm": nrm, "self": label, "pen_self": np.where(label, np.maximum(pen, 0.0), 0.0)}


# ── tactile model ──────────────────────────────────────────────────────────
def _artefact_weights(layout: Layout, kind: str) -> tuple[np.ndarray, list[str]]:
    """``W[N,J]``: coupling of each taxel to the local joints (own finger, distal-weighted; palm →
    Gaussian in distance to the knuckles). J = 15 MANO finger joints (glove) or robot joints."""
    N = layout.n
    if kind == "glove":
        jn = list(MANO_JOINTS[1:])
        sk = _skeleton()
        rest_pos, _ = taxel_poses_from_hand(layout, sk, np.zeros(3), np.zeros((15, 3)))
        joint_pos = sk.rest_joints[1:]
        chains = {f: [j - 1 for j in FINGER_JOINTS[f]] for f in FINGERS}
        seg_chain = {f"{f}{k + 1}": (f, k) for f in FINGERS for k in range(3)}
    else:
        model = _robot_model()
        jn = list(ROBOT_JOINT_NAMES)
        rest_pos, _ = taxel_poses_from_joints(layout, model, np.zeros(len(jn)))
        fk = model.fk_numpy(np.zeros(len(jn)))
        joint_pos = np.stack([fk[model.joint(n).child][:3, 3] for n in jn])
        chains = {f: _robot_joint_index(f) for f in FINGERS}
        seg_chain = {}
        for f in FINGERS:
            for k, n in enumerate(_ROBOT_JOINTS[f]):
                seg_chain[model.joint(n).child] = (f, k)
    W = np.zeros((N, len(jn)))
    knuckles = [chains[f][0] for f in FINGERS]
    for i, par in enumerate(layout.parents):
        if par in seg_chain:
            f, k = seg_chain[par]
            for m, j in enumerate(chains[f]):
                W[i, j] = 0.55 ** abs(m - k)
        else:   # palm / wrist / root links
            d = np.linalg.norm(joint_pos[knuckles] - rest_pos[i], axis=1)
            w = np.exp(-0.5 * (d / 0.03) ** 2)
            W[i, knuckles] = 0.6 * w / max(w.max(), 1e-9)
    return W, jn


def _tactile(layout: Layout, kind: str, t: np.ndarray, theta: np.ndarray, pen: np.ndarray, contact: np.ndarray,
             rng: np.random.Generator, p: SynthParams, n_channels: int) -> dict[str, np.ndarray]:
    """Motion artefact (angle + velocity through a first-order lag) + press + drift + noise → raw."""
    N = layout.n
    W, jnames = _artefact_weights(layout, kind)
    sign = rng.choice([-1.0, 1.0], N)
    g = sign * rng.uniform(*p.artefact_gain, N)
    h = rng.choice([-1.0, 1.0], N) * rng.uniform(*p.artefact_vel_gain, N)
    c = rng.uniform(-p.artefact_quad, p.artefact_quad, N)
    tau = rng.uniform(*p.artefact_tau_s, N)
    k_press = rng.uniform(*p.press_gain, N)
    P_max = rng.uniform(*p.press_max_pct, N)
    tau_press = rng.uniform(*p.press_tau_s, N)
    base = rng.uniform(*p.baseline_raw, n_channels)
    noise_sd = rng.uniform(*p.noise_pct, n_channels)
    D = max(float(t[-1]), 1e-6)
    d_lin = rng.uniform(-p.drift_pct, p.drift_pct, n_channels)
    d_wave = rng.uniform(0.0, p.drift_wave_pct, n_channels)
    d_per = rng.uniform(20.0, 60.0, n_channels)
    d_ph = rng.uniform(0, 2 * np.pi, n_channels)

    u = theta @ W.T                                                         # [T,N]
    thd = np.gradient(theta, t, axis=0)
    x = g * u + c * u * u + h * (thd @ W.T)
    artefact = _lag(x, t, tau)
    press_in = -P_max * (1.0 - np.exp(-k_press * (pen * 1e3) / P_max))
    press = _lag(press_in, t, tau_press)
    drift_all = d_lin * (t[:, None] / D) + d_wave * (np.sin(2 * np.pi * t[:, None] / d_per + d_ph) - np.sin(d_ph))
    noise_all = rng.normal(size=(len(t), n_channels)) * noise_sd
    ch = layout.channels
    drift, noise = drift_all[:, ch], noise_all[:, ch]
    # saturation: some contact runs drop out to the lower ADC rail around their strongest part
    sat = np.zeros((len(t), N), dtype=bool)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    for i in range(N):
        for a, b in _runs(contact[:, i]):
            if b - a >= 5 and rng.uniform() < p.sat_prob:
                seg = pen[a:b, i]
                k = a + int(np.argmax(seg))
                half = max(1, int(round(0.5 * rng.uniform(*p.sat_duration_s) / dt)))
                lo, hi = max(a, k - half), min(b, k + half + 1)
                sat[lo:hi, i] = pen[lo:hi, i] >= 0.5 * seg.max()
    delta = artefact + press + drift + noise
    raw = base[None, :] * (1.0 + (drift_all + noise_all) / 100.0)               # unused channels
    raw[:, ch] = base[ch] * (1.0 + delta / 100.0)
    raw_l = raw[:, ch]
    raw_l[sat] = ADC_MIN
    raw[:, ch] = raw_l
    raw = np.round(np.clip(raw, ADC_MIN, ADC_MAX))
    sat |= (raw[:, ch] <= ADC_MIN) | (raw[:, ch] >= ADC_MAX)                  # anything on a rail
    return {"raw": raw, "artefact": artefact, "press": press, "drift": drift, "noise": noise, "saturated": sat,
            "delta_true": artefact + press + drift, "baseline": base[ch],
            "params": {"W": W, "joint_names": jnames, "angle_gain": g, "vel_gain": h, "quad_gain": c,
                       "tau_s": tau, "press_gain": k_press, "press_max": P_max, "press_tau_s": tau_press,
                       "noise_pct": noise_sd[ch], "baseline_all": base}}


# ── streams ────────────────────────────────────────────────────────────────
def _imu_stream(layout: Layout, motion: _GloveMotion, t: np.ndarray, rng: np.random.Generator,
                p: SynthParams) -> tuple[dict, dict, dict]:
    """IMU arrays (raw, sensor frame), calibration dict and ground-truth summary."""
    st = motion.state(t)
    ideal = synthesize_imu(layout, motion.sk, t, st["go"], st["fp"], st["wp"], gravity=GRAVITY)
    S = len(layout.imu_sites)
    wrist = [i for i, s in enumerate(layout.imu_sites) if s.parent == "wrist"]
    mount = np.stack([np.zeros(3) if i in wrist else _random_rotation_aa(rng, math.radians(p.imu_mount_deg),
                                                                         math.radians(3.0)) for i in range(S)])
    qM = _quat(_R(mount))                                                         # [S,4]
    RM = _R(mount)
    yaw = rng.uniform(-np.pi, np.pi)
    G = _quat(_R([0.0, 0.0, yaw]))
    T = len(t)
    qn = _quat(_R(rng.normal(0.0, math.radians(p.imu_quat_noise_deg), (T, S, 3))))
    q = _qmul(_qmul(np.broadcast_to(G, (T, S, 4)), ideal["quat"]), np.broadcast_to(qM, (T, S, 4)))
    q = _qmul(q, qn)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    bias = rng.normal(0.0, p.imu_gyro_bias, (S, 3))
    gyro = np.einsum("sji,tsj->tsi", RM, ideal["gyro"]) + bias + rng.normal(0.0, p.imu_gyro_noise, (T, S, 3))
    acc = np.einsum("sji,tsj->tsi", RM, ideal["acc"]) + rng.normal(0.0, p.imu_acc_noise, (T, S, 3))
    arrays = {"t": t, "quat": _hemisphere(q).astype(np.float32), "gyro": gyro.astype(np.float32),
              "acc": acc.astype(np.float32), "sites": np.array([s.name for s in layout.imu_sites])}
    q_off = qM * np.array([1.0, -1.0, -1.0, -1.0])                                # conj(M)
    calib = imu_calibration_to_dict(q_off, world=G, sites=[s.name for s in layout.imu_sites])
    return arrays, calib, {"imu_world_yaw_rad": float(yaw), "imu_mount_aa": mount, "imu_gyro_bias": bias}


def _hand_pose_stream(motion: _GloveMotion, t: np.ndarray, rng: np.random.Generator, p: SynthParams):
    st = motion.state(t)
    T = len(t)
    conf = rng.uniform(0.85, 1.0, T)
    bad = np.zeros(T, dtype=bool)
    for _ in range(rng.poisson(p.hand_drop_rate * max(float(t[-1] - t[0]), 0.0))):
        a = int(rng.integers(0, T))
        bad[a:a + int(rng.integers(2, 7))] = True
    conf[bad] = rng.uniform(0.02, 0.3, int(bad.sum()))
    s_go, s_fp, s_wp = p.hand_label_noise
    k = np.where(bad, 10.0, 1.0)
    go = st["go"] + rng.normal(0, s_go, (T, 3)) * k[:, None]
    fp = st["fp"] + rng.normal(0, s_fp, (T, 15, 3)) * k[:, None, None]
    wp = st["wp"] + rng.normal(0, s_wp, (T, 3)) * k[:, None]
    return go, fp, wp, conf


_CAMERA_PRESETS = {"ego": (0, 2, 1.0, -1.0, (70, 80, 95), (140, 150, 160)),
                   "third": (1, 2, -1.0, -1.0, (95, 85, 70), (170, 160, 140))}


def _camera_preset(name: str):
    if name in _CAMERA_PRESETS:
        return _CAMERA_PRESETS[name]
    h = zlib.crc32(name.encode())
    a0 = (0, 1)[h % 2]
    c0 = tuple(int(v) for v in (60 + h % 50, 60 + (h >> 8) % 50, 60 + (h >> 16) % 50))
    return (a0, 2, (1.0, -1.0)[(h >> 4) % 2], -1.0, c0, tuple(min(255, v + 80) for v in c0))


def _render(name: str, hw: tuple[int, int], fingers01: np.ndarray, hand_pos: np.ndarray, ref: np.ndarray,
            obj_pos: np.ndarray | None, obj_color: tuple[int, int, int], rng: np.random.Generator,
            p: SynthParams) -> np.ndarray:
    """Tiny procedural frames ``uint8[F,H,W,3]``: background gradient, a palm with five finger bars
    that shorten and darken with flexion (curling away from the camera), shifted by the wrist
    position, plus the object disk."""
    H, W = hw
    F = fingers01.shape[0]
    a0, a1, s0, s1, c0, c1 = _camera_preset(name)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bg = np.asarray(c0, float) + (np.asarray(c1, float) - np.asarray(c0, float)) * (yy / max(H - 1, 1))[..., None]
    bg = bg + rng.normal(0, 4.0, (H, W, 3))                                            # fixed texture
    img = np.repeat(bg[None], F, 0)
    span = 0.5
    u = W * (0.5 + s0 * (hand_pos[:, a0] - ref[a0]) / span)
    v = H * (0.72 + s1 * (hand_pos[:, a1] - ref[a1]) / span)
    pw, ph = 0.36 * W, 0.22 * H
    X, Y = xx[None], yy[None]
    palm = (np.abs(X - u[:, None, None]) <= pw / 2) & (np.abs(Y - v[:, None, None]) <= ph / 2)
    skin = np.array([205.0, 165.0, 135.0])
    img[palm] = skin * 0.85
    top = v - ph / 2
    bw = pw / 4
    for k in range(4):                                                                  # index … pinky
        fl = np.clip(fingers01[:, k + 1], 0.0, 1.0)
        L = (1.0 - 0.8 * fl) * 0.38 * H
        x0 = u - pw / 2 + k * bw
        m = (X >= x0[:, None, None] + 0.1 * bw) & (X < x0[:, None, None] + 0.9 * bw) & \
            (Y < top[:, None, None]) & (Y >= (top - L)[:, None, None])
        img[m] = 0.0
        img += m[..., None] * (skin[None] * (1.0 - 0.45 * fl)[:, None])[:, None, None, :]
    flt = np.clip(fingers01[:, 0], 0.0, 1.0)                                            # thumb, sideways
    Lt = (1.0 - 0.7 * flt) * 0.3 * W
    m = (X >= (u + pw / 2)[:, None, None]) & (X < (u + pw / 2 + Lt)[:, None, None]) & \
        (np.abs(Y - (v - 0.05 * H)[:, None, None]) <= 0.07 * H)
    img[m] = 0.0
    img += m[..., None] * (skin[None] * (1.0 - 0.45 * flt)[:, None])[:, None, None, :]
    if obj_pos is not None:
        ou = W * (0.5 + s0 * (obj_pos[:, a0] - ref[a0]) / span)
        ov = H * (0.72 + s1 * (obj_pos[:, a1] - ref[a1]) / span)
        r = 0.14 * H
        m = (X - ou[:, None, None]) ** 2 + (Y - ov[:, None, None]) ** 2 <= r * r
        img[m] = 0.0
        img += m[..., None] * np.asarray(obj_color, float)[None, None, None, :]
    gain = 1.0 + rng.normal(0, 0.02, (F, 1, 1, 1))
    img = img * gain + rng.normal(0, p.image_noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _fingers01(kind: str, st: dict) -> np.ndarray:
    if kind == "glove":
        fl = np.stack([st["flex"][:, [j - 1 for j in FINGER_JOINTS[f]]].mean(1) for f in FINGERS], 1)
    else:
        fl = np.stack([st["q"][:, _robot_joint_index(f)[-3:]].mean(1) for f in FINGERS], 1)
    return fl / 1.3


# ── segments & events ──────────────────────────────────────────────────────
def _segments(blocks: list[Block], t: np.ndarray, pressed: np.ndarray, dataset: str,
              guard: float = 0.03) -> tuple[list[dict], float]:
    """Segments from the block labels (+ one ``task`` segment over the D2 task phases);
    ``no_contact`` spans are trimmed (± ``guard`` s) around every ``pressed`` sample (geometric
    contact or a press lag tail) so the label is always truthful.
    Returns (segments, trimmed seconds)."""
    segs, trimmed = [], 0.0
    ct = t[np.asarray(pressed, dtype=bool)]
    for b in blocks:
        for lab in b.labels:
            if lab != "no_contact":
                segs.append({"t0": b.t0, "t1": b.t1, "label": lab})
        if "no_contact" not in b.labels:
            continue
        bad = ct[(ct >= b.t0 - guard) & (ct <= b.t1 + guard)]
        spans = [(b.t0, b.t1)]
        for tc in bad:
            new = []
            for a, e in spans:
                if tc + guard <= a or tc - guard >= e:
                    new.append((a, e))
                else:
                    new += [(a, min(e, tc - guard)), (max(a, tc + guard), e)]
            spans = [(a, e) for a, e in new if e - a > 1e-9]
        kept = [(a, e) for a, e in spans if e - a >= 0.1]
        trimmed += b.dur - sum(e - a for a, e in kept)
        segs += [{"t0": a, "t1": e, "label": "no_contact"} for a, e in kept]
    if dataset == "task":
        tb = [b for b in blocks if b.name in TASK_PHASES]
        segs.append({"t0": tb[0].t0, "t1": tb[-1].t1, "label": "task"})
    segs = [{"t0": round(float(s["t0"]), 6), "t1": round(float(s["t1"]), 6), "label": s["label"]} for s in segs]
    return sorted(segs, key=lambda s: (s["t0"], s["label"])), float(trimmed)


def _events(blocks: list[Block], task: dict | None, success: bool | None) -> list[dict]:
    """events.jsonl records in the acquisition recorder's format (``Recorder.phase_start`` /
    ``instruction`` / ``success``: event names ``instruction`` and ``success``), time-sorted."""
    ev = []
    if task is not None:
        ev.append({"t": 0.0, "type": "instruction", "name": "instruction", "value": task["instruction"]})
    for b in blocks:
        val = {"contact": b.contact, "labels": list(b.labels), "generator": b.gen, **_jsonable(b.info)}
        ev.append({"t": round(b.t0, 6), "type": "phase_start", "name": b.name, "value": val})
        ev.append({"t": round(b.t1, 6), "type": "phase_end", "name": b.name, "value": val})
    if task is not None:
        ev.append({"t": round(blocks[-1].t1, 6), "type": "success", "name": "success", "value": bool(success)})
    order = {"phase_end": 0, "instruction": 1, "phase_start": 2, "marker": 3, "success": 4}
    return sorted(ev, key=lambda e: (e["t"], order.get(e["type"], 5)))


# ── task spec ──────────────────────────────────────────────────────────────
def _task_spec(task_id: str | None, rng: np.random.Generator, p: SynthParams) -> dict:
    tid = task_id if task_id is not None else sorted(SYNTH_TASKS)[int(rng.integers(len(SYNTH_TASKS)))]
    if tid not in SYNTH_TASKS:
        raise ValueError(f"unknown task_id {tid!r}; synthetic tasks: {sorted(SYNTH_TASKS)}")
    spec = SYNTH_TASKS[tid]
    obj = spec["objects"][int(rng.integers(len(spec["objects"])))]
    tgt = spec["targets"][int(rng.integers(len(spec["targets"])))] if spec["targets"] else ""
    return {"task_id": tid, "object": obj, "target": tgt,
            "instruction": spec["template"].format(object=obj, target=tgt),
            "success": bool(rng.uniform() < p.success_prob), "manip": spec["manip"], "grasp": spec["grasp"],
            "radius": OBJECT_RADIUS[obj]}


def _resolve_layout(layout, kind: str) -> tuple[Layout, str | None]:
    """(Layout, reference string for manifest.layout or None → write layout.yaml)."""
    if layout is None:
        layout = DEFAULT_LAYOUTS[kind]
    if isinstance(layout, Layout):
        L, ref = layout, None
    else:
        L = load_layout(layout)
        # built-in name as given; a YAML path absolute, so the manifest resolves from any cwd
        ref = str(Path(layout).resolve()) if Path(layout).suffix in (".yaml", ".yml") else str(layout)
    want = {"glove": "mano", "robot": "urdf"}[kind]
    if L.parent_frame != want:
        raise ValueError(f"{kind} sessions need a layout with parent_frame {want!r}; "
                         f"{L.name!r} has {L.parent_frame!r}")
    if kind == "glove" and not L.imu_sites:
        raise ValueError(f"glove layout {L.name!r} has no imu_sites")
    if kind == "robot":
        missing = sorted(set(L.parents) - set(_robot_model().link_names))
        if missing:
            raise ValueError(f"layout parents {missing} are not links of the synthetic robot URDF "
                             f"(links: {list(_robot_model().link_names)})")
    return L, ref


_GENERATED_FILES = ("session.json", "pressure.npz", "imu.npz", "hand_pose.npz", "joint_state.npz",
                    "object_pose.npz", EVENTS_FILE, GT_FILE, URDF_FILE, LAYOUT_FILE)


def _clear_session(out: Path) -> None:
    """``overwrite=True``: remove what a previous run wrote (its streams + generator files)."""
    names = set(_GENERATED_FILES)
    try:
        names |= {s.file for s in SessionManifest.load(out).streams.values()}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    root = out.resolve()
    for n in names:
        pth = (out / n).resolve()
        if pth.parent != root:
            continue
        if pth.is_dir() and pth.name.startswith("camera_"):
            shutil.rmtree(pth)
        elif pth.is_file():
            pth.unlink()


def _single_torch_thread(fn):
    """Run ``fn`` with one torch intra-op thread (restored afterwards). The generator's tensors
    are tiny; OpenMP workers only add barrier overhead, which explodes (~10×) when other processes
    occupy cores. Note ``torch.set_num_threads`` is process-global while ``fn`` runs."""
    @wraps(fn)
    def run(*args, **kwargs):
        prev = torch.get_num_threads()
        if prev == 1:
            return fn(*args, **kwargs)
        torch.set_num_threads(1)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(prev)
    return run


# ── public API ─────────────────────────────────────────────────────────────
@_single_torch_thread
def generate_session(out_dir: str | Path, *, kind: str = "glove", dataset: str = "motion",
                     duration_s: float = 6.0, seed: int = 0, cameras: Sequence[str] = ("ego",),
                     image_hw: tuple[int, int] = (24, 32), layout: str | Path | Layout | None = None,
                     subject: str = "s0", task_id: str | None = None, session_id: str | None = None,
                     n_channels: int | None = None, params: SynthParams | Mapping | None = None,
                     overwrite: bool = False) -> SessionManifest:
    """Write one complete synthetic raw session into ``out_dir`` (the session directory) and return
    its manifest.

    kind: ``glove`` (pressure + 7 IMUs + hand_pose labels) | ``robot`` (pressure + joint_state +
    ``robot.urdf``). dataset: ``motion`` (D1 protocol blocks) | ``task`` (D2 phases, object pose,
    ``manifest.task``; ``task_id`` from :data:`SYNTH_TASKS`, random if None). ``layout``: built-in
    name (recorded as given), YAML path (recorded as an absolute path) or a :class:`Layout`
    (written to ``layout.yaml``; the manifest then references that file by absolute path;
    ``meta.synthetic.layout_file`` names it session-relative). ``n_channels``: raw ADC channels C
    (default ``max(layout.channels) + 1``). ``params``: :class:`SynthParams` or a dict of overrides.
    ``session_id`` defaults to ``syn_<kind>_<dataset>_<subject>_<seed>``. An existing session in
    ``out_dir`` raises ``FileExistsError`` unless ``overwrite`` (then its files are removed first).
    Deterministic in ``seed`` (except ``created_utc``).
    """
    if kind not in DEFAULT_LAYOUTS:
        raise ValueError(f"kind must be 'glove' or 'robot', got {kind!r}")
    if dataset not in ("motion", "task"):
        raise ValueError(f"dataset must be 'motion' or 'task', got {dataset!r}")
    if int(seed) < 0:
        raise ValueError("seed must be ≥ 0")
    hw = tuple(int(v) for v in image_hw)
    if len(hw) != 2 or min(hw) < 4:
        raise ValueError(f"image_hw must be (H, W) with H, W ≥ 4, got {image_hw}")
    cams = [str(c) for c in cameras]
    if len(set(cams)) != len(cams) or any(not _safe_name(c) for c in cams):
        raise ValueError(f"camera names must be unique, non-empty, without path separators: {cams}")
    subject = str(subject)
    if not _safe_name(subject):
        raise ValueError(f"subject must be a non-empty name without path separators, got {subject!r}")
    if task_id is not None and task_id not in SYNTH_TASKS:
        raise ValueError(f"unknown task_id {task_id!r}; synthetic tasks: {sorted(SYNTH_TASKS)}")
    p = SynthParams.from_dict(params)
    D = _check_duration(duration_s, dataset)
    L, layout_ref = _resolve_layout(layout, kind)
    C = int(L.channels.max()) + 1
    if n_channels is not None:
        if int(n_channels) < C:
            raise ValueError(f"n_channels={n_channels} < {C} required by the layout channels")
        C = int(n_channels)
    out = Path(out_dir)
    if (out / "session.json").exists():
        if not overwrite:
            raise FileExistsError(f"{out} already holds a session (pass overwrite=True)")
        _clear_session(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(seed)
    sid = session_id or f"syn_{kind}_{dataset}_{subject}_{seed}"

    style_rng = _rng(zlib.crc32(subject.encode()), "subject_style")
    style = {"amp": float(style_rng.uniform(0.85, 1.1)), "speed": float(style_rng.uniform(0.85, 1.15))}
    task = _task_spec(task_id, _rng(seed, "task"), p) if dataset == "task" else None
    blocks = plan_task_session(D) if dataset == "task" else plan_motion_session(D, kind)
    Motion = _GloveMotion if kind == "glove" else _RobotMotion
    motion = Motion(blocks, _rng(seed, "motion"), style, task, p)

    # ── pressure (ground truth at the pressure timestamps) ────────────────
    t_p = _stream_times(_rng(seed, "clock", "pressure"), p.pressure_hz, D, p)
    st = motion.state(t_p)
    geo = _glove_contact(L, st, p) if kind == "glove" else _robot_contact(L, st["q"], p)
    pen_obj = np.zeros((len(t_p), L.n))
    obj_contact = np.zeros((len(t_p), L.n), dtype=bool)
    if task is not None:
        c_obj, _ = motion.object_pose(t_p)
        inset = 0.0 if kind == "glove" else _robot_taxel_inset(L, p)
        pen_o = task["radius"] + p.object_margin + inset - np.linalg.norm(geo["pos"] - c_obj[:, None, :], axis=-1)
        obj_contact = pen_o >= 0.0
        pen_obj = np.maximum(pen_o, 0.0)
    contact = geo["self"] | obj_contact
    pen = geo["pen_self"] + pen_obj
    theta = st["flex"] if kind == "glove" else st["q"]
    tac = _tactile(L, kind, t_p, theta, pen, contact, _rng(seed, "tactile"), p, C)
    np.savez(out / "pressure.npz", t=t_p, raw=tac["raw"])
    streams = {"pressure": StreamInfo("pressure.npz", p.pressure_hz, ["t", "raw"])}

    meta_syn: dict[str, Any] = {"style": style, "gt_file": GT_FILE, "channels_used": L.channels,
                                "n_channels": C, "blocks": [asdict(b) for b in blocks]}
    calib: dict = {}
    # ── IMU + hand labels (glove) / joint state (robot) ───────────────────
    if kind == "glove":
        t_i = _stream_times(_rng(seed, "clock", "imu"), p.imu_hz, D, p)
        imu, calib, imu_gt = _imu_stream(L, motion, t_i, _rng(seed, "imu"), p)
        np.savez(out / "imu.npz", **imu)
        streams["imu"] = StreamInfo("imu.npz", p.imu_hz, ["t", "quat", "gyro", "acc", "sites"])
        meta_syn.update(imu_gt)
        if dataset == "motion":
            meta_syn["calibration_pose"] = {"global_orient": motion.go_c, "finger_pose": "flat",
                                            "note": "flat hand, palm down, during the 'calibration' segment"}
        t_h = _stream_times(_rng(seed, "clock", "hand_pose"), p.hand_pose_hz, D, p)
        go, fp, wp, conf = _hand_pose_stream(motion, t_h, _rng(seed, "hand_pose"), p)
        save_hand_labels(out / "hand_pose.npz", t_h, go, fp, wp, conf)
        streams["hand_pose"] = StreamInfo("hand_pose.npz", p.hand_pose_hz,
                                          ["t", "global_orient", "finger_pose", "wrist_pos", "confidence"])
    else:
        model = _robot_model()
        (out / URDF_FILE).write_text(robot_hand_urdf())
        t_j = _stream_times(_rng(seed, "clock", "joint_state"), p.joint_hz, D, p)
        jr = _rng(seed, "joint_state")
        q = motion.state(t_j)["q"]
        dq = (motion.state(t_j + 1e-3)["q"] - motion.state(t_j - 1e-3)["q"]) / 2e-3
        np.savez(out / "joint_state.npz", t=t_j,
                 q=(q + jr.normal(0, p.joint_noise, q.shape)).astype(np.float32),
                 qd=(dq + jr.normal(0, 10 * p.joint_noise, q.shape)).astype(np.float32),
                 names=np.array(model.joint_names))
        streams["joint_state"] = StreamInfo("joint_state.npz", p.joint_hz, ["t", "q", "qd", "names"])

    # ── object pose (task) ────────────────────────────────────────────────
    if task is not None:
        t_o = _stream_times(_rng(seed, "clock", "object_pose"), p.object_hz, D, p)
        pos_o, quat_o = motion.object_pose(t_o)
        orng = _rng(seed, "object_pose")
        np.savez(out / "object_pose.npz", t=t_o, pos=(pos_o + orng.normal(0, 0.001, pos_o.shape)).astype(np.float32),
                 quat=quat_o.astype(np.float32))
        streams["object_pose"] = StreamInfo("object_pose.npz", p.object_hz, ["t", "pos", "quat"])
        meta_syn["object_frame"] = "world" if kind == "glove" else "hand_base"
        meta_syn["object_radius_m"] = task["radius"]
        meta_syn["grasp_closure"] = motion.grasp_closure
        meta_syn["object_center_hand"] = motion.c_h

    # ── cameras ───────────────────────────────────────────────────────────
    offsets = {}
    for cam in cams:
        crng = _rng(seed, "camera", cam)
        off = float(crng.uniform(*p.camera_offset_s))
        t_true = _stream_times(crng, p.camera_fps, D - max(off, 0.0), p)
        t_true = t_true[t_true + off >= 0.0]
        stc = motion.state(t_true)
        hand = stc["wp"] if kind == "glove" else np.zeros((len(t_true), 3))
        obj = motion.object_pose(t_true)[0] if task is not None else None
        color = (60 + zlib.crc32((task or {}).get("object", "x").encode()) % 150, 90, 200)
        frames = _render(cam, hw, _fingers01(kind, stc), hand, motion.ref_pos, obj, color, crng, p)
        cdir = out / f"camera_{cam}"
        cdir.mkdir(exist_ok=True)
        np.save(cdir / "frames.npy", frames)
        np.save(cdir / "timestamps.npy", t_true + off)
        streams[f"camera_{cam}"] = StreamInfo(f"camera_{cam}", p.camera_fps, ["frames", "timestamps"], method="zoh")
        offsets[cam] = off
    meta_syn["camera_offset_s"] = offsets

    # ── labels, events, ground truth, manifest ────────────────────────────
    pressed = contact.any(1) | (np.abs(tac["press"]) > PRESS_EPS_PCT).any(1)
    segments, trimmed = _segments(blocks, t_p, pressed, dataset)
    meta_syn["no_contact_trimmed_s"] = trimmed
    success = None if task is None else bool(task["success"])
    with open(out / EVENTS_FILE, "w") as fh:
        for e in _events(blocks, task, success):
            fh.write(json.dumps(e) + "\n")
    tp = tac["params"]
    pose_gt = ({"hand_global_orient": st["go"], "hand_finger_pose": st["fp"], "hand_wrist_pos": st["wp"]}
               if kind == "glove" else {"q": st["q"]})
    if task is not None:
        pose_gt["object_pos"] = c_obj
    np.savez(out / GT_FILE, t=t_p, **pose_gt, artefact_pct=tac["artefact"].astype(np.float32),
             press_pct=tac["press"].astype(np.float32), drift_pct=tac["drift"].astype(np.float32),
             noise_pct=tac["noise"].astype(np.float32), delta_true_pct=tac["delta_true"].astype(np.float32),
             contact=contact, self_touch=geo["self"], object_contact=obj_contact,
             penetration_m=pen.astype(np.float32), saturated=tac["saturated"], baseline_raw=tac["baseline"],
             taxel_pos=geo["pos"].astype(np.float32), joint_angle=theta.astype(np.float32),
             joint_angle_names=np.array(tp["joint_names"]), channels=L.channels,
             artefact_weights=tp["W"], angle_gain=tp["angle_gain"], vel_gain=tp["vel_gain"],
             quad_gain=tp["quad_gain"], tau_s=tp["tau_s"], press_gain=tp["press_gain"],
             press_max_pct=tp["press_max"], press_tau_s=tp["press_tau_s"], noise_std_pct=tp["noise_pct"],
             baseline_raw_all=tp["baseline_all"])
    if layout_ref is None:
        (out / LAYOUT_FILE).write_text(yaml.safe_dump(L.to_dict(units="mm"), sort_keys=False))
        layout_ref = str((out / LAYOUT_FILE).resolve())
        meta_syn["layout_file"] = LAYOUT_FILE
    meta = {"generator": "robot_skin.datasets.synthetic", "generator_version": GENERATOR_VERSION,
            "seed": seed, "duration_s": D, "synthetic": _jsonable(meta_syn)}
    if kind == "robot":
        meta["urdf"] = URDF_FILE
    task_d = None if task is None else {"task_id": task["task_id"], "instruction": task["instruction"],
                                        "object": task["object"], "success": success}
    man = SessionManifest(kind=kind, layout=layout_ref, streams=streams, session_id=sid, master_hz=200.0,
                          segments=segments, notes="synthetic session (robot_skin.datasets.synthetic)",
                          meta=meta, dataset=dataset, subject=str(subject), task=task_d,
                          calibration=_jsonable(calib))
    man.save(out)
    return man


def generate_dataset(root: str | Path, *, n_motion: int = 2, n_task: int = 2, kind: str = "glove",
                     subjects: Sequence[str] = ("s0", "s1"), seed: int = 0, **kw) -> list[Path]:
    """Generate ``n_motion`` D1 and ``n_task`` D2 sessions under ``root/<dataset>/<subject>/<session_id>``.

    Sessions are spread round-robin over ``subjects``; tasks cycle through :data:`SYNTH_TASKS`
    (unless ``task_id`` is given in ``kw``); each session gets a distinct derived seed. Extra ``kw``
    go to :func:`generate_session`. Returns the session directories (motion first).
    """
    if not subjects:
        raise ValueError("need at least one subject")
    reserved = sorted({"dataset", "seed", "subject", "session_id"} & set(kw))
    if reserved:
        raise ValueError(f"generate_dataset sets {reserved} per session; do not pass them")
    if int(n_motion) < 0 or int(n_task) < 0:
        raise ValueError(f"n_motion / n_task must be ≥ 0, got {n_motion}, {n_task}")
    root = Path(root)
    tasks = sorted(SYNTH_TASKS)
    out: list[Path] = []
    k = 0
    for dataset, n in (("motion", n_motion), ("task", n_task)):
        for i in range(int(n)):
            subj = str(subjects[i % len(subjects)])
            s = int(seed) * 10007 + k
            k += 1
            sid = f"syn_{kind}_{dataset}_{subj}_{i:03d}_{s}"
            d = root / dataset / subj / sid
            extra = dict(kw)
            if dataset == "task":
                extra.setdefault("task_id", tasks[i % len(tasks)])
            else:
                extra.pop("task_id", None)
            generate_session(d, kind=kind, dataset=dataset, seed=s, subject=subj, session_id=sid, **extra)
            out.append(d)
    return out


def load_ground_truth(session_dir: str | Path) -> dict[str, np.ndarray]:
    """Arrays of ``gt_synthetic.npz`` (pressure timestamps, layout order): ``artefact_pct`` (true
    no-contact motion artefact, ΔS % vs the true baseline), ``press_pct``, ``drift_pct``,
    ``noise_pct``, ``delta_true_pct``, ``contact`` / ``self_touch`` / ``object_contact`` /
    ``saturated`` (raw on an ADC rail) bool, ``penetration_m``, ``baseline_raw``, ``taxel_pos``, ``joint_angle`` (the
    artefact model input θ) and the artefact parameters (``artefact_weights``, ``angle_gain``,
    ``vel_gain``, ``quad_gain``, ``tau_s`` …), plus the true pose (glove: ``hand_global_orient``,
    ``hand_finger_pose``, ``hand_wrist_pos``; robot: ``q``) and, for tasks, ``object_pos``."""
    with np.load(Path(session_dir) / GT_FILE) as z:
        return {k: z[k] for k in z.files}
