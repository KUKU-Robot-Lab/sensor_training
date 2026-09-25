"""Hardware interfaces of the control loop + a simulated robot hand / camera for tests and bring-up.

The deployment loop (:class:`robot_skin.control.runner.PolicyRunner`) talks to hardware only through
two small protocols, so a real hand is brought up by implementing them (see ``docs/DEPLOYMENT.md``):

:class:`RobotHandInterface`
    ``joint_names`` (actuated joints, driver order), ``lower`` / ``upper`` (rad, ``[D]``),
    ``read_state() -> (t, q[D], qd[D] | None)``, ``read_pressure() -> (t, raw[C])`` (the tactile
    front-end's raw counts, **channel order** — the order of ``pressure.npz``; the layout maps
    channels to taxels) and ``send_joint_targets(q_target[D])`` (position targets, driver order).
    Optional: ``velocity_limits`` ``[D]`` rad/s, ``start()`` / ``stop()`` / ``estop()``, ``layout``
    (taxel layout of the skin) and ``urdf_xml`` / ``urdf_path`` (logged with the session so it can
    be re-ingested by ``datasets.build``).
    **Timestamps** are seconds on the *host* clock the runner uses (``time.monotonic`` by
    default) — a driver that receives device time must translate it (and should stamp on arrival
    when unsure): the tactile processor, safety watchdog and the session log compare them directly.

:class:`CameraInterface`
    ``name`` and ``read() -> (t, uint8[H,W,3] | None)`` — the latest frame and its timestamp.

Simulation (deterministic, driven by a :class:`~robot_skin.acquisition.sources.SimClock`):

:class:`FakeRobotHand`
    URDF kinematics (default: the synthetic 16-DoF hand of :func:`robot_skin.datasets.synthetic.
    robot_hand_urdf`, matching ``common/layouts/robot_hand_template.yaml``); first-order position
    tracking ``q̇ = clip((q* − q)/τ, ±v_max)`` integrated at ``sim_hz``; a :class:`VirtualObject`
    between the fingers — a finger touches it once its closure (mean flexion angle of the finger)
    exceeds ``angle`` and cannot sink more than ``compliance_rad`` into it; and a phenomenological
    skin with the same ingredients as the synthetic sessions: a per-taxel **motion artefact**
    ``ΔS_art = lag_τ(g·u + c·u² + h·u̇)`` with ``u = W q`` (``W`` = distal-weighted coupling of a
    taxel to the joints of its kinematic chain), a **press** ``−P_max(1 − e^{−k·pen/P_max})``
    (SATS sign: press → negative ΔS) on the taxels of touching fingers (palm taxels when ≥ 2 fingers
    touch), white noise, and raw counts ``b·(1 + ΔS/100)`` clipped to the ADC rails (optional
    injected dropouts to the lower rail). Ground truth is available from :meth:`FakeRobotHand.truth`.
:class:`FakeCamera`
    Tiny frames at ``rate_hz`` whose content follows the hand closure (enough for smoke tests of
    the vision path; not a renderer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from common.layouts import Layout, load_layout
from common.signal import ADC_MAX, ADC_MIN

__all__ = [
    "FINGERS", "RobotHandInterface", "CameraInterface", "check_robot", "check_camera", "VirtualObject",
    "FakeRobotHand", "FakeCamera", "finger_joint_groups", "taxel_coupling", "SYNTHETIC_HAND_HUMAN_TO_ROBOT",
    "urdf_tip_fk", "layout_tip_offsets", "load_urdf_model",
]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: rotation MANO canonical hand frame (fingers −x, radial/thumb side +z, palm faces −y; see
#: ``pose.mano``) → base frame of the synthetic hand (fingers +z, radial +y, pads face +x; see
#: ``datasets.synthetic.robot_hand_urdf``): the columns are the images of the MANO axes. Pass as
#: ``FingertipRetargeter(human_to_robot=...)``; a real hand needs its own.
SYNTHETIC_HAND_HUMAN_TO_ROBOT = ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0))
#: joint-name fragments treated as non-flexion (abduction / spread) when computing finger closure
_NON_FLEX = ("abd", "spread", "yaw", "twist")


# ─────────────────────────────────────────────────────────────── protocols

@runtime_checkable
class RobotHandInterface(Protocol):
    """What :class:`~robot_skin.control.runner.PolicyRunner` needs from a robot hand (module docstring)."""

    joint_names: Sequence[str]
    lower: np.ndarray
    upper: np.ndarray

    def read_state(self) -> tuple[float, np.ndarray, np.ndarray | None]:
        """``(t, q[D], qd[D] | None)`` — host-clock seconds, rad, rad/s, ``joint_names`` order."""
        ...

    def read_pressure(self) -> tuple[float, np.ndarray]:
        """``(t, raw[C])`` — latest raw tactile sample (channel order, ADC counts)."""
        ...

    def send_joint_targets(self, q_target: np.ndarray) -> None:
        """Position targets ``[D]`` (rad, ``joint_names`` order). Must not block for long."""
        ...


@runtime_checkable
class CameraInterface(Protocol):
    """A camera stream: ``read() -> (t, uint8[H,W,3] | None)`` (latest frame, host-clock time)."""

    name: str

    def read(self) -> tuple[float, np.ndarray | None]:
        ...


def check_robot(robot: Any) -> None:
    """Raise ``TypeError`` naming the missing members of a :class:`RobotHandInterface`."""
    need = ("joint_names", "lower", "upper", "read_state", "read_pressure", "send_joint_targets")
    missing = [m for m in need if not hasattr(robot, m)]
    if missing:
        raise TypeError(f"{type(robot).__name__} does not implement RobotHandInterface: missing {missing} "
                        "(see robot_skin/control/interfaces.py and docs/DEPLOYMENT.md)")
    D = len(robot.joint_names)
    for nm in ("lower", "upper"):
        v = np.asarray(getattr(robot, nm), dtype=np.float64).reshape(-1)
        if v.shape != (D,):
            raise ValueError(f"robot.{nm} must have {D} entries (one per joint), got {v.shape}")
    if np.any(np.asarray(robot.upper) < np.asarray(robot.lower)):
        raise ValueError("robot.upper < robot.lower for some joints")


def check_camera(cam: Any) -> None:
    missing = [m for m in ("name", "read") if not hasattr(cam, m)]
    if missing:
        raise TypeError(f"{type(cam).__name__} does not implement CameraInterface: missing {missing}")


# ─────────────────────────────────────────────────────────────── kinematic helpers

def load_urdf_model(urdf: Any = None) -> tuple[Any, str | None]:
    """``(URDFModel, xml | None)`` from ``None`` (the synthetic 16-DoF hand of
    :func:`robot_skin.datasets.synthetic.robot_hand_urdf`), a model, a file path or an XML string."""
    from ..pose.urdf import URDFModel

    if urdf is None:
        from ..datasets.synthetic import robot_hand_urdf

        return URDFModel.from_string(robot_hand_urdf()), robot_hand_urdf()
    if isinstance(urdf, URDFModel):
        return urdf, None
    s = str(urdf)
    if s.lstrip().startswith("<"):
        return URDFModel.from_string(s), s
    p = Path(s)
    return URDFModel.from_file(p), p.read_text()


def finger_joint_groups(model, fingers: Sequence[str] = FINGERS) -> dict[str, list[int]]:
    """``{finger: [actuated joint indices]}`` — the joints on the chain of the deepest link whose name
    starts with the finger name (``index_distal_link`` → index MCP/PIP/DIP). Fingers without links
    are omitted."""
    out: dict[str, list[int]] = {}
    act = {n: i for i, n in enumerate(model.joint_names)}
    for f in fingers:
        links = [ln for ln in model.link_names if ln.lower().startswith(f)]
        if not links:
            continue
        deepest = max(links, key=lambda ln: len(model.chain(ln)))
        idx = [act[j.name] for j in model.chain(deepest) if j.name in act]
        # keep only the joints that belong to this finger (not a shared wrist / palm joint)
        idx = [i for i in idx if model.joint(model.joint_names[i]).child.lower().startswith(f)]
        if idx:
            out[f] = idx
    return out


def taxel_coupling(layout: Layout, model, *, decay: float = 0.55, palm_weight: float = 0.6,
                   palm_sigma_m: float = 0.03) -> np.ndarray:
    """``W[N, D]`` — how strongly each taxel's motion artefact follows each joint: the joints on the
    chain of the taxel's parent link, weighted ``decay^k`` from the most distal one (a fingertip taxel
    feels its own DIP most); taxels on links without actuated joints (palm) couple to the finger
    base joints with a Gaussian in the distance to each knuckle (the same model family as
    ``datasets.synthetic``)."""
    from ..pose.robot_fk import taxel_poses_from_joints

    D = model.n_dof
    act = {n: i for i, n in enumerate(model.joint_names)}
    W = np.zeros((layout.n, D))
    groups = finger_joint_groups(model)
    rest_pos, _ = taxel_poses_from_joints(layout, model, np.zeros(D))
    fk = model.fk_numpy(np.zeros(D))
    knuckles = [g[0] for g in groups.values()]
    for n, par in enumerate(layout.parents):
        idx = [act[j.name] for j in model.chain(par) if j.name in act]
        if idx:
            for k, j in enumerate(reversed(idx)):
                W[n, j] = decay ** k
        elif knuckles:
            kp = np.stack([fk[model.joint(model.joint_names[j]).child][:3, 3] for j in knuckles])
            d = np.linalg.norm(kp - rest_pos[n], axis=1)
            w = np.exp(-0.5 * (d / palm_sigma_m) ** 2)
            W[n, knuckles] = palm_weight * w / max(float(w.max()), 1e-9)
    return W


def layout_tip_offsets(layout: Layout, tip_links: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Fingertip point per finger in its tip-link frame: the mean position of the layout's taxels on
    that link (fingertip pads), or the link origin when the link carries no taxel."""
    out = {}
    pos, par = layout.positions, layout.parents
    for f, link in tip_links.items():
        idx = [i for i, p in enumerate(par) if p == link]
        out[f] = pos[idx].mean(0) if idx else np.zeros(3)
    return out


def urdf_tip_fk(model, tip_links: Mapping[str, str], tip_offsets: Mapping[str, Sequence[float]] | None = None,
                base_link: str | None = None) -> Callable:
    """Differentiable ``q[B,D] → {finger: tip[B,3]}`` (plus ``base_link`` → ``[B,4,4]``) for
    :class:`~robot_skin.action.FingertipRetargeter`: the fingertip is the tip link's frame applied to
    ``tip_offsets[finger]`` (link frame, m; default origin) — link origins sit at the last joint, not
    at the fingertip the human keypoints describe."""
    import torch

    links = sorted(set(tip_links.values()) | ({base_link} if base_link else set()))
    offs = {f: np.zeros(3) if tip_offsets is None or f not in tip_offsets else np.asarray(tip_offsets[f], np.float64)
            for f in tip_links}

    def fk(q):
        T = model.fk(q, links=links)
        out = {}
        for f, ln in tip_links.items():
            M = T[ln]
            o = torch.as_tensor(offs[f], dtype=M.dtype, device=M.device)
            out[f] = M[..., :3, 3] + (M[..., :3, :3] @ o[:, None])[..., 0]
        if base_link:
            out[base_link] = T[base_link]
        return out

    return fk


# ─────────────────────────────────────────────────────────────── fake robot hand

@dataclass
class VirtualObject:
    """An object held between the fingers of :class:`FakeRobotHand`.

    Finger ``f`` touches it when its closure (mean of its flexion joints, rad) exceeds ``angle``
    (scalar or ``{finger: rad}``); the penetration ``closure − angle`` is capped at
    ``compliance_rad`` (the object stops the finger there). ``palm``: palm taxels are pressed when
    at least two fingers touch (a power grasp)."""

    angle: float | Mapping[str, float] = 0.9
    compliance_rad: float = 0.25
    fingers: tuple[str, ...] = FINGERS
    palm: bool = True

    def angle_of(self, finger: str) -> float:
        if isinstance(self.angle, Mapping):
            return float(self.angle.get(finger, math.inf))
        return float(self.angle)


class FakeRobotHand:
    """Simulated :class:`RobotHandInterface` with a synthetic tactile skin (module docstring).

    Args:
        urdf: ``None`` (synthetic 16-DoF hand), a :class:`~robot_skin.pose.urdf.URDFModel`, a path or
            an XML string.
        layout: skin layout (``parent_frame: urdf``; parents are links of the URDF).
        clock: host clock callable (default a new :class:`~robot_skin.acquisition.sources.SimClock`
            at 0). The simulation integrates lazily up to ``clock()`` on every call.
        sim_hz: integration rate; ``tau_s``: position-tracking time constant; ``velocity_limit``:
            rad/s (scalar / ``[D]``; default the URDF limits).
        obj: :class:`VirtualObject` (or its kwargs dict) or ``None`` (free motion only).
        n_channels: raw channels (≥ max layout channel + 1; extra channels carry baseline + noise).
        noise_pct: per-sample white noise (ΔS %, std); the artefact / press parameters are drawn
            per taxel from the given ranges with ``seed``.
    """

    def __init__(self, urdf: Any = None, layout: str | Path | Layout = "robot_hand_template", *,
                 clock: Callable[[], float] | None = None, sim_hz: float = 400.0, tau_s: float = 0.05,
                 q0: Sequence[float] | None = None, velocity_limit: float | Sequence[float] | None = None,
                 obj: VirtualObject | Mapping[str, Any] | None = None, n_channels: int | None = None,
                 noise_pct: float = 0.02, artefact_gain_pct: tuple[float, float] = (0.5, 2.0),
                 artefact_quad_pct: float = 0.3, artefact_vel_gain: tuple[float, float] = (0.02, 0.1),
                 artefact_tau_s: tuple[float, float] = (0.02, 0.08),
                 press_gain_pct_per_rad: tuple[float, float] = (60.0, 120.0),
                 press_max_pct: tuple[float, float] = (20.0, 45.0), press_tau_s: float = 0.015,
                 baseline_raw: tuple[float, float] = (2.0e6, 6.0e6), seed: int = 0):
        from ..acquisition.sources import SimClock
        from ..pose.robot_fk import taxel_poses_from_joints

        self.model, self.urdf_xml = load_urdf_model(urdf)
        self.layout = layout if isinstance(layout, Layout) else load_layout(layout)
        if self.layout.parent_frame != "urdf":
            raise ValueError(f"FakeRobotHand needs a layout with parent_frame 'urdf', got {self.layout.parent_frame!r}")
        taxel_poses_from_joints(self.layout, self.model, np.zeros(self.model.n_dof))   # validates parent links
        self.clock = clock if clock is not None else SimClock(0.0)
        if not (sim_hz > 0 and tau_s > 0):
            raise ValueError("sim_hz and tau_s must be > 0")
        self.sim_dt = 1.0 / float(sim_hz)
        self.tau_s = float(tau_s)
        D = self.model.n_dof
        self.joint_names: tuple[str, ...] = tuple(self.model.joint_names)
        self.lower = np.asarray(self.model.lower, dtype=np.float64).copy()
        self.upper = np.asarray(self.model.upper, dtype=np.float64).copy()
        vl = self.model.velocity_limits if velocity_limit is None else velocity_limit
        self.velocity_limits = np.broadcast_to(np.asarray(vl, dtype=np.float64), (D,)).copy()
        self.velocity_limits[~np.isfinite(self.velocity_limits) | (self.velocity_limits <= 0)] = np.inf
        q_init = np.zeros(D) if q0 is None else np.asarray(q0, dtype=np.float64).reshape(D)
        self.obj = VirtualObject(**dict(obj)) if isinstance(obj, Mapping) else obj
        self.groups = finger_joint_groups(self.model)
        self.closure_joints = {f: [j for j in idx if not any(s in self.joint_names[j].lower() for s in _NON_FLEX)]
                               for f, idx in self.groups.items()}
        self.closure_joints = {f: v for f, v in self.closure_joints.items() if v}
        from ..contact.self_touch import finger_of

        self.taxel_finger = [finger_of(p) for p in self.layout.parents]
        C_min = int(self.layout.channels.max()) + 1
        self.n_channels = C_min if n_channels is None else int(n_channels)
        if self.n_channels < C_min:
            raise ValueError(f"n_channels {self.n_channels} < {C_min} used by layout {self.layout.name!r}")

        rng = np.random.default_rng(seed)
        N = self.layout.n
        self.W = taxel_coupling(self.layout, self.model)
        self._g = rng.choice([-1.0, 1.0], N) * rng.uniform(*artefact_gain_pct, N)
        self._c = rng.uniform(-artefact_quad_pct, artefact_quad_pct, N)
        self._h = rng.choice([-1.0, 1.0], N) * rng.uniform(*artefact_vel_gain, N)
        self._tau_art = rng.uniform(*artefact_tau_s, N)
        self._k_press = rng.uniform(*press_gain_pct_per_rad, N)
        self._p_max = rng.uniform(*press_max_pct, N)
        self._tau_press = float(press_tau_s)
        self.baseline_raw = rng.uniform(*baseline_raw, self.n_channels)
        self.noise_pct = float(noise_pct)
        self._rng = np.random.default_rng(None if seed is None else seed + 1)
        self._q = self.lower + np.clip(q_init - self.lower, 0.0, self.upper - self.lower)
        self._qd = np.zeros(D)
        self._target = self._q.copy()
        self._art = np.zeros(N)
        self._press = np.zeros(N)
        self._pen = {f: 0.0 for f in FINGERS}
        self._dropout_until = np.full(N, -np.inf)
        self._t0 = float(self.clock())
        self._k = 0                                             # integration steps since _t0
        self._t = self._t0
        self._last_raw: tuple[float, np.ndarray] | None = None
        self.n_commands = 0
        self.estopped = False

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        self._advance()

    def stop(self) -> None:
        self._advance()

    def estop(self) -> None:
        """Freeze at the current position (targets are ignored until :meth:`reset_estop`)."""
        self._advance()
        self.estopped = True
        self._target = self._q.copy()

    def reset_estop(self) -> None:
        self.estopped = False

    # ── simulation ────────────────────────────────────────────────────────
    def _closure(self, q: np.ndarray, f: str) -> float:
        return float(np.mean(q[self.closure_joints[f]]))

    def _integrate(self, dt: float) -> None:
        q, lo, hi = self._q, self.lower, self.upper
        v = np.clip((self._target - q) / self.tau_s, -self.velocity_limits, self.velocity_limits)
        qn = np.clip(q + v * dt, lo, hi)
        pen = {f: 0.0 for f in FINGERS}
        if self.obj is not None:
            for f in self.obj.fingers:
                if f not in self.closure_joints:
                    continue
                a = self.obj.angle_of(f)
                J = self.closure_joints[f]
                cl = self._closure(qn, f)
                lim = a + float(self.obj.compliance_rad)
                if cl > lim:                                     # the object stops the finger
                    qn[J] = np.clip(qn[J] - (cl - lim), lo[J], hi[J])
                    cl = self._closure(qn, f)
                pen[f] = float(np.clip(cl - a, 0.0, self.obj.compliance_rad))
        self._qd = (qn - q) / dt
        self._q = qn
        self._pen = pen
        # skin: lagged motion artefact + lagged press
        u = self.W @ self._q
        ud = self.W @ self._qd
        x = self._g * u + self._c * u * u + self._h * ud
        self._art += (x - self._art) * (1.0 - np.exp(-dt / self._tau_art))
        p_in = np.zeros(self.layout.n)
        touching = [f for f in FINGERS if pen[f] > 0]
        for n, f in enumerate(self.taxel_finger):
            if f in pen and pen[f] > 0:
                p = pen[f]
            elif f not in pen and self.obj is not None and self.obj.palm and len(touching) >= 2:
                p = 0.5 * float(np.mean([pen[g] for g in touching]))
            else:
                continue
            p_in[n] = -self._p_max[n] * (1.0 - math.exp(-self._k_press[n] * p / self._p_max[n]))
        self._press += (p_in - self._press) * (1.0 - math.exp(-dt / self._tau_press))

    def _advance(self) -> None:
        # step count from the start time (not an accumulated float time, which drifts one step
        # behind the caller's clock after long runs); the tolerance absorbs the clock's own rounding
        now = float(self.clock())
        n = int(math.floor((now - self._t0) / self.sim_dt + 1e-6))
        while self._k < n:
            self._integrate(self.sim_dt)
            self._k += 1
        self._t = self._t0 + self._k * self.sim_dt

    # ── RobotHandInterface ────────────────────────────────────────────────
    def read_state(self) -> tuple[float, np.ndarray, np.ndarray]:
        self._advance()
        return self._t, self._q.copy(), self._qd.copy()

    def read_pressure(self) -> tuple[float, np.ndarray]:
        """Raw counts ``[C]`` at the current simulation time (a new noise draw per time step)."""
        self._advance()
        if self._last_raw is not None and self._last_raw[0] == self._t:
            return self._t, self._last_raw[1].copy()
        noise = self._rng.normal(size=self.n_channels) * self.noise_pct
        delta_c = noise.copy()
        ch = self.layout.channels
        delta_c[ch] += self._art + self._press
        raw = self.baseline_raw * (1.0 + delta_c / 100.0)
        drop = self._dropout_until > self._t
        if drop.any():
            raw[ch[drop]] = ADC_MIN
        raw = np.round(np.clip(raw, ADC_MIN, ADC_MAX))
        self._last_raw = (self._t, raw)
        return self._t, raw.copy()

    def send_joint_targets(self, q_target: np.ndarray) -> None:
        q = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if q.shape != (self.model.n_dof,):
            raise ValueError(f"q_target must have {self.model.n_dof} entries, got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_target contains non-finite values")
        self._advance()
        if not self.estopped:
            self._target = np.clip(q, self.lower, self.upper)
        self.n_commands += 1

    # ── test hooks / ground truth ─────────────────────────────────────────
    def inject_dropout(self, taxels: Sequence[int] | int, duration_s: float) -> None:
        """Force taxels onto the lower ADC rail for ``duration_s`` (a front-end dropout)."""
        self._advance()
        idx = np.atleast_1d(np.asarray(taxels, dtype=np.int64))
        self._dropout_until[idx] = self._t + float(duration_s)
        self._last_raw = None                                   # the current sample changes now

    def truth(self) -> dict[str, Any]:
        """Ground truth at the current simulation time: ``contact`` ``[N]`` (press > 0.05 %),
        ``press_pct`` / ``artefact_pct`` ``[N]`` (SATS sign), per-finger ``penetration_rad``,
        ``q``, ``target``."""
        self._advance()
        return {"t": self._t, "contact": self._press < -0.05, "press_pct": self._press.copy(),
                "artefact_pct": self._art.copy(), "penetration_rad": dict(self._pen), "q": self._q.copy(),
                "target": self._target.copy()}

    def closure(self) -> dict[str, float]:
        """Current closure (rad) per finger."""
        self._advance()
        return {f: self._closure(self._q, f) for f in self.closure_joints}

    def __repr__(self) -> str:
        return (f"FakeRobotHand(dof={self.model.n_dof}, taxels={self.layout.n}, channels={self.n_channels}, "
                f"object={self.obj})")


# ─────────────────────────────────────────────────────────────── fake camera

class FakeCamera:
    """:class:`CameraInterface` for tests: frames on a ``1/rate_hz`` grid whose brightness pattern
    follows the hand closure (one vertical band per finger, length ∝ closure)."""

    def __init__(self, name: str, hand: FakeRobotHand | None = None, *, hw: tuple[int, int] = (24, 32),
                 rate_hz: float = 30.0, clock: Callable[[], float] | None = None, seed: int = 0):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.name = str(name)
        self.hand = hand
        self.hw = (int(hw[0]), int(hw[1]))
        self.rate_hz = float(rate_hz)
        self.clock = clock or (hand.clock if hand is not None else None)
        if self.clock is None:
            raise ValueError("FakeCamera needs a clock (or a hand to share its clock)")
        self._bg = np.random.default_rng(seed).integers(0, 40, size=(*self.hw, 3)).astype(np.uint8)
        self._last: tuple[float, np.ndarray] | None = None

    def _render(self) -> np.ndarray:
        H, W = self.hw
        img = self._bg.copy()
        if self.hand is None:
            return img
        cl = self.hand.closure()
        for i, f in enumerate(FINGERS):
            if f not in cl:
                continue
            x0 = int((i + 0.5) * W / (len(FINGERS) + 1))
            L = int(np.clip((1.0 - cl[f] / 1.6) * (H - 2), 1, H - 2))
            img[H - L:, x0:x0 + max(1, W // 12), :] = 200
        return img

    def read(self) -> tuple[float, np.ndarray]:
        now = float(self.clock())
        t = math.floor(now * self.rate_hz + 1e-9) / self.rate_hz
        if self._last is None or self._last[0] != t:
            self._last = (t, self._render())
        return self._last[0], self._last[1].copy()
