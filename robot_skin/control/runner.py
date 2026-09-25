"""Closed-loop deployment: :class:`PolicyRunner` (sensing → tactile processing → VTLA → safety → robot)
and :class:`DeploymentLogger` (every run is recorded as a RAW session that ``datasets.build`` can
re-ingest).

Rates (``docs/DEPLOYMENT.md``)::

    control_hz (= master / tactile rate, 200 Hz)   every tick: read joints + pressure, tactile
        processor step, interpolate toward the current joint target, SafetyFilter, send
    policy_hz (bundle, 20 Hz = every stride ticks)  read cameras, build the observation exactly like
        VTLADataset (make_observation + collate_vtla), policy.predict → chunk [H,A] (normalized)
        → unnormalize → make_absolute(state) → TemporalEnsembler.add / step → joint target
          hand_mano:   hand_action_fingertips → FingertipRetargeter.step → robot joints
          robot_joint: the action is the joint target (bundle action names → robot order)

Observation contract (identical to ``vtla.dataset.VTLADataset.__getitem__`` at a policy tick ``t``):

- ``tactile_values`` = :class:`~robot_skin.representation.TactileHistory` of the processor (the
  bundle's feature spec, history in master ticks — so ``control_hz`` should equal the bundle's
  ``source_hz``); ``contact`` = the bundle's contact rule; ``taxel_pos`` / ``taxel_nrm`` in the
  hand / robot base frame (pose function of the processor);
- ``proprio`` = the last ``obs_history`` action-space states at policy ticks (oldest first, edge
  padded), normalized by the bundle's proprio normalizer. For ``robot_joint`` the state is the
  measured joint vector; for ``hand_mano`` a robot has no human hand state, so the state is the
  **commanded** hand action (initialised from the proprio normalizer centre, i.e. the training
  mean) unless ``hand_state_fn(q_robot, commanded)`` estimates it (e.g.
  :class:`robot_skin.transfer.RobotToManoEstimator`);
- ``images`` = the bundle's eval transform of each camera's latest frame; before a camera's first
  frame the dataset's stand-in (eval transform of a zero image) with ``vision_valid`` False;
- ``instruction`` text, tokenized by the bundle's tokenizer.

``chunk[i]`` is the action at policy tick ``t + chunk_offset + i`` (ACT; Zhao et al.,
arXiv:2304.13705); the ensembler receives the chunk from the entry for ``t + 1`` on (offset 0: the
entry for "now" is dropped; offset > 1 warns), so ``TemporalEnsembler.step()`` right after
``add()`` yields the target for the *next* policy tick. Between policy ticks the joint target is
interpolated linearly from the last command so the position loop sees a smooth ramp
(``interpolate=False`` holds it instead). The ensembler, retargeter, hand-state estimator, processor
streams, rings, safety state and loop metrics are reset at every rollout start.

Clock: a :class:`~robot_skin.acquisition.sources.SimClock` (the fake hand's) is advanced by exactly
``1/control_hz`` per tick — deterministic, faster than real time; with a real clock the loop keeps
its rate by deadline scheduling and counts overruns (ticks that started more than one period late).
"""
from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .latency import LatencyMeter

__all__ = ["DeploymentLogger", "PolicyRunner", "joint_permutation", "initial_hand_state", "DEPLOY_LOG_NAME"]

log = logging.getLogger("robot_skin.control.runner")

DEPLOY_LOG_NAME = "deploy_log.npz"      # per-tick sidecar (not a manifest stream; ignored by datasets.build)


def joint_permutation(src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """Index array ``p`` with ``x_dst = x_src[p]``; raises ``KeyError`` naming missing joints."""
    pos = {str(n): i for i, n in enumerate(src)}
    missing = [n for n in dst if str(n) not in pos]
    if missing:
        raise KeyError(f"joints {missing} not found among {list(src)}")
    return np.asarray([pos[str(n)] for n in dst], dtype=np.int64)


def initial_hand_state(bundle: Any, init: str | Sequence[float] = "mean") -> np.ndarray:
    """Initial hand-action state ``[54]`` for ``hand_mano`` deployment: ``mean`` = the proprio
    normalizer centre (the state that normalizes to 0 — the training mean for ``std``) with its 6D
    rotation re-orthonormalised; ``flat`` = the same wrist pose with a flat hand (zero finger pose);
    or an explicit ``[54]`` vector."""
    from ..action.space import FINGERS_AA, HAND_MANO_DIM, hand_action_from_arrays, hand_action_to_arrays

    if not isinstance(init, str):
        a = np.asarray(init, dtype=np.float32).reshape(-1)
        if a.shape != (HAND_MANO_DIM,):
            raise ValueError(f"initial hand state must be [{HAND_MANO_DIM}], got {a.shape}")
        return a
    if init not in ("mean", "flat"):
        raise ValueError("hand_state_init must be 'mean', 'flat' or a [54] vector")
    pn = bundle.proprio_normalizer
    c = (np.zeros(HAND_MANO_DIM, np.float32) if pn is None
         else np.asarray(pn.unnormalize(np.zeros(pn.dim, np.float32)), np.float32))
    if pn is None:
        c[3], c[7] = 1.0, 1.0                                   # identity rotation 6D (first two columns)
    h = hand_action_to_arrays(c)
    a = np.asarray(hand_action_from_arrays(h.global_orient, h.finger_pose, h.wrist_pos), np.float32)
    if init == "flat":
        a[FINGERS_AA] = 0.0
    return a


# ─────────────────────────────────────────────────────────────── logger

class _PushSource:
    """Minimal :class:`~robot_skin.acquisition.sources.StreamSource` fed by the runner."""

    def __init__(self, name: str, kind: str, rate_hz: float | None = None, info: Mapping[str, Any] | None = None):
        self.name, self.kind = str(name), str(kind)
        self.rate_hz = None if rate_hz is None else float(rate_hz)
        self._info = dict(info or {})
        self._q: list[tuple[float, dict]] = []

    def start(self, clock: Any = None) -> None:
        self._q = []

    def push(self, t: float, sample: Mapping[str, Any]) -> None:
        self._q.append((float(t), dict(sample)))

    def poll(self) -> list[tuple[float, dict]]:
        out, self._q = self._q, []
        return out

    def stop(self) -> None:
        pass

    def info(self) -> dict:
        return dict(self._info)


class DeploymentLogger:
    """Record a deployment run in the RAW session format (``acquisition.manifest``): ``pressure.npz``
    (raw channel order), ``joint_state.npz`` (q, qd when the driver measures it, names; one row per
    new sample time), ``camera_<name>/`` (new frames only),
    ``events.jsonl`` (phases ``baseline`` / ``calibration`` → ``no_contact`` segments, ``rollout`` →
    ``task``; the instruction; safety markers; the success verdict), ``session.json`` (kind robot,
    ``layout: layout.yaml``, ``meta.urdf: robot.urdf``, ``meta.deployment``), ``layout.yaml``,
    ``robot.urdf`` and the per-tick sidecar ``deploy_log.npz`` (targets, commands, levels, policy
    ticks). ``datasets.build.preprocess_session`` ingests it like any robot session — deployment
    data can be relabelled and reused (e.g. success / failure rollouts for preference learning).

    Built on :class:`robot_skin.acquisition.recorder.Recorder` (synchronous mode) so the files are
    byte-for-byte the recorder's formats.
    """

    def __init__(self, session_dir: str | Path, *, layout: Any, joint_names: Sequence[str], clock: Callable[[], float],
                 cameras: Sequence[str] = (), dataset: str = "other", subject: str = "robot",
                 session_id: str | None = None, task_id: str = "deploy", instruction: str = "",
                 urdf_xml: str | None = None, camera_format: str = "auto", overwrite: bool = False,
                 meta: Mapping[str, Any] | None = None, pressure_hz: float | None = None,
                 joint_hz: float | None = None, camera_hz: float | None = None):
        import yaml

        from common.layouts import Layout, load_layout

        from ..acquisition.manifest import SessionManifest
        from ..acquisition.recorder import Recorder

        self.session_dir = Path(session_dir)
        self.layout = layout if isinstance(layout, Layout) else load_layout(layout)
        self.joint_names = [str(n) for n in joint_names]
        self.clock = clock
        m = dict(meta or {})
        man = SessionManifest(kind="robot", layout="layout.yaml", dataset=dataset, subject=str(subject),
                              task={"task_id": task_id, "instruction": instruction or None, "object": None,
                                    "success": None},
                              notes="robot_skin.control deployment log", meta={"deployment": m})
        if session_id:
            man.session_id = str(session_id)
        if urdf_xml:
            man.meta["urdf"] = "robot.urdf"
        self.manifest = man
        self._src = {"pressure": _PushSource("pressure", "pressure", pressure_hz),
                     "joint_state": _PushSource("joint_state", "joint_state", joint_hz, {"names": self.joint_names})}
        for c in cameras:
            self._src[f"camera_{c}"] = _PushSource(f"camera_{c}", "camera", camera_hz)
        sim = type(clock).__name__ == "SimClock"
        self.recorder = Recorder(list(self._src.values()), self.session_dir, man, clock=clock,
                                 camera_format=camera_format, overwrite=overwrite)
        self.recorder.start(threaded=False)
        self._sim = sim
        (self.session_dir / "layout.yaml").write_text(yaml.safe_dump(self.layout.to_dict(units="m"), sort_keys=False))
        if urdf_xml:
            (self.session_dir / "robot.urdf").write_text(urdf_xml)
        self._last_cam_t: dict[str, float] = {}
        self._last_t: dict[str, float] = {}
        self._has_qd: bool | None = None
        self._ticks: dict[str, list] = {}
        self._policy: dict[str, list] = {}
        self._open: list[str] = []
        self.closed = False

    # ── streams ───────────────────────────────────────────────────────────
    def _new_sample(self, stream: str, t: float) -> bool:
        """A driver read twice within one sample period returns the same sample: record it once."""
        if self._last_t.get(stream) == float(t):
            return False
        self._last_t[stream] = float(t)
        return True

    def log_pressure(self, t: float, raw: np.ndarray) -> None:
        if self._new_sample("pressure", t):
            self._src["pressure"].push(t, {"raw": np.asarray(raw, dtype=np.float64)})

    def log_joint_state(self, t: float, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        """``qd`` only when the driver measures it (the first sample fixes the schema; a later
        missing ``qd`` is NaN — never a fabricated zero velocity that ``qd.source: file`` would use)."""
        if not self._new_sample("joint_state", t):
            return
        s = {"q": np.asarray(q, dtype=np.float64)}
        if self._has_qd is None:
            self._has_qd = qd is not None
        if self._has_qd:
            s["qd"] = np.full_like(s["q"], np.nan) if qd is None else np.asarray(qd, dtype=np.float64)
        self._src["joint_state"].push(t, s)

    def log_camera(self, name: str, t: float, frame: np.ndarray | None) -> None:
        key = f"camera_{name}"
        if frame is None or key not in self._src or self._last_cam_t.get(name) == float(t):
            return
        self._last_cam_t[name] = float(t)
        self._src[key].push(t, {"frame": np.asarray(frame, dtype=np.uint8)})

    def flush(self) -> None:
        self.recorder.step()

    # ── events ────────────────────────────────────────────────────────────
    def phase_start(self, name: str, *, contact: str = "any", labels: Sequence[str] | None = None,
                    **value: Any) -> None:
        self.recorder.phase_start(name, {"contact": contact, "labels": list(labels or []), **value})
        self._open.append(name)

    def phase_end(self, name: str) -> None:
        if name in self._open:
            self.recorder.phase_end(name)
            self._open.remove(name)

    def marker(self, name: str, value: Any = None) -> None:
        self.recorder.marker(name, value)

    def instruction(self, text: str) -> None:
        self.recorder.instruction(text)

    def set_baseline(self, baseline_raw_channels: np.ndarray) -> None:
        """Per-channel raw baseline measured at start-up (``manifest.baseline``, a preprocessing
        fallback)."""
        self.manifest.baseline = [float(v) for v in np.asarray(baseline_raw_channels, dtype=np.float64).reshape(-1)]

    def log_tick(self, **rec: Any) -> None:
        for k, v in rec.items():
            self._ticks.setdefault(k, []).append(np.asarray(v))

    def log_policy(self, **rec: Any) -> None:
        for k, v in rec.items():
            self._policy.setdefault(k, []).append(np.asarray(v))

    # ── close ─────────────────────────────────────────────────────────────
    def close(self, *, success: bool | None = None, meta: Mapping[str, Any] | None = None):
        """Close open phases, log the verdict, write every file; returns the ``SessionManifest``."""
        if self.closed:
            return self.manifest
        self.flush()
        for name in list(self._open):
            self.phase_end(name)
        if success is not None:
            self.recorder.success(bool(success))
            self.manifest.task["success"] = bool(success)
        if meta:
            self.manifest.meta["deployment"].update(dict(meta))
        side = {}
        t0 = float(self.recorder.t0 or 0.0)
        for pre, rows in (("tick_", self._ticks), ("policy_", self._policy)):
            for k, v in rows.items():
                try:
                    a = np.stack(v)
                except ValueError:
                    continue
                if k == "t":
                    a = a.astype(np.float64) - t0                    # session clock, like the streams
                side[pre + k] = a
        if side:
            np.savez(self.session_dir / DEPLOY_LOG_NAME, **side)
        man = self.recorder.stop()
        self.closed = True
        return man


# ─────────────────────────────────────────────────────────────── runner

class PolicyRunner:
    """Run a VTLA policy bundle on a robot hand (module docstring).

    Args:
        robot: a :class:`~robot_skin.control.interfaces.RobotHandInterface`.
        bundle: :class:`~robot_skin.control.bundle.PolicyBundle` (or a path → ``load_policy_bundle``).
        processor: :class:`~robot_skin.control.online.OnlineTactileProcessor` for the robot's skin
            (``raw_order="channel"``); its feature spec / contact rule must match the bundle.
        cameras: ``{name: CameraInterface}`` or a list (names from ``.name``); every bundle camera
            is required.
        retargeter: :class:`~robot_skin.action.FingertipRetargeter` (required for ``hand_mano``).
        safety: :class:`~robot_skin.control.safety.SafetyFilter` (default: limits only).
        control_hz / policy_hz: default the bundle's ``source_hz`` / ``policy_hz``.
        instruction: task text (tokenized with the bundle tokenizer).
        ensembler: :class:`~robot_skin.action.TemporalEnsembler` (default: horizon, action dim and
            ``k`` of the bundle).
        clock: host clock (default ``robot.clock`` if present, else ``time.monotonic``).
        logger: :class:`DeploymentLogger` or ``None``.
        hand_state_fn: ``(q in the retargeter's joint order, commanded_hand_action) → hand_action``
            state estimate for ``hand_mano`` proprio (default: the commanded action), e.g.
            :class:`robot_skin.transfer.RobotToManoEstimator`.
        hand_state_init: ``mean`` | ``flat`` | ``[54]`` (:func:`initial_hand_state`).
        interpolate: linear joint-target ramp between policy ticks.
        seed: generator seed of the flow head's sampling noise (deterministic rollouts).
        stop_on_estop: end the rollout when the safety e-stop latches.
    """

    def __init__(self, robot: Any, bundle: Any, processor: Any, *, cameras: Any = None, retargeter: Any = None,
                 safety: Any = None, control_hz: float | None = None, policy_hz: float | None = None,
                 instruction: str = "", ensembler: Any = None, clock: Callable[[], float] | None = None,
                 logger: DeploymentLogger | None = None,
                 hand_state_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 hand_state_init: str | Sequence[float] = "mean", interpolate: bool = True, device: Any = None,
                 seed: int = 0, allow_bootstrap: bool = False, stop_on_estop: bool = True,
                 n_steps: int | None = None):
        import torch

        from ..action.chunking import TemporalEnsembler
        from .bundle import PolicyBundle, load_policy_bundle
        from .interfaces import check_camera, check_robot
        from .safety import SafetyFilter

        check_robot(robot)
        self.robot = robot
        self.bundle = bundle if isinstance(bundle, PolicyBundle) else load_policy_bundle(bundle, device=device or "cpu")
        self.bundle.check_deployable(allow_bootstrap=allow_bootstrap)
        b = self.bundle
        self.processor = processor
        if processor.raw_order != "channel":
            raise ValueError("the runner feeds read_pressure() output: use an OnlineTactileProcessor with "
                             "raw_order='channel'")
        if b.uses_tactile:
            if processor.feature_spec != b.feature_spec:
                raise ValueError(f"processor feature spec {processor.feature_spec} != bundle {b.feature_spec}")
            if processor.contact_rule != b.contact_rule:
                raise ValueError(f"processor contact rule {processor.contact_rule!r} != bundle {b.contact_rule!r}")
        self.device = torch.device(device) if device is not None else next(b.policy.parameters()).device
        b.policy.to(self.device).eval()
        self.control_hz = float(control_hz or b.source_hz)
        self.policy_hz = float(policy_hz or b.policy_hz)
        self.dt = 1.0 / self.control_hz
        ratio = self.control_hz / self.policy_hz
        self.policy_every = max(1, int(round(ratio)))
        if abs(ratio - self.policy_every) > 1e-6:
            warnings.warn(f"control_hz / policy_hz = {ratio:.3f} is not an integer: the policy runs every "
                          f"{self.policy_every} ticks ({self.control_hz / self.policy_every:g} Hz)", stacklevel=2)
        if abs(self.control_hz - float(processor.hz)) > 1e-6:
            warnings.warn(f"control_hz {self.control_hz:g} differs from the tactile processor's master rate "
                          f"{processor.hz:g}: tactile history / qd / FSM timing no longer match training", stacklevel=2)
        if abs(self.policy_every * b.policy_hz - self.control_hz) > 1e-6 * self.control_hz and policy_hz is None:
            warnings.warn(f"policy runs every {self.policy_every} ticks at control_hz {self.control_hz:g}, the bundle "
                          f"was trained at {b.policy_hz:g} Hz (stride {b.stride} of {b.source_hz:g} Hz)", stacklevel=2)

        # cameras
        cams = {} if cameras is None else (dict(cameras) if isinstance(cameras, Mapping)
                                           else {c.name: c for c in cameras})
        for c in cams.values():
            check_camera(c)
        missing = [c for c in b.cameras if c not in cams]
        if missing:
            raise ValueError(f"the policy needs cameras {list(b.cameras)}; missing {missing}")
        self.cameras = cams
        self.eval_transform = b.eval_transform
        if b.cameras and self.eval_transform is None:
            from ..vision.transforms import EvalTransform

            enc = b.policy.vision_encoder
            self.eval_transform = EvalTransform(None, mean=getattr(enc, "mean", None), std=getattr(enc, "std", None))

        # joint orders
        R = [str(n) for n in robot.joint_names]
        self.robot_joint_names = R
        self._p_proc = (None if not getattr(processor, "joint_names", None)
                        else joint_permutation(R, processor.joint_names))
        self.kind = b.action_kind
        if self.kind == "robot_joint":
            names = [str(n) for n in (b.action_spec.names or [])]
            if names and all(n in R for n in names):
                self._p_act = joint_permutation(R, names)          # action joints by name
            elif b.action_dim == len(R):
                if names and names != [f"j{i}" for i in range(len(names))]:
                    warnings.warn(f"bundle action joints {names} are not the robot's {R}: assuming the robot's "
                                  "order", stacklevel=2)
                self._p_act = np.arange(len(R))
            else:
                raise ValueError(f"robot_joint policy acts on {b.action_dim} joints {names}, robot has {len(R)} "
                                 f"{R}; the bundle's action_spec needs the robot's joint names")
            self.retargeter = None
        elif self.kind == "hand_mano":
            if retargeter is None:
                raise ValueError("a hand_mano policy needs a FingertipRetargeter (human hand action → robot joints)")
            self.retargeter = retargeter
            rn = getattr(retargeter, "joint_names", None)
            self._p_ret = (np.arange(len(R)) if rn is None else joint_permutation(list(rn), R))
            self._p_ret_inv = np.argsort(self._p_ret)             # robot order → retargeter order
            if rn is None and int(retargeter.dof) != len(R):
                raise ValueError(f"retargeter has {retargeter.dof} joints, robot {len(R)}")
            self.hand_state_init = initial_hand_state(b, hand_state_init)
        else:
            raise ValueError(f"unsupported action kind {self.kind!r}")
        self.hand_state_fn = hand_state_fn

        self.safety = safety if safety is not None else SafetyFilter(robot.lower, robot.upper, dt=self.dt,
                                                                     joint_names=R)
        if getattr(self.safety, "estop_callback", None) is None and hasattr(robot, "estop"):
            self.safety.estop_callback = lambda reason, t: robot.estop()
        self.ensembler = ensembler if ensembler is not None else TemporalEnsembler(b.horizon, b.action_dim,
                                                                                 k=b.ensemble_k)
        # chunk[i] was trained as the action at policy tick t + (chunk_offset + i); the ensembler's
        # step right after add() must yield the target for the *next* policy tick (t + 1)
        off = int(b.chunk_offset)
        self._chunk_skip = max(0, 1 - off)
        if off > 1:
            warnings.warn(f"bundle chunk_offset {off}: chunk[0] is the action {off} policy ticks ahead; it is executed "
                          f"at the next tick (commands lead the training timing by {off - 1} tick(s))", stacklevel=2)
        if b.horizon - self._chunk_skip < 1:
            raise ValueError("chunk_offset 0 with horizon 1: the chunk holds no future action to execute")
        self.instruction = str(instruction or "")
        if not self.instruction and b.policy.text_encoder is not None:
            warnings.warn("no instruction given to a language-conditioned policy", stacklevel=2)
        rc = getattr(robot, "clock", None)
        if clock is None and rc is not None:
            clock = rc
        if clock is None:
            from ..acquisition.sources import MonotonicClock

            clock = MonotonicClock()
        self.clock = clock
        self.sim = type(clock).__name__ == "SimClock"
        self.logger = logger
        self.interpolate = bool(interpolate)
        self.seed = int(seed)
        self.stop_on_estop = bool(stop_on_estop)
        self.n_steps = n_steps
        self.tick_meter = LatencyMeter("tick")
        self.infer_meter = LatencyMeter("inference")
        self.retarget_meter = LatencyMeter("retarget")
        self.n_ticks = self.n_policy_ticks = self.overruns = 0
        self.n_contact_ticks = 0
        self._wall0 = self._t_start = None
        self._deadline: float | None = None
        self._rollout = False

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def _logging(self) -> bool:
        return self.logger is not None and not self.logger.closed

    def _read(self) -> tuple[float, np.ndarray, np.ndarray | None, float, np.ndarray]:
        tq, q, qd = self.robot.read_state()
        tp, raw = self.robot.read_pressure()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qd = None if qd is None else np.asarray(qd, dtype=np.float64).reshape(-1)
        if self._logging:
            self.logger.log_joint_state(tq, q, qd)
            self.logger.log_pressure(tp, raw)
        return float(tq), q, qd, float(tp), np.asarray(raw, dtype=np.float64)

    def _proc_q(self, q: np.ndarray) -> np.ndarray:
        return q if self._p_proc is None else q[self._p_proc]

    def _advance(self) -> None:
        if self.sim:
            self.clock.advance(self.dt)
            return
        self._deadline += self.dt
        delay = self._deadline - float(self.clock())
        if delay > 0:
            time.sleep(delay)
        elif -delay > self.dt:
            self.overruns += 1
            if -delay > 10 * self.dt:                         # far behind: re-synchronise, do not burst
                self._deadline = float(self.clock())

    def _hold_tick(self, q_hold: np.ndarray) -> tuple[float, np.ndarray, np.ndarray | None, float, np.ndarray]:
        r = self._read()
        self.robot.send_joint_targets(q_hold)
        if self._logging:
            self.logger.flush()
        return r

    # ── start-up ──────────────────────────────────────────────────────────
    def startup(self, baseline_s: float = 1.0, calib_s: float = 0.0, *, calib_kw: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
        """Hold the hand still (no contact!) for ``baseline_s``: capture the tactile raw baseline
        (as preprocessing: median over the no-contact window, rail samples excluded). If the
        processor has no calibrator and ``calib_s > 0``, keep holding for ``calib_s`` and fit a
        bring-up calibrator on the residuals (:func:`~robot_skin.control.online.startup_calibrator`).
        Logged as ``baseline`` / ``calibration`` phases (``no_contact`` segments)."""
        from .online import startup_calibrator

        for fn in ("start",):
            if hasattr(self.robot, fn):
                getattr(self.robot, fn)()
        self._deadline = float(self.clock())
        _, q0, _, _, _ = self._read()
        self.safety.reset(q0)
        q_hold = self.safety.q_last.copy()
        proc = self.processor
        proc.begin_baseline()
        n_b = max(2, int(round(float(baseline_s) * self.control_hz)))
        raws = []
        if self._logging:
            self.logger.phase_start("baseline", contact="none", labels=["no_contact"], kind="static",
                                    note="start-up tactile baseline (hand still, no contact)")
        for _ in range(n_b):
            _, _, _, tp, raw = self._hold_tick(q_hold)
            proc.add_baseline_sample(raw, tp)
            raws.append(raw)
            self._advance()
        base = proc.finish_baseline(baseline_s)
        if proc.dead.any():
            warnings.warn(f"dead tactile channels (baseline ≤ 0) on taxels {np.flatnonzero(proc.dead).tolist()}: they "
                          "read SATURATED on every tick, so a tactile stop on SATURATED keeps their kinematic chains "
                          "from closing (fail safe) — repair the channel or drop SATURATED from safety levels",
                          stacklevel=2)
        if self._logging:
            self.logger.phase_end("baseline")
            ch = np.median(np.stack(raws), axis=0)                  # per channel; taxels: the processor's
            ch[np.asarray(proc.layout.channels)] = base
            self.logger.set_baseline(ch)
        warm_ms = self.warmup()
        self._deadline = float(self.clock())                      # the warm-up is not a loop overrun
        out: dict[str, Any] = {"baseline_raw": base.tolist(), "baseline_samples": len(raws),
                               "dead_taxels": [int(i) for i in np.flatnonzero(proc.dead)], "calibrator": "given",
                               "warmup_ms": warm_ms}
        if proc.calibrator is None:
            if calib_s and calib_s > 0:
                if self._logging:
                    self.logger.phase_start("calibration", contact="none", labels=["no_contact"], kind="static",
                                            note="bring-up residual calibration (hand still, no contact)")
                proc.reset()
                res, sat, lv = [], [], []
                for _ in range(max(4, int(round(float(calib_s) * self.control_hz)))):
                    _, q, qd, tp, raw = self._hold_tick(q_hold)
                    fr = proc.step(raw, self._proc_q(q), qd=None if qd is None else self._proc_q(qd), t=tp)
                    res.append(fr.residual)
                    sat.append(fr.saturated)
                    if fr.baseline_logvar is not None:
                        lv.append(fr.baseline_logvar)
                    self._advance()
                cal = startup_calibrator(np.stack(res), np.stack(sat), np.stack(lv) if lv else None,
                                         **dict(calib_kw or {}))
                proc.set_calibrator(cal)
                out["calibrator"] = "startup"
                if self._logging:
                    self.logger.phase_end("calibration")
            else:
                warnings.warn("tactile processor without a calibrator: z is NaN and only saturation reaches the "
                              "policy (pass the contact stage's calibrator.json, or startup calib_s > 0)",
                              stacklevel=2)
                out["calibrator"] = None
        return out

    def warmup(self, n: int = 2) -> float:
        """Run ``n`` dummy inferences (lazy CUDA / kernel initialisation — the first call can take
        hundreds of ms and would otherwise stall the first control ticks). Returns the time (ms);
        the robot keeps holding meanwhile (call it while the hand is still, e.g. from
        :meth:`startup`)."""
        import torch

        from .latency import _to_device, example_batch

        b = self.bundle
        batch = _to_device(example_batch(b, n_taxels=self.processor.layout.n, instruction=self.instruction or "x"),
                           self.device)
        w0 = time.perf_counter()
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        for _ in range(max(0, int(n))):
            b.policy.predict(batch, self.n_steps, generator=gen)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - w0) * 1e3

    # ── rollout ───────────────────────────────────────────────────────────
    def begin_rollout(self) -> None:
        """Reset every stream (processor, ensembler, retargeter, hand-state estimator, proprio /
        camera rings, safety state from the current measured position — a latched e-stop is
        cleared here, with a warning: starting a rollout is the operator's explicit restart) and
        open the ``rollout`` phase."""
        import torch

        _, q, _, _, _ = self._read()
        self.processor.reset(keep_baseline=True)
        self.ensembler.reset()
        self._deadline = float(self.clock())                  # do not carry start-up lag into the rollout
        if self.safety.estopped:
            log.warning("begin_rollout: clearing the latched e-stop (%s)", self.safety.estop_reason)
        self.safety.reset(q)
        q_cmd = self.safety.q_last.copy()
        if self.retargeter is not None:
            self.retargeter.reset(q_cmd[self._p_ret_inv])
        if self.hand_state_fn is not None and hasattr(self.hand_state_fn, "reset"):
            self.hand_state_fn.reset()                        # e.g. RobotToManoEstimator warm start
        self._hand_state = None if self.kind != "hand_mano" else self.hand_state_init.copy()
        self._proprio: deque = deque(maxlen=self.bundle.obs_history)
        self._frames: dict[str, deque] = {c: deque(maxlen=self.bundle.obs_history) for c in self.bundle.cameras}
        self._target_prev = q_cmd.copy()
        self._target_next = q_cmd.copy()
        self._since_policy = 0
        self._gen = torch.Generator(device=self.device).manual_seed(self.seed)
        # metrics() describes one rollout: restart the loop counters and latency meters
        self.n_ticks = self.n_policy_ticks = self.overruns = self.n_contact_ticks = 0
        for m in (self.tick_meter, self.infer_meter, self.retarget_meter):
            m.reset()
        self._t_end = self._wall_end = None
        self._t_start = float(self.clock())
        self._wall0 = time.perf_counter()
        self._rollout = True
        if self._logging:
            self.logger.phase_start("rollout", contact="any", labels=["task"], policy_hz=self.policy_hz,
                                    control_hz=self.control_hz)
            if self.instruction:
                self.logger.instruction(self.instruction)

    def _state(self, q: np.ndarray) -> np.ndarray:
        if self.kind == "robot_joint":
            return q[self._p_act].astype(np.float32)
        if self.hand_state_fn is not None:
            q_ret = q[self._p_ret_inv]
            return np.asarray(self.hand_state_fn(q_ret, self._hand_state), dtype=np.float32).reshape(-1)
        return self._hand_state.astype(np.float32)

    def _images(self) -> tuple[dict | None, dict | None, dict[str, float]]:
        import torch

        b = self.bundle
        if not b.cameras:
            return None, None, {}
        k = b.obs_history
        imgs, valid, stamps = {}, {}, {}
        for c in b.cameras:
            t_c, frame = self.cameras[c].read()
            stamps[f"camera_{c}"] = float(t_c)
            if self._logging:
                self.logger.log_camera(c, t_c, frame)
            ring = self._frames[c]
            if frame is None:
                x, ok = (ring[-1][0] if ring else self._blank_image()), False
            else:
                x, ok = self.eval_transform(np.asarray(frame, dtype=np.uint8)[None])[0], True
            if x is None:
                continue                                              # no frame yet, size unknown: raise below
            ring.append((x, ok))
            while len(ring) < k:
                ring.appendleft(ring[0])
            xs = torch.stack([r[0] for r in ring])
            imgs[c] = xs[0] if k == 1 else xs
            valid[c] = np.asarray([r[1] for r in ring], dtype=bool)
        if len(imgs) != len(b.cameras):
            raise RuntimeError(f"no frame yet from cameras {[c for c in b.cameras if c not in imgs]} and the eval "
                               "transform has no fixed output size for a blank stand-in")
        return imgs, valid, stamps

    def _blank_image(self) -> Any:
        """Stand-in before a camera's first frame, as ``VTLADataset`` builds it (frame index −1): the
        eval transform of an all-zero uint8 image (constant, so its input size is irrelevant),
        flagged ``vision_valid = False``. ``None`` when the transform has no fixed output size."""
        out = getattr(self.eval_transform, "out_size", None)
        if not out:
            return None
        return self.eval_transform(np.zeros((1, int(out[0]), int(out[1]), 3), np.uint8))[0]

    def _policy_tick(self, t: float, q: np.ndarray, frame: Any) -> tuple[np.ndarray, dict[str, float]]:
        import torch

        from ..action.space import make_absolute
        from ..vtla.dataset import collate_vtla, make_observation

        b = self.bundle
        state = self._state(q)
        self._proprio.append(state)
        while len(self._proprio) < b.obs_history:
            self._proprio.appendleft(self._proprio[0])
        images, valid, stamps = self._images()
        N = self.processor.layout.n
        values = frame.features if frame.features is not None else np.zeros((N, 0), np.float32)
        obs = make_observation(proprio_states=np.stack(self._proprio), tactile_values=values, taxel_pos=frame.pos,
                               taxel_nrm=frame.nrm, contact=frame.contact, instruction=self.instruction,
                               proprio_normalizer=b.proprio_normalizer, images=images, vision_valid=valid)
        batch = collate_vtla([obs], b.tokenizer)
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else
                     ({c: x.to(self.device) for c, x in v.items()} if isinstance(v, Mapping) else v))
                 for k, v in batch.items()}
        sync = (lambda: torch.cuda.synchronize(self.device)) if self.device.type == "cuda" else None
        if sync is not None:
            sync()
        w0 = time.perf_counter()
        chunk_n = b.policy.predict(batch, self.n_steps, generator=self._gen)[0].float().cpu().numpy()
        if sync is not None:
            sync()
        infer_ms = (time.perf_counter() - w0) * 1e3
        self.infer_meter.record(infer_ms)
        rel = np.asarray(b.action_normalizer.unnormalize(chunk_n), dtype=np.float32)
        chunk = np.asarray(make_absolute(rel, state, b.action_spec, b.rel_mode), dtype=np.float64)
        self.ensembler.add(chunk[self._chunk_skip:])
        a = self.ensembler.step()
        q_target = self.safety.q_last.copy()
        if self.kind == "robot_joint":
            q_target[self._p_act] = a
        else:
            from ..action.retarget import hand_action_fingertips

            with self.retarget_meter.time():
                tips = hand_action_fingertips(a)
                q_ret = np.asarray(self.retargeter.step(tips), dtype=np.float64).reshape(-1)
            q_target = q_ret[self._p_ret]
            self._hand_state = a.astype(np.float32)
        self.n_policy_ticks += 1
        if self._logging:
            self.logger.log_policy(t=t, action=a.astype(np.float32), q_target=q_target.astype(np.float32),
                                   state=state, inference_ms=infer_ms)
        return q_target, stamps

    def step(self) -> Any:
        """One control tick; returns the processor's :class:`~robot_skin.control.online.TactileFrame`."""
        if not self._rollout:
            raise RuntimeError("call begin_rollout() (or run()) first")
        w0 = time.perf_counter()
        t = float(self.clock())
        tq, q, qd, tp, raw = self._read()
        frame = self.processor.step(raw, self._proc_q(q), qd=None if qd is None else self._proc_q(qd), t=tp)
        stamps: dict[str, float] = {"pressure": tp, "joint_state": tq}
        if self._since_policy % self.policy_every == 0:
            self._target_prev = self.safety.q_last.copy()
            self._target_next, cam_stamps = self._policy_tick(t, q, frame)
            stamps.update(cam_stamps)
            self._since_policy = 0
        elif self._logging:                         # record camera streams at their own rate
            for c, cam in self.cameras.items():
                t_c, img = cam.read()
                self.logger.log_camera(c, t_c, img)
        j = self._since_policy
        if self.interpolate:
            f = min(1.0, (j + 1) / self.policy_every)
            q_des = self._target_prev + f * (self._target_next - self._target_prev)
        else:
            q_des = self._target_next
        self._since_policy += 1
        n_ev = len(self.safety.events)
        q_cmd = self.safety.filter(q_des, q, t=t, level=frame.level, stamps=stamps)
        self.robot.send_joint_targets(q_cmd)
        self.n_ticks += 1
        self.n_contact_ticks += int(frame.any_contact)
        if self._logging:
            for ev in self.safety.events[n_ev:]:
                self.logger.marker("safety", ev.to_dict())
            self.logger.log_tick(t=t, q=q.astype(np.float32), q_des=q_des.astype(np.float32),
                                 q_cmd=q_cmd.astype(np.float32), level=frame.level.astype(np.int8),
                                 contact=frame.contact, stop_active=self.safety.stop_active, stale=self.safety.stale)
            self.logger.flush()
        self.tick_meter.record((time.perf_counter() - w0) * 1e3)
        return frame

    def end_rollout(self) -> None:
        if self._logging and self._rollout:
            self.logger.phase_end("rollout")
        self._rollout = False
        self._t_end = float(self.clock())
        self._wall_end = time.perf_counter()

    def run(self, duration_s: float, *, startup: bool = True, baseline_s: float = 1.0, calib_s: float = 0.0,
            success: bool | None = None, close: bool = True) -> dict[str, Any]:
        """Start-up (optional) → rollout for ``duration_s`` → metrics (and close the log)."""
        info = self.startup(baseline_s, calib_s) if startup else {}
        self.begin_rollout()
        n = int(round(float(duration_s) * self.control_hz))
        for _ in range(n):
            self.step()
            self._advance()
            if self.safety.estopped and self.stop_on_estop:
                log.warning("e-stop latched (%s): rollout stopped", self.safety.estop_reason)
                break
        self.end_rollout()
        m = self.metrics()
        m["startup"] = info
        if self._logging and close:
            man = self.logger.close(success=success, meta={"metrics": {k: v for k, v in m.items()
                                                                       if k not in ("safety", "startup")},
                                                           "safety": self.safety.summary()})
            m["session_dir"] = str(self.logger.session_dir)
            m["session_id"] = man.session_id
        if hasattr(self.robot, "stop"):
            self.robot.stop()
        return m

    def metrics(self) -> dict[str, Any]:
        """Loop / latency / safety summary of the last rollout."""
        t_end = getattr(self, "_t_end", None)
        t_end = float(self.clock()) if t_end is None else t_end            # (a SimClock may start at 0.0)
        w_end = getattr(self, "_wall_end", None)
        w_end = time.perf_counter() if w_end is None else w_end
        sim_s = max(t_end - (t_end if self._t_start is None else self._t_start), 0.0)
        wall_s = max(w_end - (w_end if self._wall0 is None else self._wall0), 1e-9)
        tick, inf = self.tick_meter.summary(), self.infer_meter.summary()
        out = {
            "n_ticks": self.n_ticks, "n_policy_ticks": self.n_policy_ticks, "control_hz": self.control_hz,
            "policy_hz": self.control_hz / self.policy_every, "policy_every": self.policy_every,
            "duration_s": sim_s, "wall_s": wall_s, "clock": "sim" if self.sim else "host",
            "loop_hz": self.n_ticks / sim_s if sim_s > 0 else float("nan"),
            "loop_hz_wall": self.n_ticks / wall_s,
            "tick_ms": tick, "inference_ms": inf, "latency_p50_ms": inf["p50_ms"], "latency_p95_ms": inf["p95_ms"],
            "tick_p50_ms": tick["p50_ms"], "tick_p95_ms": tick["p95_ms"], "overruns": self.overruns,
            "budget_ms": 1e3 / self.control_hz,
            "tick_over_budget_frac": (float(np.mean(np.asarray(self.tick_meter.samples) > 1e3 / self.control_hz))
                                      if self.tick_meter.samples else float("nan")),
            "contact_frac": self.n_contact_ticks / self.n_ticks if self.n_ticks else float("nan"),
            "safety": self.safety.summary(), "safety_counts": dict(self.safety.counts),
            "estop": bool(self.safety.estopped), "action_kind": self.kind,
        }
        if self.retarget_meter.samples:
            out["retarget_ms"] = self.retarget_meter.summary()
        return out

    def __repr__(self) -> str:
        return (f"PolicyRunner(kind={self.kind!r}, control_hz={self.control_hz:g}, policy_every={self.policy_every}, "
                f"cameras={list(self.cameras)}, logger={self.logger is not None})")

