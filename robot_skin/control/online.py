"""Online tactile processing: raw counts → ΔS → baseline residual → calibrated z / levels → features.

:class:`OnlineTactileProcessor` is the per-tick (master clock, 200 Hz) mirror of the offline chain
that produced the training data, step for step, with the *same* functions and stored parameters::

    raw[C] ─ layout.by_channel ─ relative_change(raw, baseline_raw) ─ saturation_mask ────────┐
      (datasets.build: ΔS % SATS sign — press → negative; rails / |ΔS| ≥ max_abs_pct)          │
    q[D] ─ CausalJointVelocity (= datasets.build.joint_velocity, causal Savitzky–Golay) ─ qd  │
    q ─ pose_fn (glove: MANO skeleton, go = 0; robot: URDF FK) ─ taxel pos / nrm (hand frame)  │
    (q, qd, pos, nrm) ─ CausalBaselineStream (baseline.temporal) ─ mean, logvar ───────────────┤
    residual = ΔS − mean ─ SaturationFSM (calibrator.fsm, optional) ─ ResidualCalibrator ──────┤
      z = calibrated press-positive z-score, level = NONE / WEAK / STRONG / SATURATED          │
    TactileHistory(feature_spec) → tactile_value_features [N, F]; contact = level rule;        │
    optional CausalDetectorStream + HysteresisFilter (contact.detector / .hysteresis) ─────────┘

so that a policy trained on ``derived/residual_z`` / ``contact_level`` sees bit-compatible inputs
at deployment (``test_control.py`` replays a processed synthetic episode through the processor and
checks every intermediate against ``baseline.temporal.predict_episode``,
``contact.calibration.residual_levels`` and ``TactileFeatureSpec.from_arrays``). Remaining
unavoidable train/deploy differences, by design:

- **Resampling**: offline, raw pressure is linearly interpolated onto the master grid and a frame
  is also flagged saturated when a *bracketing* native sample sits on a rail; online the latest
  sample is used (zero-order hold). Feed the processor at the tactile front-end rate (= master
  rate, 200 Hz) to keep this negligible; ``saturated=`` lets a driver OR in its own flags.
- **Hand labels** (glove only): preprocessing smooths vision hand labels non-causally; online
  joint states are whatever the robot / IMU model measures now.
- **Baseline window**: offline ``baseline_raw`` is the median of the first ``baseline.duration_s``
  (1 s) of the first ``no_contact`` segment; online it is captured at start-up while the hand
  holds still without contact (:meth:`OnlineTactileProcessor.capture_baseline`) — the same
  estimator (:func:`common.signal.estimate_baseline`) over the same kind of window.

The joint state ``q`` must be in the processor's ``joint_names`` order: the baseline model's
(``bundle_meta["joint_names"]``: URDF order for robots, ``datasets.build.HAND_Q_NAMES`` for gloves)
or, without a model, the URDF actuated-joint order for URDF-parented skins (the
:class:`~robot_skin.control.runner.PolicyRunner` permutes the driver's order into it); ``qd`` is
recomputed from ``q`` with the model's stored ``joint_velocity`` settings unless the model was
trained on the driver's velocities (``bundle_meta["qd_source"] == "file"``: pass ``qd=``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common.layouts import Layout, load_layout
from common.signal import estimate_baseline, press_intensity, relative_change, saturation_mask

__all__ = [
    "CausalJointVelocity", "TactileFrame", "OnlineTactileProcessor", "glove_pose_fn", "robot_pose_fn",
    "static_pose_fn", "make_pose_fn", "load_calibrator", "startup_calibrator", "replay_episode",
]

PoseFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _pre_defaults() -> dict:
    from ..datasets.build import DEFAULTS

    return DEFAULTS


# ─────────────────────────────────────────────────────────────── joint velocity

class CausalJointVelocity:
    """Streaming :func:`robot_skin.datasets.build.joint_velocity` (``method="savgol_causal"``).

    Keeps the last ``w`` joint vectors (``w`` = the odd Savitzky–Golay window of
    :func:`~robot_skin.datasets.build.qd_support`), edge-padded with the first one, and returns the
    derivative of the local polynomial fit at the newest sample — row ``t`` of the offline result,
    exactly (same coefficients, float64 accumulation, float32 output). Without scipy both sides fall
    back to backward differences + a causal moving average (offline equal once ``T ≥ w``).
    The centred variants (``savgol``, ``gradient``) need future samples and are rejected.
    """

    def __init__(self, hz: float, *, method: str = "savgol_causal", window_s: float = 0.05,
                 polyorder: int = 2):
        from ..datasets.build import qd_support

        if method != "savgol_causal":
            raise ValueError(f"qd method {method!r} is not causal — an online controller can only reproduce "
                             "'savgol_causal' (preprocess with qd.method: savgol_causal)")
        if not hz > 0:
            raise ValueError("hz must be > 0")
        self.hz, self.dt = float(hz), 1.0 / float(hz)
        self.method, self.window_s, self.polyorder = method, float(window_s), int(polyorder)
        self.w = int(qd_support(self.hz, method=method, window_s=self.window_s, polyorder=self.polyorder)[0])
        try:
            from scipy.signal import savgol_coeffs
        except ImportError:  # pragma: no cover - scipy is normally present
            self._c = None
        else:
            self._c = savgol_coeffs(self.w, self.polyorder, deriv=1, delta=self.dt, pos=self.w - 1, use="dot")
        self.reset()

    def reset(self) -> None:
        self._buf: np.ndarray | None = None            # [w, D] float64, oldest first
        self._dbuf: np.ndarray | None = None           # fallback: last w backward differences

    def push(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if self._buf is None:
            self._buf = np.repeat(q[None], self.w, axis=0)
            self._dbuf = np.zeros((self.w, q.shape[0]))
        else:
            if q.shape[0] != self._buf.shape[1]:
                raise ValueError(f"q has {q.shape[0]} joints, expected {self._buf.shape[1]}")
            d = (q - self._buf[-1]) / self.dt
            self._buf = np.concatenate([self._buf[1:], q[None]], axis=0)
            self._dbuf = np.concatenate([self._dbuf[1:], d[None]], axis=0)
        if self._c is not None:
            return (self._buf.T @ self._c).astype(np.float32)
        return self._dbuf.mean(0).astype(np.float32)


# ─────────────────────────────────────────────────────────────── taxel poses

def glove_pose_fn(layout: Layout, skeleton: Any = None) -> PoseFn:
    """``q[45]`` (MANO finger pose, ``HAND_Q_NAMES`` order) → hand-frame taxel poses, exactly as
    preprocessing (``pose.mano.taxel_poses_from_hand`` with ``global_orient = 0``, wrist at origin)."""
    from ..pose.mano import ManoSkeleton, taxel_poses_from_hand

    sk = skeleton if skeleton is not None else ManoSkeleton()
    go = np.zeros(3)

    def fn(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fp = np.asarray(q, dtype=np.float64).reshape(15, 3)
        p, n = taxel_poses_from_hand(layout, sk, go, fp, None)
        return p.astype(np.float32), n.astype(np.float32)

    return fn


class _NumpyChainFK:
    """Single-configuration URDF forward kinematics in numpy for the links a layout needs — the same
    transforms as :meth:`robot_skin.pose.urdf.URDFModel.fk` (origin · Rodrigues motion, mimic joints
    following their source) without torch's per-op overhead (≈ 10× faster per tick)."""

    def __init__(self, model: Any, links: Sequence[str]):
        act = {n: i for i, n in enumerate(model.joint_names)}
        need: dict[str, Any] = {}
        for ln in links:
            for j in model.chain(ln):
                need[j.name] = j
        joints = sorted(need.values(), key=lambda j: len(model.chain(j.child)))      # parents first
        self.root = model.root_link
        self.links = list(links)
        self.steps = []
        for j in joints:
            drive = None
            if j.type in ("revolute", "continuous", "prismatic"):
                drive = self._drive(model, j, act, set())
            ax = np.asarray(j.axis, dtype=np.float64)
            K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
            self.steps.append((j.parent, j.child, j.origin_matrix(), j.type, ax, K, drive))

    @staticmethod
    def _drive(model: Any, j: Any, act: Mapping[str, int], seen: set) -> tuple[int, float, float] | None:
        if j.mimic is None:
            return (act[j.name], 1.0, 0.0) if j.name in act else None
        if getattr(model, "mimic_mode", "follow") == "ignore" or j.name in seen:
            return None
        src, mult, off = j.mimic
        base = _NumpyChainFK._drive(model, model.joint(src), act, seen | {j.name})
        return None if base is None else (base[0], base[1] * float(mult), base[2] * float(mult) + float(off))

    def __call__(self, q: np.ndarray) -> dict[str, np.ndarray]:
        out = {self.root: np.eye(4)}
        for parent, child, T_o, typ, ax, K, drive in self.steps:
            T = out[parent] @ T_o
            if drive is not None:
                qj = float(q[drive[0]]) * drive[1] + drive[2]
                M = np.eye(4)
                if typ == "prismatic":
                    M[:3, 3] = qj * ax
                else:
                    M[:3, :3] += math.sin(qj) * K + (1.0 - math.cos(qj)) * (K @ K)
                T = T @ M
            out[child] = T
        return out


def _urdf_model(urdf: Any):
    from ..pose.urdf import URDFModel

    return urdf if isinstance(urdf, URDFModel) else URDFModel.from_file(urdf)


def robot_pose_fn(layout: Layout, urdf: Any, joint_names: Sequence[str] | None = None) -> PoseFn:
    """``q[D]`` → URDF-root-frame taxel poses, the transforms of
    ``pose.robot_fk.taxel_poses_from_joints`` (as preprocessing), evaluated in numpy per tick.

    ``q`` is in URDF actuated-joint order unless ``joint_names`` gives its column order (e.g. a
    baseline model's ``bundle_meta["joint_names"]``); every URDF joint must then be named."""
    from ..pose.robot_fk import taxel_poses_from_joints

    model = _urdf_model(urdf)
    taxel_poses_from_joints(layout, model, np.zeros(model.n_dof))          # validate links early
    perm = None
    if joint_names is not None and list(joint_names) != list(model.joint_names):
        pos = {str(n): i for i, n in enumerate(joint_names)}
        missing = [n for n in model.joint_names if n not in pos]
        if missing:
            raise ValueError(f"q columns {list(joint_names)} lack URDF joints {missing}")
        perm = np.asarray([pos[n] for n in model.joint_names], dtype=np.int64)
    n_in = model.n_dof if joint_names is None else len(joint_names)
    links = sorted(set(layout.parents))
    li = {ln: i for i, ln in enumerate(links)}
    idx = np.array([li[p] for p in layout.parents])
    pl, nl = layout.positions, layout.normals
    fk = _NumpyChainFK(model, links)

    def fn(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qv = np.asarray(q, dtype=np.float64).reshape(-1)
        if qv.shape[0] != n_in:
            raise ValueError(f"q must have {n_in} joints "
                             f"({list(joint_names if joint_names is not None else model.joint_names)}), "
                             f"got {qv.shape[0]}")
        if perm is not None:
            qv = qv[perm]
        T = fk(qv)
        LT = np.stack([T[ln] for ln in links])[idx]                           # [N,4,4]
        R, t = LT[:, :3, :3], LT[:, :3, 3]
        p = np.einsum("nij,nj->ni", R, pl) + t
        n = np.einsum("nij,nj->ni", R, nl)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return p.astype(np.float32), n.astype(np.float32)

    return fn


def static_pose_fn(layout: Layout) -> PoseFn:
    """Constant layout poses (bench / flat sensors — the preprocessing ``static`` source)."""
    p, n = layout.positions.astype(np.float32), layout.normals.astype(np.float32)
    return lambda q: (p, n)


def make_pose_fn(layout: Layout, *, skeleton: Any = None, urdf: Any = None,
                 joint_names: Sequence[str] | None = None) -> PoseFn:
    """Pose function for the layout's parent frame: ``mano`` → :func:`glove_pose_fn`, ``urdf`` →
    :func:`robot_pose_fn` (needs ``urdf``; ``joint_names`` = column order of ``q``; without a URDF
    a warning and static poses, like preprocessing without a URDF), otherwise static."""
    if layout.parent_frame == "mano":
        return glove_pose_fn(layout, skeleton)
    if layout.parent_frame == "urdf":
        if urdf is None:
            static = static_pose_fn(layout)
            warned = []

            def fn(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                if not warned:
                    warnings.warn(f"layout {layout.name!r} is URDF-parented but no URDF was given: static taxel "
                                  "poses (the baseline model was trained on FK poses — pass urdf=)", stacklevel=3)
                    warned.append(True)
                return static(q)

            return fn
        return robot_pose_fn(layout, urdf, joint_names)
    return static_pose_fn(layout)


# ─────────────────────────────────────────────────────────────── calibrator helpers

def load_calibrator(src: Any):
    """:class:`~robot_skin.contact.calibration.ResidualCalibrator` from an instance, its dict, a
    ``calibrator.json`` path or the contact stage's output directory; ``None`` → ``None``."""
    from ..contact.calibration import ResidualCalibrator

    if src is None or isinstance(src, ResidualCalibrator):
        return src
    if isinstance(src, Mapping):
        return ResidualCalibrator.from_dict(src)
    p = Path(src)
    if p.is_dir():
        p = p / "calibrator.json"
    return ResidualCalibrator.load(p)


def startup_calibrator(residual: np.ndarray, saturated: np.ndarray | None = None,
                       logvar: np.ndarray | None = None, **fit_kw: Any):
    """**Bring-up fallback** when no contact-stage calibrator exists: fit a
    :class:`~robot_skin.contact.calibration.ResidualCalibrator` on residuals ``[T,N]`` recorded while
    the hand holds still without contact at start-up (``valid = ~saturated``). A static hold does
    not exercise the motion artefact, so σ is optimistic during motion — train the contact stage
    for real deployments. ``fit_kw`` go to ``ResidualCalibrator.fit`` (``min_samples`` defaults to
    ``min(50, T // 2)``)."""
    from ..contact.calibration import ResidualCalibrator

    r = np.asarray(residual, dtype=np.float32)
    valid = None if saturated is None else ~np.asarray(saturated, dtype=bool)
    fit_kw.setdefault("min_samples", max(2, min(50, r.shape[0] // 2)))
    cal = ResidualCalibrator.fit(r, valid, logvar, **fit_kw)
    cal.info.update(source="startup_hold", n_frames=int(r.shape[0]))
    return cal


# ─────────────────────────────────────────────────────────────── frame

@dataclass
class TactileFrame:
    """Everything the processor computed for one tick (``[N]`` arrays in layout order)."""

    t: float
    raw: np.ndarray                    # layout order, ADC counts (float64)
    delta: np.ndarray                  # ΔS % float32 (SATS sign)
    saturated: np.ndarray              # bool — rails / |ΔS| ≥ max_abs_pct / dead / driver flag
    baseline_mean: np.ndarray          # predicted no-contact ΔS % float32 (0 without a model)
    baseline_logvar: np.ndarray | None
    residual: np.ndarray               # ΔS − mean (FSM-corrected when the gate is on) float32
    residual_z: np.ndarray             # calibrated press-positive z float32 (NaN without calibrator)
    level: np.ndarray                  # int8 ContactLevel
    untrusted: np.ndarray              # bool — saturated or FSM-recovering (→ level SATURATED)
    press_pct: np.ndarray              # centred press intensity %
    contact: np.ndarray                # bool — the policy's contact rule on (level, saturated)
    features: np.ndarray | None = None  # [N, F] tactile_value_features (history-stacked)
    prob: np.ndarray | None = None      # detector probability
    contact_on: np.ndarray | None = None  # hysteresis-filtered detector state
    q: np.ndarray | None = None
    qd: np.ndarray | None = None
    pos: np.ndarray | None = None
    nrm: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def any_contact(self) -> bool:
        return bool(np.any(self.contact))


# ─────────────────────────────────────────────────────────────── processor

class OnlineTactileProcessor:
    """Per-tick tactile pipeline (module docstring); call :meth:`step` once per master-clock tick.

    Args:
        layout: skin layout (name / path / :class:`~common.layouts.Layout`).
        baseline_model: :class:`~robot_skin.baseline.temporal.TemporalBaselinePredictor` (its
            ``bundle_meta`` supplies ``hz``, ``qd`` settings, ``qd_source``, ``joint_names``) or
            ``None`` (no motion model: residual = ΔS).
        calibrator: :class:`~robot_skin.contact.calibration.ResidualCalibrator` (or anything
            :func:`load_calibrator` accepts); ``None`` → z = NaN and levels only mark saturation
            until :meth:`set_calibrator` (see :func:`startup_calibrator`).
        joint_stats: only for models trained on externally normalised joints (as
            ``predict_episode(joint_stats=...)``); stage-trained models normalise internally.
        fsm: override of the calibrator's :class:`~robot_skin.contact.SaturationFSM` gate config.
        window: override of the model window (frames).
        hz: master rate (default ``bundle_meta["hz"]`` or 200); ``qd``: ``joint_velocity`` kwargs
            (default ``bundle_meta["qd"]`` or the preprocessing defaults); ``qd_source``:
            ``derivative`` | ``file`` (use the ``qd`` passed to :meth:`step`).
        feature_spec: :class:`~robot_skin.representation.TactileFeatureSpec` (dict ok) → per-tick
            ``features``; ``contact_rule``: ``vtla.dataset.CONTACT_RULES`` entry.
        pose_fn: ``q → (pos[N,3], nrm[N,3])``; default :func:`make_pose_fn` (``urdf`` / ``skeleton``).
        detector / hysteresis: optional :class:`~robot_skin.contact.detector.ContactDetector` and
            :class:`~robot_skin.contact.hysteresis.HysteresisFilter` (config dict ok; default from
            ``detector.bundle_meta["hysteresis"]``).
        pressure: ``{adc_min, adc_max, rail_margin, max_abs_pct}`` (default: preprocessing defaults).
        baseline_raw: per-taxel raw baseline (layout order) if already known; else capture it.
        baseline_s: duration of the start-up baseline window (preprocessing ``baseline.duration_s``).
        raw_order: ``channel`` (``read_pressure`` output, mapped with ``layout.by_channel``) or
            ``layout``.
    """

    def __init__(self, layout: str | Path | Layout, baseline_model: Any = None, calibrator: Any = None, *,
                 joint_stats: Any = None, fsm: Mapping[str, Any] | None = None, window: int | None = None,
                 hz: float | None = None, qd: Mapping[str, Any] | None = None, qd_source: str | None = None,
                 feature_spec: Any = None, contact_rule: str = "level_ge_weak", pose_fn: PoseFn | None = None,
                 urdf: Any = None, skeleton: Any = None, detector: Any = None, hysteresis: Any = None,
                 pressure: Mapping[str, Any] | None = None, baseline_raw: Any = None, baseline_s: float | None = None,
                 raw_order: str = "channel", device: Any = None, joint_names: Sequence[str] | None = None):
        from ..contact.hysteresis import HysteresisFilter
        from ..representation.encoder import TactileFeatureSpec
        from ..vtla.dataset import CONTACT_RULES

        self.layout = layout if isinstance(layout, Layout) else load_layout(layout)
        N = self.layout.n
        meta = dict(getattr(baseline_model, "bundle_meta", {}) or {})
        pre = _pre_defaults()
        self.model = baseline_model
        if baseline_model is not None and int(baseline_model.n_taxels) != N:
            raise ValueError(f"baseline model has {baseline_model.n_taxels} taxels, layout {self.layout.name!r} {N}")
        self.hz = float(hz or meta.get("hz") or 200.0)
        self.dt = 1.0 / self.hz
        qd_kw = dict(qd if qd is not None else (meta.get("qd") or {k: pre["qd"][k] for k in ("method", "window_s",
                                                                                                "polyorder")}))
        self.qd_settings = qd_kw
        self.qd_source = str(qd_source or meta.get("qd_source") or "derivative")
        if self.qd_source not in ("derivative", "file"):
            raise ValueError("qd_source must be 'derivative' or 'file'")
        self._qd_filter = CausalJointVelocity(self.hz, **qd_kw) if self.qd_source == "derivative" else None
        self.joint_names = list(joint_names if joint_names is not None else meta.get("joint_names") or [])
        urdf_model = None
        if self.layout.parent_frame == "urdf" and urdf is not None and pose_fn is None:
            urdf_model = _urdf_model(urdf)
            if not self.joint_names:
                # q order of the URDF pose function (preprocessing reorders robot q to URDF order too);
                # the runner permutes the driver's joint order into it
                self.joint_names = list(urdf_model.joint_names)
        if baseline_model is not None and self.joint_names and len(self.joint_names) != int(baseline_model.joint_dim):
            raise ValueError(f"baseline model expects {baseline_model.joint_dim} joints, joint_names has "
                             f"{len(self.joint_names)}")
        self.window = int(window or (baseline_model.window if baseline_model is not None else 1))
        self.joint_stats = joint_stats
        self.calibrator = self._check_calibrator(load_calibrator(calibrator))
        self._fsm_override = None if fsm is None else dict(fsm)
        if feature_spec is not None and not isinstance(feature_spec, TactileFeatureSpec):
            feature_spec = TactileFeatureSpec.from_dict(feature_spec)
        self.feature_spec = feature_spec
        if contact_rule not in CONTACT_RULES:
            raise ValueError(f"contact_rule must be one of {CONTACT_RULES}, got {contact_rule!r}")
        self.contact_rule = contact_rule
        self.pose_fn = pose_fn if pose_fn is not None else make_pose_fn(
            self.layout, skeleton=skeleton, urdf=urdf_model if urdf_model is not None else urdf,
            joint_names=self.joint_names if urdf_model is not None else None)
        self.detector = detector
        if detector is not None and hysteresis is None:
            hysteresis = (getattr(detector, "bundle_meta", {}) or {}).get("hysteresis")
        self.hysteresis = None if hysteresis is None else HysteresisFilter.from_config(hysteresis)
        pc = {k: pre["pressure"][k] for k in ("adc_min", "adc_max", "rail_margin", "max_abs_pct")}
        unknown = sorted(set(pressure or {}) - set(pc))
        if unknown:
            raise ValueError(f"unknown pressure keys {unknown}; valid: {sorted(pc)}")
        pc.update(dict(pressure or {}))
        self.pressure_cfg = pc
        if raw_order not in ("channel", "layout"):
            raise ValueError("raw_order must be 'channel' or 'layout'")
        self.raw_order = raw_order
        self.baseline_s = float(baseline_s if baseline_s is not None else pre["baseline"]["duration_s"])
        self.device = device
        self.baseline_raw: np.ndarray | None = None
        self.dead = np.zeros(N, dtype=bool)
        if baseline_raw is not None:
            self.set_baseline(baseline_raw)
        self._bl_rows: list[np.ndarray] = []
        self._bl_t: list[float] = []
        self._bl_rail: list[np.ndarray] = []
        self._bl_n = self._bl_clean = 0
        self.reset()

    # ── construction helpers ──────────────────────────────────────────────
    @classmethod
    def from_stage_outputs(cls, layout: Any, *, baseline: Any = None, calibrator: Any = None,
                           detector: Any = None, device: Any = "cpu", **kw: Any) -> "OnlineTactileProcessor":
        """Build from the stage-1 artefacts: ``baseline`` (``baseline_model.pt`` / its run dir / a
        model), ``calibrator`` (``calibrator.json`` / the contact run dir / dict / instance) and an
        optional ``detector`` (``contact_detector.pt`` / model). Other kwargs as the constructor."""
        model = baseline
        if baseline is not None and not hasattr(baseline, "forward"):
            from ..baseline.temporal import load_baseline_model

            model = load_baseline_model(baseline, map_location=device or "cpu")
        det = detector
        if detector is not None and not hasattr(detector, "forward"):
            from ..contact.detector import load_detector

            p = Path(detector)
            det = load_detector(p, map_location=device or "cpu")
        return cls(layout, model, calibrator, detector=det, device=device, **kw)

    @classmethod
    def from_policy_bundle(cls, bundle: Any, layout: Any, *, baseline: Any = None, calibrator: Any = None,
                           detector: Any = None, device: Any = "cpu", **kw: Any) -> "OnlineTactileProcessor":
        """Build from a :class:`~robot_skin.control.bundle.PolicyBundle` (or its path): the tactile
        feature spec and contact rule come from the bundle. Unless given, the calibrator (the
        bundle's embedded ``calibrator.json`` or its path) and the baseline model (the bundle's
        ``baseline_model`` reference, when the file exists) are taken from the bundle **only if they
        belong to this skin** (:meth:`PolicyBundle.stage1_matches` — a glove-trained bundle's stage-1
        models do not describe a robot skin); a bundle calibrator fitted with the baseline
        log-variance is dropped (with a warning) when its baseline model is unavailable."""
        from .bundle import PolicyBundle, load_policy_bundle

        b = bundle if isinstance(bundle, PolicyBundle) else load_policy_bundle(bundle, device=device)
        lay = layout if isinstance(layout, Layout) else load_layout(layout)
        refs_ok = b.stage1_matches(lay)
        if not refs_ok and (calibrator is None or baseline is None):
            warnings.warn(f"the bundle's stage-1 references belong to layouts {b.tactile.get('layouts')}, not "
                          f"{lay.name!r}: not used (pass calibrator= / baseline= of this skin)", stacklevel=2)
        cal = calibrator if calibrator is not None else (b.calibrator() if refs_ok else None)
        base = baseline
        if base is None and refs_ok:
            ref = b.baseline_model_path()
            if ref is not None:
                base = ref
            elif (b.tactile or {}).get("baseline_model"):
                warnings.warn(f"baseline model {b.tactile['baseline_model']!r} referenced by the bundle was not "
                              "found: running without a motion-artefact model", stacklevel=2)
        if calibrator is None and cal is not None and cal.use_logvar and base is None:
            warnings.warn("the bundle's calibrator was fitted with the baseline log-variance but no baseline model "
                          "is available: calibrator not used (fit one at start-up or pass calibrator=)", stacklevel=2)
            cal = None
        kw.setdefault("feature_spec", b.feature_spec)
        kw.setdefault("contact_rule", b.contact_rule)
        if base is None:
            kw.setdefault("hz", b.source_hz)                  # else the baseline model's own rate
        proc = cls.from_stage_outputs(lay, baseline=base, calibrator=cal, detector=detector, device=device, **kw)
        if abs(proc.hz - float(b.source_hz)) > 1e-6:
            warnings.warn(f"tactile processor rate {proc.hz:g} Hz (baseline model) differs from the policy's "
                          f"source rate {b.source_hz:g} Hz: tactile history strides will not match training",
                          stacklevel=2)
        return proc

    # ── state ─────────────────────────────────────────────────────────────
    def reset(self, *, keep_baseline: bool = True) -> None:
        """Start a new stream (episode): clears the qd buffer, the baseline / detector windows, the
        FSM, tactile history and hysteresis. The captured baseline is kept unless
        ``keep_baseline=False``."""
        from ..baseline.temporal import CausalBaselineStream
        from ..contact.detector import CausalDetectorStream
        from ..contact.saturation_fsm import SaturationFSM
        from ..representation.encoder import TactileHistory

        if self._qd_filter is not None:
            self._qd_filter.reset()
        self._stream = (None if self.model is None else
                        CausalBaselineStream(self.model, window=self.window, device=self.device))
        fcfg = self.fsm_config
        self._fsm = None
        if fcfg.get("enabled", False):
            kw = {k: fcfg[k] for k in ("ok_pct", "ok_sec", "max_recover_s") if k in fcfg}
            self._fsm = SaturationFSM(self.layout.n, **kw)
        self._hist = None if self.feature_spec is None else TactileHistory(self.feature_spec)
        self._det = None if self.detector is None else CausalDetectorStream(self.detector, device=self.device)
        if self.hysteresis is not None:
            self.hysteresis.reset()
        self._q_last: np.ndarray | None = None
        self.n_steps = 0
        if not keep_baseline:
            self.baseline_raw = None
            self.dead = np.zeros(self.layout.n, dtype=bool)
            self.begin_baseline()

    @property
    def fsm_config(self) -> dict:
        if self._fsm_override is not None:
            return dict(self._fsm_override)
        return dict((self.calibrator.fsm or {}) if self.calibrator is not None else {})

    def _check_calibrator(self, cal: Any) -> Any:
        """A calibrator must match the skin and the residual it was fitted on: ``use_logvar``
        calibrators scale z by the baseline model's predicted variance, so they need that model
        (without it the residual is raw ΔS — the motion artefact would read as contact)."""
        if cal is None:
            return None
        if cal.n_taxels != self.layout.n:
            raise ValueError(f"calibrator has {cal.n_taxels} taxels, layout {self.layout.name!r} {self.layout.n}")
        if cal.use_logvar and self.model is None:
            raise ValueError("the calibrator was fitted with the baseline model's log-variance (use_logvar) but no "
                             "baseline model is loaded: pass the baseline_model.pt it was fitted with, or use a "
                             "calibrator fitted without it (e.g. startup_calibrator)")
        return cal

    def set_calibrator(self, calibrator: Any) -> None:
        """Install a calibrator (e.g. :func:`startup_calibrator`) and restart the streams."""
        self.calibrator = self._check_calibrator(load_calibrator(calibrator))
        self.reset()

    # ── baseline capture ──────────────────────────────────────────────────
    @property
    def baseline_ready(self) -> bool:
        return self.baseline_raw is not None

    def _to_layout(self, raw: np.ndarray) -> np.ndarray:
        r = np.asarray(raw, dtype=np.float64).reshape(-1)
        if self.raw_order == "channel":
            if r.shape[0] <= int(self.layout.channels.max()):
                raise ValueError(f"raw has {r.shape[0]} channels; layout {self.layout.name!r} uses channel "
                                 f"{int(self.layout.channels.max())}")
            return self.layout.by_channel(r)
        if r.shape[0] != self.layout.n:
            raise ValueError(f"raw (layout order) must have {self.layout.n} values, got {r.shape[0]}")
        return r

    def _rails(self, raw_n: np.ndarray) -> np.ndarray:
        pc = self.pressure_cfg
        bad = ~np.isfinite(raw_n)
        r = np.where(bad, 0.5 * (pc["adc_min"] + pc["adc_max"]), raw_n)
        return saturation_mask(raw=r, adc_min=pc["adc_min"], adc_max=pc["adc_max"], rail_margin=pc["rail_margin"],
                               max_abs_pct=None) | bad

    def rail_mask(self, raw: np.ndarray) -> np.ndarray:
        """``bool[N]`` (layout order): taxels of a raw sample (``raw_order``) on an ADC rail or
        non-finite — e.g. to flag a sample interpolated between two raw samples as saturated when
        either bracketing sample sits on a rail (the preprocessing resampling rule)."""
        return self._rails(self._to_layout(raw))

    def begin_baseline(self) -> None:
        self._bl_rows, self._bl_t, self._bl_rail = [], [], []
        self._bl_n = self._bl_clean = 0

    def add_baseline_sample(self, raw: np.ndarray, t: float | None = None) -> int:
        """Collect one no-contact raw sample for the start-up baseline. Samples with a taxel on a
        rail are not used (preprocessing uses frames without a saturated taxel) — except for taxels
        on a rail in *every* sample, which :meth:`finish_baseline` marks dead instead (a dead /
        unplugged channel must not void the whole window). ``t`` defaults to the sample's tick time
        (``index · dt``, skipped samples included — the window is real time, as preprocessing's
        master-clock timestamps). Returns the number of samples so far without any rail taxel."""
        r = self._to_layout(raw)
        ts = float(self._bl_n * self.dt if t is None else t)
        self._bl_n += 1
        self._bl_rows.append(r)
        self._bl_t.append(ts)
        rail = self._rails(r)
        self._bl_rail.append(rail)
        self._bl_clean += int(not rail.any())
        return self._bl_clean

    def finish_baseline(self, duration_s: float | None = None) -> np.ndarray:
        """Baseline = per-taxel median over the first ``duration_s`` (default ``baseline_s``) of the
        usable samples (:func:`common.signal.estimate_baseline`): samples without a rail on any
        live taxel. A taxel on a rail (or non-finite) in every sample is **dead**: baseline 0 → ΔS 0,
        always saturated (:meth:`set_baseline`, as preprocessing marks a non-positive baseline)."""
        n = len(self._bl_rows)
        rails = np.stack(self._bl_rail) if n else np.zeros((0, self.layout.n), bool)
        dead = rails.all(0) if n >= 2 else np.zeros(self.layout.n, bool)
        use = ~rails[:, ~dead].any(1)
        if dead.all() or int(use.sum()) < 2:
            raise RuntimeError(f"only {int(use.sum()) if not dead.all() else 0} usable baseline samples (hold the "
                               "hand still, without contact, and check for rail / dropout samples)")
        rows = np.stack(self._bl_rows)[use]
        base = estimate_baseline(rows, t=np.asarray(self._bl_t)[use], duration_s=float(duration_s or self.baseline_s))
        base[dead] = 0.0
        self.set_baseline(base)
        return self.baseline_raw

    def capture_baseline(self, raw_rows: np.ndarray, t: np.ndarray | None = None,
                         duration_s: float | None = None) -> np.ndarray:
        """:meth:`begin_baseline` + :meth:`add_baseline_sample` per row + :meth:`finish_baseline`."""
        self.begin_baseline()
        rows = np.asarray(raw_rows)
        for i in range(rows.shape[0]):
            self.add_baseline_sample(rows[i], None if t is None else float(t[i]))
        return self.finish_baseline(duration_s)

    def set_baseline(self, baseline_raw: np.ndarray) -> None:
        """Set the per-taxel raw baseline (layout order). Non-positive / non-finite entries mark the
        taxel dead (ΔS = 0, always saturated), like preprocessing."""
        b = np.asarray(baseline_raw, dtype=np.float64).reshape(-1)
        if b.shape != (self.layout.n,):
            raise ValueError(f"baseline_raw must have {self.layout.n} entries (layout order), got {b.shape}")
        self.dead = ~(np.isfinite(b) & (b > 0))
        self.baseline_raw = b

    # ── per-tick pipeline ─────────────────────────────────────────────────
    def step(self, raw: np.ndarray, q: np.ndarray | None = None, *, qd: np.ndarray | None = None,
             pos: np.ndarray | None = None, nrm: np.ndarray | None = None, t: float | None = None,
             saturated: np.ndarray | None = None, q_valid: bool = True) -> TactileFrame:
        """One master-clock tick. ``raw`` (``raw_order``), ``q`` in :attr:`joint_names` order (non-finite
        entries hold the last finite reading and mark the tick ``q_valid = False``); ``pos``/``nrm``
        override ``pose_fn``; ``saturated`` ORs driver flags into the saturation mask; ``q_valid``
        feeds the detector (False while the joint state is not a measurement)."""
        from ..contact.ordinal import ContactLevel
        from ..contact.saturation_fsm import SatState
        from ..vtla.dataset import contact_from_level

        if self.baseline_raw is None:
            raise RuntimeError("no tactile baseline: capture it first (capture_baseline / add_baseline_sample + "
                               "finish_baseline) or pass baseline_raw=")
        N = self.layout.n
        pc = self.pressure_cfg
        raw_n = self._to_layout(raw)
        rails = self._rails(raw_n)
        raw_safe = np.where(np.isfinite(raw_n), raw_n, self.baseline_raw)
        delta = relative_change(raw_safe, np.where(self.dead, 1.0, self.baseline_raw))
        delta[self.dead] = 0.0
        sat = saturation_mask(raw=raw_safe, delta_pct=delta, adc_min=pc["adc_min"], adc_max=pc["adc_max"],
                              rail_margin=pc["rail_margin"], max_abs_pct=pc["max_abs_pct"]) | rails | self.dead
        if saturated is not None:
            sat = sat | np.asarray(saturated, dtype=bool).reshape(N)

        qv = qdv = None
        if q is not None:
            qv = np.asarray(q, dtype=np.float32).reshape(-1)
            bad_q = ~np.isfinite(qv)
            if bad_q.any():
                # a driver glitch must not poison the causal windows (baseline / qd / detector) for
                # W ticks: hold the last finite reading (zero-order hold) and flag the joint state
                last = self._q_last if self._q_last is not None and self._q_last.shape == qv.shape else None
                if last is None:
                    raise ValueError("non-finite joint state and no previous reading to hold")
                qv = np.where(bad_q, last, qv).astype(np.float32)
                q_valid = False
            self._q_last = qv
            if self.qd_source == "derivative":
                qdv = self._qd_filter.push(qv)
            else:
                if qd is None:
                    raise ValueError("qd_source 'file': pass the measured qd to step()")
                qdv = np.asarray(qd, dtype=np.float32).reshape(-1)
        if pos is None or nrm is None:
            if qv is None:
                raise ValueError("pass q (for the pose function) or explicit pos / nrm")
            pos, nrm = self.pose_fn(qv)
        pos = np.asarray(pos, dtype=np.float32).reshape(N, 3)
        nrm = np.asarray(nrm, dtype=np.float32).reshape(N, 3)

        if self.model is not None:
            if qv is None:
                raise ValueError("the baseline model needs q every tick")
            qm, qdm = qv, qdv
            if self.joint_stats is not None:
                from ..baseline.temporal import _as_stats

                js = self.joint_stats
                qs, qds = _as_stats(js["q"]), _as_stats(js["qd"])
                qm, qdm = (qv - qs[0]) / qs[1], (qdv - qds[0]) / qds[1]
            mean, logvar = self._stream.push(qm, qdm, pos, nrm)
        else:
            mean, logvar = np.zeros(N, np.float32), None
        residual = (delta - mean).astype(np.float32)

        if self._fsm is not None:
            state = self._fsm.step(np.nan_to_num(residual, nan=0.0), sat, self.dt)
            residual = self._fsm.corrected(residual)
            untrusted = state != SatState.OK
        else:
            untrusted = sat.copy()
        cal = self.calibrator
        if cal is not None:
            z = cal.transform(residual, logvar if cal.use_logvar else None)
            press = cal.press(residual)
            level = cal.levels(z, untrusted, press_pct=press)
        else:
            z = np.full(N, np.nan, np.float32)
            press = press_intensity(residual)
            level = np.where(untrusted, int(ContactLevel.SATURATED), int(ContactLevel.NONE)).astype(np.int8)
        contact = contact_from_level(level, sat, self.contact_rule)
        feats = None if self._hist is None else self._hist.push(z, level, sat)
        prob = on = None
        if self._det is not None:
            prob = self._det.push(z, sat, qdv if self.detector.use_motion else None, q_valid)
            if self.hysteresis is not None:
                on = self.hysteresis.step(prob)
        self.n_steps += 1
        return TactileFrame(t=float(self.n_steps - 1) * self.dt if t is None else float(t), raw=raw_n,
                            delta=delta, saturated=sat, baseline_mean=np.asarray(mean, np.float32),
                            baseline_logvar=None if logvar is None else np.asarray(logvar, np.float32),
                            residual=residual, residual_z=np.asarray(z, np.float32), level=level,
                            untrusted=np.asarray(untrusted, bool), press_pct=np.asarray(press, np.float32),
                            contact=contact, features=feats, prob=prob, contact_on=on, q=qv, qd=qdv,
                            pos=pos, nrm=nrm)

    def __repr__(self) -> str:
        return (f"OnlineTactileProcessor(layout={self.layout.name!r}, N={self.layout.n}, hz={self.hz:g}, "
                f"baseline_model={self.model is not None}, calibrator={self.calibrator is not None}, "
                f"fsm={bool(self.fsm_config.get('enabled'))}, features={self.feature_spec}, "
                f"detector={self.detector is not None})")


# ─────────────────────────────────────────────────────────────── replay

def replay_episode(processor: OnlineTactileProcessor, episode: Any, *, poses: str = "episode",
                   qd: str = "online", baseline: str = "episode", extra_saturated: bool = True,
                   frames: slice | None = None) -> dict[str, np.ndarray]:
    """Stream a processed :class:`~robot_skin.datasets.episode.Episode` through ``processor`` tick by
    tick (``pressure_raw`` in layout order, ``q``) and stack the per-tick outputs ``[T, ...]``.

    ``poses``: ``episode`` (use ``taxel_pos``/``taxel_nrm``) | ``model`` (``processor.pose_fn``);
    ``qd``: ``online`` (recomputed) | ``episode`` (fed as measured, needs ``qd_source="file"``);
    ``baseline``: ``episode`` (static ``baseline_raw``) | ``capture`` (keep the processor's);
    ``extra_saturated``: OR the episode ``saturated`` flags in (covers the bracketing-rail rule of
    preprocessing, see module docstring). Used to validate a deployment configuration against
    the offline stage outputs of recorded data."""
    from ..datasets.episode import K_PRESSURE_RAW, K_Q, K_QD, K_SATURATED, K_TAXEL_NRM, K_TAXEL_POS, S_BASELINE_RAW

    if processor.raw_order != "layout":
        raise ValueError("replay_episode needs a processor with raw_order='layout' (episodes store layout order)")
    if baseline == "episode":
        processor.set_baseline(episode.static[S_BASELINE_RAW])
    processor.reset()
    raw = np.asarray(episode[K_PRESSURE_RAW])
    q = np.asarray(episode[K_Q]) if episode.has(K_Q) else None
    qd_ep = np.asarray(episode[K_QD]) if (qd == "episode" and episode.has(K_QD)) else None
    sat = np.asarray(episode[K_SATURATED], dtype=bool) if extra_saturated and episode.has(K_SATURATED) else None
    use_ep_pose = poses == "episode" and episode.has(K_TAXEL_POS)
    pos = np.asarray(episode[K_TAXEL_POS]) if use_ep_pose else None
    nrm = np.asarray(episode[K_TAXEL_NRM]) if use_ep_pose else None
    t = np.asarray(episode.t)
    idx = range(episode.T)[frames] if frames is not None else range(episode.T)
    out: dict[str, list] = {}
    for i in idx:
        fr = processor.step(raw[i], None if q is None else q[i], qd=None if qd_ep is None else qd_ep[i],
                            pos=None if pos is None else pos[i], nrm=None if nrm is None else nrm[i], t=float(t[i]),
                            saturated=None if sat is None else sat[i])
        for k in ("delta", "saturated", "baseline_mean", "baseline_logvar", "residual", "residual_z", "level",
                  "untrusted", "press_pct", "contact", "features", "prob", "contact_on", "qd", "pos", "nrm"):
            v = getattr(fr, k)
            if v is not None:
                out.setdefault(k, []).append(v)
    return {k: np.stack(v) for k, v in out.items()}
