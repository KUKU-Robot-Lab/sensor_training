"""Safety layer between the policy and the robot: every joint command passes :class:`SafetyFilter`.

Per control tick, in this order (the first applicable rule wins where they conflict)::

    e-stop latched ─────────────────────────────▶ hold the position at the e-stop
    sensor watchdog (stale pressure / joints / …) ▶ hold the last command; e-stop after estop_after_s
    non-finite target ──────────────────────────▶ hold the last command
    tactile stop (level ∈ stop_levels sustained  ▶ freeze_closing: joints may open, not close further
      ≥ min_ticks on any taxel)                    hold: freeze all joints · estop: e-stop
    joint limits (− margin) ────────────────────▶ clamp
    velocity limit |Δq| ≤ max_vel·dt ───────────▶ clamp the step
    acceleration limit |Δv| ≤ max_acc·dt ───────▶ clamp the step change

The tactile stop is the skin's reflex: a taxel that stays STRONG / SATURATED for ``min_ticks``
ticks (≈ 50 ms at 200 Hz by default) means the hand squeezes something hard — or a sensor fault
(dropout to the ADC rail), which is treated the same way (fail safe). With ``taxel_joints``
(:func:`taxel_joint_mask`) only the joints on the kinematic chain of the triggering taxels are
frozen — at their *measured* position when the stop engages (a position servo lags its command,
so freezing the last command would keep squeezing by the tracking lag); closing = the joint moving
in its ``closing_sign`` direction (+1: increasing q closes, the URDF flexion convention of the
synthetic hand; 0: never frozen). The stop releases after
``release_ticks`` ticks without a triggering taxel. Rate / limit clamps are counted, transitions
(``tactile_stop_on/off``, ``stale_on/off``, ``estop``) are logged as :class:`SafetyEvent` s with the
tick time; the runner copies them into the session's ``events.jsonl`` as markers.

This is a software layer on top of — not a replacement for — the hand's own current / torque
limits and a hardware e-stop.
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["SafetyEvent", "TACTILE_STOP_MODES", "SafetyFilter", "taxel_joint_mask", "closing_signs"]

TACTILE_STOP_MODES = ("freeze_closing", "hold", "estop")
_TACTILE_DEFAULTS = {"enabled": True, "levels": ("STRONG", "SATURATED"), "min_ticks": 10, "release_ticks": 20,
                     "mode": "freeze_closing"}
_WATCHDOG_DEFAULTS = {"enabled": True, "max_age_s": {"pressure": 0.05, "joint_state": 0.05},
                      "estop_after_s": 0.5}


@dataclass
class SafetyEvent:
    t: float
    type: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _level_ids(levels: Sequence[Any]) -> list[int]:
    from ..contact.ordinal import ContactLevel

    out = []
    for lv in levels:
        out.append(int(ContactLevel[lv.upper()]) if isinstance(lv, str) else int(lv))
    return out


def taxel_joint_mask(layout: Any, model: Any, joint_names: Sequence[str] | None = None) -> np.ndarray:
    """``bool[N, D]``: joint ``d`` lies on the kinematic chain of taxel ``n``'s parent link (URDF
    ``model``). Taxels on links without actuated joints (palm) map to every joint (a palm press
    stops the whole grasp). ``joint_names`` = the command order (default the model's)."""
    names = list(joint_names if joint_names is not None else model.joint_names)
    col = {n: i for i, n in enumerate(names)}
    M = np.zeros((layout.n, len(names)), dtype=bool)
    for i, par in enumerate(layout.parents):
        js = [col[j.name] for j in model.chain(par) if j.name in col]
        if js:
            M[i, js] = True
        else:
            M[i, :] = True
    return M


def closing_signs(joint_names: Sequence[str], spec: float | Mapping[str, float] | Sequence[float] | None = 1.0
                  ) -> np.ndarray:
    """Per-joint closing direction ``[D]`` from a scalar, a sequence or ``{name_or_glob: sign}``
    (later patterns win; unmatched joints +1)."""
    D = len(joint_names)
    if spec is None:
        return np.ones(D)
    if isinstance(spec, Mapping):
        s = np.ones(D)
        for pat, v in spec.items():
            for i, n in enumerate(joint_names):
                if fnmatch.fnmatch(n, str(pat)):
                    s[i] = float(v)
        return np.sign(s)
    return np.sign(np.broadcast_to(np.asarray(spec, dtype=np.float64), (D,)).copy())


class SafetyFilter:
    """Clamp / freeze / hold joint commands (module docstring).

    Args:
        lower, upper: joint limits ``[D]`` (command order); ``margin`` (rad) keeps commands inside.
        dt: control period (s).
        max_vel: rad/s (scalar / ``[D]``; ``None`` = off). ``max_acc``: rad/s² (``None`` = off).
        tactile_stop: ``None`` = off, or ``{enabled, levels, min_ticks, release_ticks, mode}``
            (defaults: STRONG / SATURATED, 10, 20, ``freeze_closing``).
        closing_sign: see :func:`closing_signs`; ``taxel_joints``: ``bool[N, D]`` (default: all).
        watchdog: ``None`` = off, or ``{enabled, max_age_s: {stream: s (glob ok)}, estop_after_s}``.
        estop_callback: called once with ``(reason, t)`` when the e-stop latches (e.g. ``robot.estop``).
        joint_names: for readable events / :func:`closing_signs` patterns.
    """

    def __init__(self, lower: Sequence[float], upper: Sequence[float], *, dt: float,
                 max_vel: float | Sequence[float] | None = None, max_acc: float | Sequence[float] | None = None,
                 margin: float = 0.0, tactile_stop: Mapping[str, Any] | None = None,
                 closing_sign: Any = 1.0, taxel_joints: np.ndarray | None = None,
                 watchdog: Mapping[str, Any] | None = None, estop_callback: Callable[[str, float], Any] | None = None,
                 joint_names: Sequence[str] | None = None, max_events: int = 10000):
        lo = np.asarray(lower, dtype=np.float64).reshape(-1)
        hi = np.asarray(upper, dtype=np.float64).reshape(-1)
        if lo.shape != hi.shape or np.any(hi < lo):
            raise ValueError("lower / upper must be [D] with upper ≥ lower")
        if not dt > 0:
            raise ValueError("dt must be > 0")
        D = lo.shape[0]
        self.D, self.dt = D, float(dt)
        m = float(margin)
        self.lower = np.where(hi - lo > 2 * m, lo + m, lo)
        self.upper = np.where(hi - lo > 2 * m, hi - m, hi)
        self.max_vel = None if max_vel is None else np.broadcast_to(np.asarray(max_vel, np.float64), (D,)).copy()
        self.max_acc = None if max_acc is None else np.broadcast_to(np.asarray(max_acc, np.float64), (D,)).copy()
        for nm, v in (("max_vel", self.max_vel), ("max_acc", self.max_acc)):
            if v is not None and np.any(~(v > 0)):
                raise ValueError(f"{nm} must be > 0")
        self.joint_names = list(joint_names) if joint_names is not None else [f"j{i}" for i in range(D)]
        if len(self.joint_names) != D:
            raise ValueError(f"{len(self.joint_names)} joint names for {D} joints")

        tcfg = None if tactile_stop is None else {**_TACTILE_DEFAULTS, **dict(tactile_stop)}
        if tcfg is not None:
            unknown = sorted(set(tcfg) - set(_TACTILE_DEFAULTS))
            if unknown:
                raise ValueError(f"unknown tactile_stop keys {unknown}; valid: {sorted(_TACTILE_DEFAULTS)}")
            if tcfg["mode"] not in TACTILE_STOP_MODES:
                raise ValueError(f"tactile_stop.mode must be one of {TACTILE_STOP_MODES}")
            if int(tcfg["min_ticks"]) < 1 or int(tcfg["release_ticks"]) < 1:
                raise ValueError("tactile_stop min_ticks / release_ticks must be ≥ 1")
            if not tcfg["enabled"]:
                tcfg = None
        self.tactile_cfg = tcfg
        self._stop_levels = _level_ids(tcfg["levels"]) if tcfg else []
        self.closing = closing_signs(self.joint_names, closing_sign)
        self.taxel_joints = None if taxel_joints is None else np.asarray(taxel_joints, dtype=bool)
        if self.taxel_joints is not None and self.taxel_joints.shape[1] != D:
            raise ValueError(f"taxel_joints must be [N, {D}], got {self.taxel_joints.shape}")

        wcfg = None if watchdog is None else {**_WATCHDOG_DEFAULTS, **dict(watchdog)}
        if wcfg is not None:
            unknown = sorted(set(wcfg) - set(_WATCHDOG_DEFAULTS))
            if unknown:
                raise ValueError(f"unknown watchdog keys {unknown}; valid: {sorted(_WATCHDOG_DEFAULTS)}")
            if not wcfg["enabled"]:
                wcfg = None
        self.watchdog_cfg = wcfg
        self.estop_callback = estop_callback
        self.max_events = int(max_events)
        self.reset(None)

    # ── state ─────────────────────────────────────────────────────────────
    def reset(self, q_current: Sequence[float] | None) -> None:
        """Start (or restart) from the robot's current position; clears the e-stop latch."""
        q = None if q_current is None else np.clip(np.asarray(q_current, np.float64).reshape(self.D),
                                                   self.lower, self.upper)
        self.q_last = q
        self.v_last = np.zeros(self.D)
        self.estopped = False
        self.estop_reason: str | None = None
        self._q_estop = q
        self._cnt: np.ndarray | None = None
        self._release = 0
        self.stop_active = False
        self._stop_joints = np.zeros(self.D, dtype=bool)
        self._q_freeze: np.ndarray | None = None
        self._stale_since: float | None = None
        self.stale = False
        self.events: list[SafetyEvent] = []
        self.counts: dict[str, int] = {}
        self.n_ticks = 0

    def _event(self, t: float, typ: str, **detail: Any) -> None:
        self.counts[typ] = self.counts.get(typ, 0) + 1
        if len(self.events) < self.max_events:
            self.events.append(SafetyEvent(float(t), typ, detail))

    def _count(self, typ: str, n: int = 1) -> None:
        if n:
            self.counts[typ] = self.counts.get(typ, 0) + int(n)

    def trigger_estop(self, reason: str, t: float, q_hold: Sequence[float] | None = None) -> None:
        """Latch the e-stop until :meth:`reset`: hold ``q_hold`` (the tactile stop passes the measured
        position — holding the last command would keep squeezing by the servo lag) or the last
        command."""
        if self.estopped:
            return
        self.estopped = True
        self.estop_reason = str(reason)
        if q_hold is not None:
            self._q_estop = np.clip(np.asarray(q_hold, np.float64).reshape(self.D), self.lower, self.upper)
        else:
            self._q_estop = None if self.q_last is None else self.q_last.copy()
        self._event(t, "estop", reason=str(reason))
        if self.estop_callback is not None:
            self.estop_callback(str(reason), float(t))

    # ── per tick ──────────────────────────────────────────────────────────
    def _watch(self, t: float, stamps: Mapping[str, float] | None) -> bool:
        cfg = self.watchdog_cfg
        if cfg is None or not stamps:
            return False
        stale = []
        for name, ts in stamps.items():
            lim = None
            for pat, v in (cfg["max_age_s"] or {}).items():
                if fnmatch.fnmatch(name, str(pat)):
                    lim = float(v)
            if lim is not None and (ts is None or not math.isfinite(float(ts)) or t - float(ts) > lim):
                stale.append(name)
        if stale:
            if not self.stale:
                self._event(t, "stale_on", streams=stale)
                self._stale_since = t
            self.stale = True
            self._count("stale_ticks")
            if cfg.get("estop_after_s") is not None and t - self._stale_since >= float(cfg["estop_after_s"]):
                self.trigger_estop(f"stale sensors {stale}", t)
            return True
        if self.stale:
            self._event(t, "stale_off")
        self.stale, self._stale_since = False, None
        return False

    def _tactile(self, t: float, level: np.ndarray | None, q_hold: np.ndarray) -> None:
        cfg = self.tactile_cfg
        if cfg is None or level is None:
            return
        lv = np.asarray(level).reshape(-1)
        hot = np.isin(lv, self._stop_levels)
        if self._cnt is None or self._cnt.shape != hot.shape:
            self._cnt = np.zeros(hot.shape, dtype=np.int64)
        self._cnt = np.where(hot, self._cnt + 1, 0)
        trig = self._cnt >= int(cfg["min_ticks"])
        if trig.any():
            self._release = 0
            if self.taxel_joints is not None and self.taxel_joints.shape[0] == trig.shape[0]:
                joints = self.taxel_joints[trig].any(0)
            else:
                joints = np.ones(self.D, dtype=bool)
            if not self.stop_active:
                self._event(t, "tactile_stop_on", taxels=[int(i) for i in np.flatnonzero(trig)], mode=cfg["mode"])
                self.stop_active = True
                self._q_freeze = q_hold.copy()
                if cfg["mode"] == "estop":
                    self.trigger_estop("tactile stop", t, q_hold)
            new = joints & ~self._stop_joints
            if new.any() and self._q_freeze is not None:        # freeze newly involved joints where they are now
                self._q_freeze[new] = q_hold[new]
            self._stop_joints |= joints
        elif self.stop_active:
            self._release += 1
            if self._release >= int(cfg["release_ticks"]):
                self._event(t, "tactile_stop_off")
                self.stop_active = False
                self._stop_joints[:] = False
                self._q_freeze = None
                self._release = 0

    def filter(self, q_target: Sequence[float], q_current: Sequence[float] | None = None, *, t: float,
               level: np.ndarray | None = None, stamps: Mapping[str, float] | None = None) -> np.ndarray:
        """Safe command ``[D]`` for this tick. ``q_current`` (measured) seeds the state on the first
        call; ``level`` = the tick's ``ContactLevel`` per taxel; ``stamps`` = ``{stream: last
        sample time}`` for the watchdog (host clock, same as ``t``)."""
        self.n_ticks += 1
        qt = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if qt.shape != (self.D,):
            raise ValueError(f"q_target must have {self.D} entries, got {qt.shape}")
        if self.q_last is None:
            if q_current is None:
                raise ValueError("first filter() call needs q_current (or call reset(q_current))")
            self.reset(q_current)
        prev = self.q_last
        if self.estopped:
            return (self._q_estop if self._q_estop is not None else prev).copy()
        if self._watch(t, stamps):
            if self.estopped:
                return self._q_estop.copy()
            self.v_last = np.zeros(self.D)
            return prev.copy()
        if not np.all(np.isfinite(qt)):
            self._event(t, "nonfinite_target")
            qt = np.where(np.isfinite(qt), qt, prev)
        # freeze where the joints *are* (a position servo lags its command: freezing the last command
        # would keep closing into the object by the tracking lag)
        qm = None if q_current is None else np.asarray(q_current, dtype=np.float64).reshape(-1)
        hold_at = prev if qm is None or qm.shape != prev.shape or not np.all(np.isfinite(qm)) else \
            np.clip(qm, self.lower, self.upper)
        self._tactile(t, level, hold_at)
        if self.estopped:
            return self._q_estop.copy()
        q = qt.copy()
        if self.stop_active and self._q_freeze is not None:
            mode = self.tactile_cfg["mode"]
            J = self._stop_joints
            if mode == "hold":
                q[J] = self._q_freeze[J]
            else:                                                # freeze_closing: opening is allowed
                s = self.closing
                closing = J & (s != 0) & (s * (q - self._q_freeze) > 0)
                q[closing] = self._q_freeze[closing]
                self._count("tactile_frozen_ticks", int(closing.any()))
        c = np.clip(q, self.lower, self.upper)
        self._count("limit_clamp", int(np.any(c != q)))
        q = c
        dq = q - prev
        if self.max_vel is not None:
            lim = self.max_vel * self.dt
            c = np.clip(dq, -lim, lim)
            self._count("vel_clamp", int(np.any(c != dq)))
            dq = c
        if self.max_acc is not None:
            v = dq / self.dt
            lim = self.max_acc * self.dt
            v2 = np.clip(v, self.v_last - lim, self.v_last + lim)
            self._count("acc_clamp", int(np.any(np.abs(v2 - v) > 1e-12)))
            dq = v2 * self.dt
        q = np.clip(prev + dq, self.lower, self.upper)
        self.v_last = (q - prev) / self.dt
        self.q_last = q
        return q.copy()

    def summary(self) -> dict:
        """Counts, e-stop state and the logged transition events (JSON-able)."""
        return {"counts": dict(self.counts), "estop": bool(self.estopped), "estop_reason": self.estop_reason,
                "tactile_stop_active": bool(self.stop_active), "stale": bool(self.stale), "n_ticks": int(self.n_ticks),
                "events": [e.to_dict() for e in self.events]}
