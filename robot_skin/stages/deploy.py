"""Stage ``deploy``: run a trained VTLA policy bundle on a robot hand (closed loop, safety, logging).

Pipeline position: after ``vtla`` (``policy_bundle.pt``) and the robot's own stage-1 models
(``baseline_model.pt`` / ``calibrator.json`` of the robot skin — the bundle's references belong to
the skin the policy was trained on, usually the glove). ``run(cfg) -> metrics``
(``robot_skin/configs/stages/deploy.yaml``; unknown keys raise):

- ``robot: fake`` — :class:`~robot_skin.control.interfaces.FakeRobotHand` (synthetic 16-DoF hand +
  synthetic skin with motion artefact and a virtual object) and :class:`~robot_skin.control.
  interfaces.FakeCamera` s for the bundle's cameras, on a simulated clock (``realtime: false``:
  deterministic, faster than real time). Used by tests and for bring-up of the software path.
- any other ``robot`` value without an instance passed to ``run(cfg, robot=...)`` raises
  ``NotImplementedError`` with instructions: implement
  :class:`~robot_skin.control.interfaces.RobotHandInterface` for the hand (``docs/DEPLOYMENT.md`` §5).

Steps: load the bundle (:func:`robot_skin.control.load_policy_bundle`; bootstrap-trained bundles are
refused unless ``allow_bootstrap``) → tactile processor (:class:`~robot_skin.control.
OnlineTactileProcessor`: robot stage-1 models from ``stage1.*``, bundle references only when they
belong to this skin) → retargeter for ``hand_mano`` bundles (:func:`build_retargeter`) → safety
filter (limits, velocity / acceleration, tactile stop per kinematic chain, sensor watchdog, e-stop
→ ``robot.estop``) → :class:`~robot_skin.control.PolicyRunner` (start-up baseline capture and
optional bring-up calibration, rollout for ``duration_s``) → deployment session log (RAW format,
re-ingestible) → optional policy latency benchmark → ``<out_dir>/metrics.json``.

Metrics: ``loop_hz`` (simulated / host clock) and ``loop_hz_wall``, ``latency_p50_ms`` /
``latency_p95_ms`` (policy inference), ``tick_p50_ms`` / ``tick_p95_ms`` (whole control tick),
``overruns``, ``safety_counts`` / ``safety_events`` / ``estop``, ``contact_frac``, ``n_policy_ticks``,
``session_dir`` and ``benchmark`` (``latency.benchmark_policy``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import check_stage_keys, finite_json, parse_overrides, write_json_atomic

log = logging.getLogger("robot_skin.stages.deploy")

STAGE = "deploy"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "deploy.yaml"
METRICS_NAME = "metrics.json"

#: built-in defaults (mirrored by configs/stages/deploy.yaml; a test keeps them in sync)
DEFAULTS: dict[str, Any] = {
    "stage": STAGE,
    "hardware": None,
    "hardware_applied": False,
    "out_dir": None,
    "robot": "fake",
    "bundle": None,
    "device": "cpu",
    "duration_s": 5.0,
    "control_hz": None,
    "policy_hz": None,
    "instruction": None,
    "seed": 0,
    "allow_bootstrap": False,
    "realtime": False,
    "layout": None,
    "urdf": None,
    "stage1": {"baseline_model": None, "calibrator": None, "detector": None, "use_bundle_refs": True},
    "startup": {"baseline_s": 1.0, "calib_s": 0.5},
    "tactile": {"adc_min": 0.0, "adc_max": 16777215.0, "rail_margin": 0.0, "max_abs_pct": 90.0, "fsm": None},
    "fake_robot": {"sim_hz": 400.0, "tau_s": 0.05, "seed": 0, "noise_pct": 0.02, "n_channels": None,
                   "object": {"angle": 0.9, "compliance_rad": 0.25, "palm": True}},
    "cameras": {"hw": [24, 32], "rate_hz": 30.0},
    "retarget": {"tip_links": None, "base_link": None, "human_to_robot": None, "scale": "auto",
                 "tip_offsets": "layout", "method": "lm", "iters": 10, "jacobian": "fd",
                 "reg_weight": 1.0e-6, "smooth_weight": 1.0e-5},
    "hand_state": {"init": "mean", "estimate": False},
    "safety": {"max_vel": 3.0, "max_acc": None, "margin": 0.0, "closing_sign": 1.0, "per_taxel_joints": True,
               "tactile_stop": {"enabled": True, "levels": ["STRONG", "SATURATED"], "min_ticks": 10,
                                "release_ticks": 20, "mode": "freeze_closing"},
               "watchdog": {"enabled": True, "max_age_s": {"pressure": 0.05, "joint_state": 0.05, "camera_*": 0.5},
                            "estop_after_s": 0.5}},
    "log": {"enabled": True, "dir": None, "dataset": "other", "subject": "robot", "session_id": None,
            "camera_format": "auto"},
    "latency": {"benchmark": True, "n": 20, "warmup": 3},
}

REAL_ROBOT_HELP = (
    "robot {robot!r}: robot_skin ships no driver for real hands. Implement "
    "robot_skin.control.interfaces.RobotHandInterface for your hand (joint_names, lower, upper, "
    "read_state() -> (t, q, qd), read_pressure() -> (t, raw[C]) in the tactile front-end's channel order, "
    "send_joint_targets(q); optional estop(), layout, urdf_xml / urdf_path; host-clock timestamps) and call "
    "robot_skin.stages.deploy.run(cfg, robot=MyHand(...), cameras={{'ego': MyCamera(...)}}) — see "
    "docs/DEPLOYMENT.md §5 (bring-up checklist) and robot_skin/control/README.md. Use robot: fake to exercise "
    "the whole software path in simulation.")


def load_stage_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/deploy.yaml``) ⊕ overrides; unknown keys
    raise. (A ``hardware`` profile only exports its environment and supplies ``device: auto``, at
    :func:`run` time — deployment has no training section.)"""
    import copy

    import yaml

    from ..config import deep_merge

    p = Path(path) if path is not None else CONFIG_PATH
    if p.is_file():
        cfg = deep_merge(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    elif path is not None:
        raise FileNotFoundError(f"stage config not found: {p}")
    else:
        cfg = copy.deepcopy(DEFAULTS)
    cfg = deep_merge(cfg, dict(overrides or {}))
    check_stage_keys(cfg, DEFAULTS, STAGE, open_sections=())
    return cfg


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    """Merge over :data:`DEFAULTS`, validate keys, default ``out_dir`` to ``robot_skin/runs/deploy``."""
    from ..config import deep_merge

    out = deep_merge(DEFAULTS, dict(cfg or {}))
    check_stage_keys(out, DEFAULTS, STAGE, open_sections=())
    out["out_dir"] = str(out.get("out_dir") or "robot_skin/runs/deploy")
    return out


def _apply_hardware(cfg: dict) -> str:
    """Export the ``hardware`` profile's environment (before CUDA init) and resolve ``device``."""
    dev = str(cfg.get("device") or "cpu")
    hw = cfg.get("hardware")
    prof = None
    if hw and not cfg.get("hardware_applied"):
        from ..train.hardware import apply_profile_env, load_hw_profile

        prof = dict(hw) if isinstance(hw, Mapping) else load_hw_profile(hw)
        apply_profile_env(prof)
        cfg["hardware_applied"] = True
    if dev == "auto":
        from ..train.hardware import resolve_device

        pref = ((prof or {}).get("train") or {}).get("device", "auto")
        dev = str(resolve_device(pref))
    return dev


# ─────────────────────────────────────────────────────────────── builders

def build_retargeter(model: Any, layout: Any, cfg: Mapping[str, Any] | None = None, *,
                     synthetic: bool = False):
    """:class:`~robot_skin.action.FingertipRetargeter` for a URDF hand (``retarget`` section):
    ``tip_links`` (default ``{finger: <finger>_distal_link}`` for links that exist), fingertip points
    = tip link ⊕ ``tip_offsets`` (``layout``: the mean fingertip-pad taxel position on that link,
    a dict, or null), ``human_to_robot`` (default :data:`~robot_skin.control.interfaces.
    SYNTHETIC_HAND_HUMAN_TO_ROBOT` for the synthetic hand, else identity with a warning), ``scale``
    (``auto``: :meth:`~robot_skin.action.FingertipRetargeter.estimate_scale` between the flat MANO
    hand and the robot at q = 0 — the AnyTeleop convention, the factor multiplies human vectors)."""
    from ..action.retarget import FingertipRetargeter, human_fingertips
    from ..control.interfaces import FINGERS, SYNTHETIC_HAND_HUMAN_TO_ROBOT, layout_tip_offsets, urdf_tip_fk

    c = dict(cfg or {})
    tips = c.get("tip_links") or {f: f"{f}_distal_link" for f in FINGERS if f"{f}_distal_link" in model.link_names}
    if not tips:
        raise ValueError("retarget.tip_links: no <finger>_distal_link in the URDF — give {finger: link}")
    offs = c.get("tip_offsets", "layout")
    if offs == "layout":
        offs = layout_tip_offsets(layout, tips)
    R = c.get("human_to_robot")
    if R is None:
        if not synthetic:
            warnings.warn("retarget.human_to_robot not set: assuming the robot base frame equals the MANO hand "
                          "frame (almost never true — set the 3x3 rotation)", stacklevel=2)
        R = SYNTHETIC_HAND_HUMAN_TO_ROBOT if synthetic else np.eye(3)
    base = c.get("base_link")
    fk = urdf_tip_fk(model, tips, offs, base)
    kw = {k: c[k] for k in ("method", "iters", "jacobian", "reg_weight", "smooth_weight") if c.get(k) is not None}
    rt = FingertipRetargeter(fk, {f: f for f in tips}, model.lower, model.upper, base_link=base,
                             human_to_robot=np.asarray(R, np.float64), **kw)
    rt.joint_names = tuple(model.joint_names)
    scale = c.get("scale", "auto")
    if scale == "auto" or scale is None:
        flat = human_fingertips(np.zeros((15, 3)))
        scale = rt.estimate_scale(flat, q_ref=np.clip(np.zeros(model.n_dof), model.lower, model.upper))
    rt.scale = float(scale)
    return rt


def _baseline_fits(bmodel: Any, layout: Any, model: Any, robot_joints: Sequence[str] | None) -> str | None:
    """Why a loaded baseline model cannot drive this robot skin (``None`` = it fits): taxel count,
    a glove-kind model, or joints the robot does not have (its q columns must all be robot joints)."""
    meta = getattr(bmodel, "bundle_meta", {}) or {}
    if int(bmodel.n_taxels) != layout.n:
        return f"{bmodel.n_taxels} taxels, layout {layout.n}"
    if meta.get("kind") and meta.get("kind") != "robot":
        return f"trained on {meta.get('kind')!r} data"
    names = [str(n) for n in (meta.get("joint_names") or [])]
    avail = [str(n) for n in (robot_joints or (model.joint_names if model is not None else []))]
    if names and avail and not set(names) <= set(avail):
        return f"joints {sorted(set(names) - set(avail))} not on the robot"
    if not names and avail and int(bmodel.joint_dim) != len(avail):
        return f"{bmodel.joint_dim} joints, robot {len(avail)}"
    return None


def build_processor(cfg: Mapping[str, Any], bundle: Any, layout: Any, model: Any = None, *, device: Any = "cpu",
                    robot_joints: Sequence[str] | None = None):
    """The robot skin's :class:`~robot_skin.control.OnlineTactileProcessor` (``stage1`` / ``tactile``
    / ``startup`` sections): explicit ``stage1`` paths win; the bundle's embedded calibrator and
    baseline reference are used only when ``use_bundle_refs`` and they match the robot layout
    (:meth:`~robot_skin.control.PolicyBundle.stage1_matches`). Artefacts that do not fit the skin /
    robot (taxel count, glove-kind or foreign-joint baseline, a log-variance calibrator without its
    baseline model) are dropped with a note — start-up calibration (``startup.calib_s``) then
    stands in."""
    from ..baseline.temporal import load_baseline_model
    from ..control.online import OnlineTactileProcessor, load_calibrator

    s1 = cfg["stage1"]
    notes = []
    cal = load_calibrator(s1.get("calibrator"))
    base = s1.get("baseline_model")
    use_refs = bool(s1.get("use_bundle_refs", True))
    if use_refs and (cal is None or base is None):
        if bundle.stage1_matches(layout):
            if cal is None:
                cal = bundle.calibrator()
            if base is None:
                base = bundle.baseline_model_path()
                if base is None and (bundle.tactile or {}).get("baseline_model"):
                    notes.append(f"bundle baseline model {bundle.tactile['baseline_model']!r} not found")
        else:
            notes.append(f"bundle stage-1 references belong to layouts {bundle.tactile.get('layouts')}, not "
                         f"{layout.name!r}: not used (set stage1.calibrator / baseline_model for the robot skin)")
    if cal is not None and cal.n_taxels != layout.n:
        notes.append(f"calibrator has {cal.n_taxels} taxels, layout {layout.n}: ignored")
        cal = None
    bmodel = None
    if base is not None:
        bmodel = load_baseline_model(base, map_location=device)
        why = _baseline_fits(bmodel, layout, model, robot_joints)
        if why is not None:
            notes.append(f"baseline model {base} does not fit the robot skin ({why}): ignored")
            bmodel = None
    if bmodel is None:
        notes.append("no motion-artefact baseline model for the robot skin: residual = ΔS (train the baseline "
                     "stage on robot D1 data)")
        if cal is not None and cal.use_logvar:
            notes.append("calibrator fitted with the baseline log-variance cannot run without its baseline model: "
                         "ignored (start-up calibration stands in when startup.calib_s > 0)")
            cal = None
    tc = dict(cfg["tactile"])
    fsm = tc.pop("fsm", None)
    det = s1.get("detector")
    proc = OnlineTactileProcessor.from_stage_outputs(
        layout, baseline=bmodel, calibrator=cal, detector=det, device=device, urdf=model, fsm=fsm, pressure=tc,
        feature_spec=bundle.feature_spec, contact_rule=bundle.contact_rule, raw_order="channel",
        baseline_s=float(cfg["startup"]["baseline_s"]),
        hz=None if bmodel is not None else bundle.source_hz)
    return proc, notes


def build_safety(cfg: Mapping[str, Any], robot: Any, layout: Any, model: Any, dt: float):
    """:class:`~robot_skin.control.SafetyFilter` from the ``safety`` section (per-taxel joint masks
    from the URDF chains when ``per_taxel_joints``; e-stop → ``robot.estop`` when available)."""
    from ..control.safety import SafetyFilter, taxel_joint_mask

    s = cfg["safety"]
    mask = None
    if s.get("per_taxel_joints") and model is not None and layout.parent_frame == "urdf":
        mask = taxel_joint_mask(layout, model, robot.joint_names)
    tstop = s.get("tactile_stop")
    wd = s.get("watchdog")
    return SafetyFilter(robot.lower, robot.upper, dt=dt, max_vel=s.get("max_vel"), max_acc=s.get("max_acc"),
                        margin=float(s.get("margin") or 0.0),
                        tactile_stop=None if not tstop or not tstop.get("enabled", True) else tstop,
                        closing_sign=s.get("closing_sign", 1.0), taxel_joints=mask,
                        watchdog=None if not wd or not wd.get("enabled", True) else wd,
                        estop_callback=(lambda reason, t: robot.estop()) if hasattr(robot, "estop") else None,
                        joint_names=list(robot.joint_names))


# ─────────────────────────────────────────────────────────────── run

def run(cfg: Mapping[str, Any] | None = None, *, robot: Any = None, cameras: Any = None) -> dict:
    """Deploy (see module docstring). ``robot`` / ``cameras``: hardware instances
    (:class:`~robot_skin.control.interfaces.RobotHandInterface` / ``{name: CameraInterface}``);
    without them ``cfg.robot`` must be ``fake``."""
    import torch

    from common.layouts import load_layout

    from ..acquisition.sources import MonotonicClock, SimClock
    from ..control.bundle import load_policy_bundle
    from ..control.interfaces import FakeCamera, FakeRobotHand, check_robot
    from ..control.latency import benchmark_policy
    from ..control.runner import DeploymentLogger, PolicyRunner

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    notes: list[str] = []
    if robot is None and str(cfg["robot"]) != "fake":
        raise NotImplementedError(REAL_ROBOT_HELP.format(robot=cfg["robot"]))
    if not cfg["bundle"]:
        raise ValueError("deploy: set `bundle` to a policy_bundle.pt (or the vtla run directory)")
    torch.manual_seed(int(cfg["seed"]))
    device = _apply_hardware(cfg)
    bundle = load_policy_bundle(cfg["bundle"], device=device, allow_bootstrap=bool(cfg["allow_bootstrap"]))
    bundle.check_deployable(allow_bootstrap=bool(cfg["allow_bootstrap"]))

    # ── robot + cameras ─────────────────────────────────────────────────
    synthetic = False
    if robot is None:
        clock = MonotonicClock() if cfg["realtime"] else SimClock(0.0)
        fr = dict(cfg["fake_robot"])
        obj = fr.pop("object", None)
        robot = FakeRobotHand(cfg["urdf"], cfg["layout"] or "robot_hand_template", clock=clock, obj=obj, **fr)
        synthetic = cfg["urdf"] is None
        ccfg = cfg["cameras"]
        cameras = {c: FakeCamera(c, robot, hw=tuple(ccfg["hw"]), rate_hz=float(ccfg["rate_hz"]), seed=i)
                   for i, c in enumerate(bundle.cameras)}
        notes.append("robot: fake (FakeRobotHand + FakeCamera, " + ("simulated clock)" if not cfg["realtime"]
                                                                    else "real-time clock)"))
    check_robot(robot)
    clock = getattr(robot, "clock", None) or MonotonicClock()
    layout = getattr(robot, "layout", None) or (load_layout(cfg["layout"]) if cfg["layout"] else None)
    if layout is None:
        raise ValueError("deploy: the skin layout is unknown — give robot.layout or cfg.layout")
    if not hasattr(layout, "n"):
        layout = load_layout(layout)
    model = getattr(robot, "model", None)
    urdf_xml = getattr(robot, "urdf_xml", None)
    if model is None and (cfg["urdf"] or getattr(robot, "urdf_path", None)):
        from ..pose.urdf import URDFModel

        up = Path(cfg["urdf"] or robot.urdf_path)
        model, urdf_xml = URDFModel.from_file(up), (urdf_xml or up.read_text())

    # ── tactile processor, retargeter, safety ──────────────────────────────
    proc, pnotes = build_processor(cfg, bundle, layout, model, device=device, robot_joints=list(robot.joint_names))
    notes += pnotes
    control_hz = float(cfg["control_hz"] or bundle.source_hz)
    retargeter = hand_state_fn = None
    if bundle.action_kind == "hand_mano":
        if model is None:
            raise ValueError("deploy: a hand_mano policy needs the robot URDF for retargeting (cfg.urdf)")
        retargeter = build_retargeter(model, layout, cfg["retarget"], synthetic=synthetic)
        notes.append(f"retarget scale {retargeter.scale:.3f} (robot/human)")
        if cfg["hand_state"].get("estimate"):
            from ..transfer.reverse import RobotToManoEstimator

            hand_state_fn = RobotToManoEstimator(retargeter)
    safety = build_safety(cfg, robot, layout, model, 1.0 / control_hz)

    # ── logger ──────────────────────────────────────────────────────────
    logger = None
    lc = cfg["log"]
    instruction = cfg["instruction"] if cfg["instruction"] is not None else \
        str((bundle.meta or {}).get("instruction") or "")
    if lc["enabled"]:
        sid = lc["session_id"] or f"deploy_{int(cfg['seed'])}"
        sdir = Path(lc["dir"]) if lc["dir"] else out_dir / "sessions" / sid
        logger = DeploymentLogger(sdir, layout=layout, joint_names=robot.joint_names, clock=clock,
                                  cameras=list(cameras or {}), dataset=lc["dataset"], subject=lc["subject"],
                                  session_id=sid, instruction=instruction, urdf_xml=urdf_xml,
                                  camera_format=lc["camera_format"], overwrite=True,
                                  meta={"bundle": str(bundle.path), "robot": str(cfg["robot"]),
                                        "control_hz": control_hz, "policy_hz": cfg["policy_hz"] or bundle.policy_hz},
                                  pressure_hz=control_hz, joint_hz=control_hz,
                                  camera_hz=float(cfg["cameras"]["rate_hz"]))

    runner = PolicyRunner(robot, bundle, proc, cameras=cameras, retargeter=retargeter, safety=safety,
                          control_hz=control_hz, policy_hz=cfg["policy_hz"], instruction=instruction, clock=clock,
                          logger=logger, hand_state_fn=hand_state_fn, hand_state_init=cfg["hand_state"]["init"],
                          device=device, seed=int(cfg["seed"]), allow_bootstrap=bool(cfg["allow_bootstrap"]))
    st = cfg["startup"]
    m = runner.run(float(cfg["duration_s"]), baseline_s=float(st["baseline_s"]), calib_s=float(st["calib_s"] or 0.0))

    metrics: dict[str, Any] = {"stage": STAGE, "robot": str(cfg["robot"]), "out_dir": str(out_dir),
                               "bundle": bundle.summary(), "processor": repr(proc), "notes": notes}
    metrics.update({k: v for k, v in m.items() if k != "safety"})
    metrics["safety_events"] = m["safety"]["events"]
    metrics["safety_summary"] = {k: v for k, v in m["safety"].items() if k != "events"}
    if hasattr(robot, "truth"):
        metrics["fake_truth"] = {"final_q": robot.truth()["q"].tolist(),
                                 "penetration_rad": robot.truth()["penetration_rad"]}
    lat = cfg["latency"]
    if lat["benchmark"]:
        metrics["benchmark"] = benchmark_policy(bundle, device=device, n=int(lat["n"]), warmup=int(lat["warmup"]),
                                                seed=int(cfg["seed"]), n_taxels=layout.n)
    metrics = finite_json(metrics)
    write_json_atomic(out_dir / METRICS_NAME, metrics)
    log.info("deploy done: loop %.1f Hz, inference p50 %.2f ms / p95 %.2f ms, safety %s", m["loop_hz"],
             m["latency_p50_ms"], m["latency_p95_ms"], m["safety_counts"])
    return metrics


# ─────────────────────────────────────────────────────────────── CLI

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.stages.deploy --set bundle=runs/vtla [--config yaml] [--set robot=fake]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.deploy",
                                 description="Run a VTLA policy bundle on a (fake) robot hand.")
    ap.add_argument("--config", default=None, help=f"stage YAML (default {CONFIG_PATH})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, value parsed as YAML (repeatable)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    metrics = run(load_stage_config(args.config, parse_overrides(args.set)))
    json.dump({k: v for k, v in metrics.items() if k not in ("safety_events",)}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
