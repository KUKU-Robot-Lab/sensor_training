"""``python -m robot_skin <command>`` — one command line for the whole robot_skin pipeline.

::

    record glove|robot ...  acquisition.glove_logger / robot_logger (arguments passed through)
    postprocess ...         acquisition.session: sync → IMU calibration → QC of recorded sessions
    qc ...                  acquisition.qc
    synth                   datasets.synthetic.generate_dataset (synthetic raw sessions for demos / dry runs)
    preprocess ...          datasets.build (raw sessions → processed episodes)
    splits                  one leakage-safe splits.json over a processed root (datasets.splits)
    train <stage>           one training stage: imu_pose | baseline | contact | pretrain | vtla
    pipeline                splits (once) → imu_pose → baseline → contact → pretrain → vtla, wired, resumable
    deploy                  stages.deploy (policy bundle on a robot hand; robot: fake = simulation)
    env                     train.hardware (GPU / torch / arch-list report, suggested profile)
    sweep ...               train.sweep (hyper-parameter sweeps)

The command is thin: every sub-command delegates to the module that owns the work (its own ``main``
or ``load_stage_config`` + ``run``), so ``python -m robot_skin preprocess ...`` ≡
``python -m robot_skin.datasets.build ...``. Flags are the same everywhere: ``--config`` (YAML),
``--set KEY=VALUE`` (dotted key, value parsed as YAML, repeatable) and ``--hardware`` (a profile
name from ``configs/hardware``, a YAML path or ``auto``). The profile's ``env`` is exported before any
CUDA call; precedence is stage YAML ``train`` < profile ``suggest.<stage>`` / ``train`` < ``--set``.
Defaults (paths, stage config files, pipeline stage list, split policy) come from
``configs/default.yaml`` (:func:`robot_skin.config.load_config`).

``train`` and ``pipeline`` work under ``torchrun`` (``torchrun --standalone --nproc_per_node=2 -m
robot_skin train vtla --hardware rtx5090``): the Trainer joins the process group, only rank 0 writes
files and prints, and the process group is destroyed at exit.

Pipeline (:func:`run_pipeline`): ``<out>/splits.json`` is created once (``make_splits`` over the
processed root, by subject by default) and passed as ``data.splits`` to **every** stage, so no stage
trains on another stage's held-out episodes. Each stage runs with ``out_dir = <out>/<stage>`` and
``data.processed_root`` = the processed root; outputs are wired into later stages — the baseline's
derived residuals → contact, the contact stage's derived ``residual_z`` / ``contact_level`` →
pretrain and vtla (``data.tactile_source: derived``, never the bootstrap fallback), its
``calibrator.json`` and the baseline model → the policy bundle's stage-1 references
(``tactile.calibrator`` / ``tactile.baseline_model``), the pretrained encoder → ``tactile.pretrained``.
Resumable: a stage whose ``metrics.json`` and main artefact exist (built from the same processed root
and splits — its record and its metrics.json ``data_provenance`` agree — and its derived arrays still in
the episodes) is skipped unless ``--force``. Rerun rule: once a stage runs, every later stage runs too
(its inputs changed) — also across invocations: each stage's record keeps the fingerprints
(:func:`stage_fingerprint`: sha256 of metrics.json + main artefact) of the upstream runs it consumed, so
after ``pipeline --stages baseline --force`` (or a standalone ``train baseline`` into ``<out>/baseline``)
the next ``pipeline`` retrains contact, pretrain and vtla. ``train.resume`` is honoured only for an
interrupted attempt on the same data and upstream runs (``<out>/<stage>/pipeline_attempt.json``); a stage
that is *retrained* starts from scratch. The default stage list skips imu_pose (``no_data``) when no episode
has IMUs and hand labels (robot / ``--no-imu`` data) unless a stage uses ``data.q_source: hand_pose_imu``.
Split flags only apply when ``splits.json`` is created; flags that contradict an existing file are an error.
Every selected stage's hardware-profile ``env`` is exported once, before the first stage
(:func:`export_pipeline_env`). ``<out>/pipeline.json`` records what ran, with which config
(``<out>/<stage>/pipeline_config.yaml``), splits (sha256), profile env, wiring and upstream fingerprints;
the summary lists the stages of this invocation and flags unselected ones that are now out of date.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger("robot_skin.cli")

#: training stages in pipeline order
STAGES = ("imu_pose", "baseline", "contact", "pretrain", "vtla")
#: the file (besides metrics.json) that marks a finished stage
STAGE_ARTEFACTS = {"imu_pose": "imu_pose_model.pt", "baseline": "baseline_model.pt", "contact": "calibrator.json",
                   "pretrain": "encoder_state.pt", "vtla": "policy_bundle.pt"}
#: upstream stages whose outputs a stage consumes — through the episodes' derived arrays
#: (``hand_finger_pose_imu`` → baseline with ``q_source: hand_pose_imu``; ``residual`` /
#: ``baseline_logvar`` → contact; ``residual_z`` / ``contact_level`` → pretrain, vtla) or through
#: files (:func:`_wiring`); ``True`` = required
STAGE_INPUTS = {"imu_pose": {}, "baseline": {"imu_pose": False}, "contact": {"baseline": True},
                "pretrain": {"contact": True}, "vtla": {"baseline": False, "contact": True, "pretrain": False}}
#: the derived array each stage-1 stage writes into the episodes (resume: still there?)
STAGE_DERIVED = {"imu_pose": "hand_finger_pose_imu", "baseline": "residual", "contact": "residual_z"}
#: stages whose finished run later stages consume — fingerprinted (:func:`stage_fingerprint`) in pipeline.json
UPSTREAM_STAGES = tuple(s for s in STAGES if any(s in ins for ins in STAGE_INPUTS.values()))
PIPELINE_RECORD = "pipeline.json"
PIPELINE_FORMAT = 2
STAGE_CONFIG_NAME = "pipeline_config.yaml"
#: written into ``<out>/<stage>`` before a stage runs: the data it trains on (``train.resume`` check)
STAGE_ATTEMPT_NAME = "pipeline_attempt.json"
SPLITS_NAME = "splits.json"
#: stage config sections whose keys the stage validates itself (TrainConfig / build_transforms)
_OPEN_SECTIONS = ("train", "image")

#: sub-commands that forward their arguments unchanged to a module's ``main(argv)``
_PASS_THROUGH = {
    "postprocess": "robot_skin.acquisition.session",
    "qc": "robot_skin.acquisition.qc",
    "env": "robot_skin.train.hardware",
    "sweep": "robot_skin.train.sweep",
}
_RECORDERS = {"glove": "robot_skin.acquisition.glove_logger", "robot": "robot_skin.acquisition.robot_logger"}

#: the command table of the module docstring (printed by ``--help``)
_USAGE = __doc__.split("::\n\n", 1)[1].split("\n\n", 1)[0] if __doc__ else ""


# ─────────────────────────────────────────────────────────────── helpers

def _defaults() -> dict:
    from .config import load_config

    return load_config()


def _stage_module(stage: str):
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    return importlib.import_module(f"robot_skin.stages.{stage}")


def stage_config_path(stage: str, explicit: str | Path | None = None, defaults: Mapping | None = None) -> Path | None:
    """The stage YAML: ``explicit`` (``--config``), else ``configs/default.yaml`` ``stages.<stage>``
    (a bare file name — no directory part — is always ``robot_skin/configs/stages/<name>``, never a
    file of that name in the working directory; a path with a directory is used as given), else
    ``None`` (the stage's built-in ``CONFIG_PATH``)."""
    if explicit is not None:
        return Path(explicit)
    ref = ((defaults if defaults is not None else _defaults()).get("stages") or {}).get(stage)
    if not ref:
        return None
    p = Path(ref)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    from .config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.parent / "stages" / p


def export_profile_env(hardware: Any) -> dict | None:
    """Load the hardware profile (name / YAML path / mapping / ``auto``) and export its ``env``
    (``setdefault``) — before the first CUDA call. ``auto`` without a matching profile → ``None``
    (the stage then warns and keeps its defaults)."""
    if not hardware:
        return None
    from .train.hardware import apply_profile_env, load_hw_profile

    try:
        prof = dict(hardware) if isinstance(hardware, Mapping) else load_hw_profile(hardware)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    except ValueError:                          # auto: no built-in profile matches this GPU
        return None
    apply_profile_env(prof)
    return prof


def _stage_hardware(stage: str, overrides: Mapping[str, Any], hardware: Any, cfg_path: str | Path | None) -> Any:
    """The profile ``stage`` runs with in the pipeline — the precedence of :func:`run_pipeline` +
    ``load_stage_config``: ``--set [<stage>.]hardware`` > ``--hardware`` / ``default.yaml`` > the stage YAML's own."""
    if "hardware" in overrides:
        return overrides["hardware"]
    if hardware:
        return hardware
    import yaml

    p = Path(cfg_path) if cfg_path is not None else Path(_stage_module(stage).CONFIG_PATH)
    try:
        y = yaml.safe_load(p.read_text()) or {}
    except OSError:
        return None
    return y.get("hardware") if isinstance(y, Mapping) else None


def export_pipeline_env(stage_hardware: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    """Export the ``env`` of **every** stage's hardware profile once, before ``init_distributed`` and the first CUDA
    call: variables such as ``PYTORCH_CUDA_ALLOC_CONF`` / ``NCCL_*`` are read when the allocator / communicator is
    initialised, so a later stage cannot change them in the same process. A variable already in the environment (the
    user's shell) wins; two profiles that set one variable to different values are an error (``SystemExit``). Returns
    ``{stage: {var: effective value}}`` for the variables each stage's profile sets."""
    from .train.hardware import load_hw_profile

    wanted: dict[str, tuple[str, str]] = {}
    per_stage: dict[str, dict[str, str]] = {}
    for stage, hw in stage_hardware.items():
        env: dict[str, str] = {}
        if hw:
            try:
                prof = dict(hw) if isinstance(hw, Mapping) else load_hw_profile(hw)
            except FileNotFoundError as e:
                raise SystemExit(str(e)) from None
            except ValueError:                  # auto: no built-in profile matches this GPU
                prof = {}
            env = {str(k): str(v) for k, v in (prof.get("env") or {}).items()}
        for k, v in env.items():
            if k in os.environ:
                continue
            if k in wanted and wanted[k][1] != v:
                raise SystemExit(
                    f"pipeline: the hardware profiles of {wanted[k][0]} and {stage} set {k} to {wanted[k][1]!r} and "
                    f"{v!r}; process-level CUDA / NCCL settings cannot change between the stages of one pipeline "
                    f"process — export {k} yourself or run these stages in separate pipeline invocations")
            wanted.setdefault(k, (stage, v))
        per_stage[stage] = env
    for k, (_, v) in wanted.items():
        os.environ[k] = v
    return {s: {k: os.environ.get(k) for k in env} for s, env in per_stage.items()}


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0") or 0)
    except ValueError:
        return 0


def _setup_logging(quiet: bool = False) -> None:
    from .train.logging_utils import setup_logging

    setup_logging(logging.WARNING if quiet else logging.INFO, rank=_rank())


def _parse_items(items: Sequence[str]) -> list[tuple[list[str], Any]]:
    """``["a.b=1", "c=[x, y]"]`` → ``[(["a", "b"], 1), (["c"], ["x", "y"])]`` (values parsed as YAML)."""
    import yaml

    out = []
    for it in items or ():
        key, sep, val = str(it).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects KEY=VALUE, got {it!r}")
        out.append(([k for k in key.strip().split(".")], yaml.safe_load(val)))
    return out


def _nest(items: Sequence[tuple[list[str], Any]]) -> dict:
    from .train.sweep import set_by_path

    out: dict = {}
    for parts, v in items:
        set_by_path(out, ".".join(parts), copy.deepcopy(v))    # a global value is shared by several stages
    return out


def _accepts(defaults: Mapping[str, Any], parts: Sequence[str]) -> bool:
    """Whether a stage whose :data:`DEFAULTS` are ``defaults`` has the dotted key ``parts`` (keys
    below a null / empty-mapping default and inside the open ``train`` / ``image`` sections count)."""
    node: Any = defaults
    for i, k in enumerate(parts):
        if not isinstance(node, Mapping):
            return i > 0
        if k in node:
            node = node[k]
            continue
        return i > 0 and (parts[0] in _OPEN_SECTIONS or not node)
    return True


def _print_json(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _add_common(ap: argparse.ArgumentParser, *, config_help: str, multi_config: bool = False) -> None:
    """The flags shared by ``train`` and ``pipeline``: ``--config``, ``--set``, ``--hardware``, ``-q``."""
    ap.add_argument("--config", action="append" if multi_config else "store", default=None, help=config_help)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, value parsed as YAML (repeatable)")
    ap.add_argument("--hardware", default=None,
                    help="hardware profile: rtx5090 | rtx4090 | rtx3090 | a100 | cpu | auto | a YAML path "
                         "(default: configs/default.yaml `hardware`)")
    ap.add_argument("-q", "--quiet", action="store_true", help="log warnings only")


# ─────────────────────────────────────────────────────────────── splits

def _parse_holdout(items: Sequence[str] | None) -> dict | None:
    """``["subject=S07,S08", "val:task=pour"]`` → ``{"test": {"subject": [...]}, "val": {"task": [...]}}``."""
    if not items:
        return None
    out: dict = {}
    for it in items:
        head, sep, vals = it.partition("=")
        if not sep:
            raise SystemExit(f"--holdout expects [split:]field=v1,v2, got {it!r}")
        split, _, field = head.rpartition(":")
        split = split or "test"
        out.setdefault(split, {})[field] = [v.strip() for v in vals.split(",") if v.strip()]
    return out


def create_splits(processed_root: str | Path, out: str | Path, *, by: str | Sequence[str] = "subject",
                  val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
                  datasets: Sequence[str] | None = ("motion", "task"), holdout: Mapping | None = None) -> dict:
    """:func:`robot_skin.datasets.splits.make_splits` over every episode of ``datasets`` under
    ``processed_root`` → ``out`` (``save_splits``, paths relative to the processed root, so the file
    survives moving the dataset). Returns ``{train, val, test}`` (episode dir strings)."""
    from .datasets.episode import list_episodes
    from .datasets.splits import check_splits, make_splits, save_splits

    root = Path(processed_root)
    if not root.is_dir():
        raise FileNotFoundError(f"processed root {root} does not exist (run `python -m robot_skin preprocess`)")
    dss = list(datasets) if datasets else [None]
    eps = sorted({p for ds in dss for p in list_episodes(root, ds)})
    if not eps:
        raise FileNotFoundError(f"no processed episodes of datasets {list(dss)} under {root}")
    by = by if isinstance(by, str) else tuple(by)
    sp = make_splits(eps, by=by, val_frac=float(val_frac), test_frac=float(test_frac), seed=int(seed),
                     holdout=holdout)
    if not holdout:
        check_splits(sp, by)
    save_splits(sp, out, root=root, meta={
        "by": by, "val_frac": float(val_frac), "test_frac": float(test_frac), "seed": int(seed),
        "datasets": list(datasets) if datasets else None, "holdout": holdout, "n_episodes": len(eps),
        "processed_root": str(root.resolve()), "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "python -m robot_skin splits"})
    return sp


def _file_sha256(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_splits(argv: Sequence[str]) -> int:
    d = _defaults()
    sd = (d.get("pipeline") or {}).get("splits") or {}
    ap = argparse.ArgumentParser(prog="python -m robot_skin splits",
                                 description="Create one leakage-safe splits.json over a processed root "
                                             "(pass it as data.splits to every stage).")
    ap.add_argument("--processed", default=(d.get("paths") or {}).get("processed_root"), help="processed root")
    ap.add_argument("--out", default=None, help="splits file (default <processed>/splits.json)")
    ap.add_argument("--by", default=sd.get("by", "subject"),
                    help="group key (a comma list combines keys): subject | session | object | task | dataset | "
                         "kind | episode_id")
    ap.add_argument("--val-frac", type=float, default=sd.get("val_frac", 0.15))
    ap.add_argument("--test-frac", type=float, default=sd.get("test_frac", 0.15))
    ap.add_argument("--seed", type=int, default=sd.get("seed", 0))
    ap.add_argument("--datasets", default=",".join(sd.get("datasets") or []) or None,
                    help="comma list of datasets to split (default motion,task)")
    ap.add_argument("--holdout", action="append", default=None, metavar="[SPLIT:]FIELD=V1,V2",
                    help="force episodes into a split (default test), e.g. subject=S07 or val:task=pour")
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits file")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.processed) / SPLITS_NAME
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists — splits define what every stage may train on; use --force to replace it")
    by = a.by.split(",") if "," in a.by else a.by
    datasets = [x for x in (a.datasets or "").split(",") if x] or None
    sp = create_splits(a.processed, out, by=by, val_frac=a.val_frac, test_frac=a.test_frac, seed=a.seed,
                       datasets=datasets, holdout=_parse_holdout(a.holdout))
    print(f"wrote {out}: " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))
    return 0


# ─────────────────────────────────────────────────────────────── synth

def cmd_synth(argv: Sequence[str]) -> int:
    d = _defaults()
    sy = d.get("synthetic") or {}
    ap = argparse.ArgumentParser(prog="python -m robot_skin synth",
                                 description="Synthetic raw sessions (datasets.synthetic.generate_dataset) for "
                                             "demos and dry runs of the pipeline — never mix them into real data.")
    ap.add_argument("--out", default=(d.get("paths") or {}).get("synthetic_root", "robot_skin/data/synthetic"),
                    help="raw root: sessions go to <out>/<dataset>/<subject>/<session_id>")
    ap.add_argument("--kind", choices=("glove", "robot"), default=sy.get("kind", "glove"))
    ap.add_argument("--n-motion", type=int, default=sy.get("n_motion", 4), help="D1 motion sessions")
    ap.add_argument("--n-task", type=int, default=sy.get("n_task", 4), help="D2 task sessions")
    ap.add_argument("--subjects", default=",".join(sy.get("subjects") or ["s0", "s1"]), help="comma list")
    ap.add_argument("--seed", type=int, default=sy.get("seed", 0))
    ap.add_argument("--duration", type=float, default=sy.get("duration_s", 8.0), help="seconds per session")
    ap.add_argument("--cameras", default=",".join(sy.get("cameras") or ["ego"]), help="comma list or 'none'")
    ap.add_argument("--image-hw", type=int, nargs=2, default=sy.get("image_hw", [24, 32]), metavar=("H", "W"))
    ap.add_argument("--per-session-skin", action="store_true",
                    help="a different glove / robot skin per session (default: one shared skin, like real data)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing sessions")
    a = ap.parse_args(argv)
    from .datasets.synthetic import generate_dataset

    cams = tuple(c for c in a.cameras.split(",") if c and c != "none")
    dirs = generate_dataset(a.out, n_motion=a.n_motion, n_task=a.n_task, kind=a.kind,
                            subjects=[s for s in a.subjects.split(",") if s], seed=a.seed,
                            shared_glove=not a.per_session_skin, duration_s=a.duration, cameras=cams,
                            image_hw=tuple(a.image_hw), overwrite=a.overwrite)
    print(f"wrote {len(dirs)} synthetic {a.kind} sessions under {a.out}")
    return 0


# ─────────────────────────────────────────────────────────────── train

def run_stage(stage: str, config: str | Path | None = None, overrides: Mapping[str, Any] | None = None,
              hardware: Any = None) -> dict:
    """``stages.<stage>.load_stage_config(config, overrides ⊕ hardware)`` then ``run(cfg)``; the
    profile env is exported first. Returns the stage metrics."""
    ov = copy.deepcopy(dict(overrides or {}))
    if hardware:
        ov["hardware"] = hardware
    export_profile_env(ov.get("hardware"))
    mod = _stage_module(stage)
    cfg = mod.load_stage_config(stage_config_path(stage, config), ov)
    return mod.run(cfg)


def cmd_train(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot_skin train",
                                 description="Train one stage (works under torchrun).")
    ap.add_argument("stage", choices=STAGES)
    _add_common(ap, config_help="stage YAML (default: configs/default.yaml `stages.<stage>`)")
    a = ap.parse_args(argv)
    _setup_logging(a.quiet)
    ov = _nest(_parse_items(a.set))
    # precedence: --hardware > --set hardware=… > configs/default.yaml `hardware` > the stage YAML
    hw = a.hardware or (None if "hardware" in ov else _defaults().get("hardware"))
    from .train.distributed import cleanup

    try:
        metrics = run_stage(a.stage, a.config, ov, hw)
    finally:
        cleanup()
    if _rank() == 0:
        _print_json({k: v for k, v in metrics.items() if k != "skipped"})
    return 0


# ─────────────────────────────────────────────────────────────── pipeline

def _stage_overrides(items: Sequence[tuple[list[str], Any]], stages: Sequence[str]) -> dict[str, dict]:
    """Route ``--set`` items: ``<stage>.<key>=v`` → that stage; any other key → every selected stage
    whose DEFAULTS have it (an error if none does — a typo must not be silently dropped). A
    stage-specific key beats a global one whatever their order; among equals the later one wins.
    Keys of stages that are not selected are ignored."""
    glob: dict[str, list] = {s: [] for s in stages}
    spec: dict[str, list] = {s: [] for s in stages}
    for parts, v in items:
        if parts[0] in STAGES:
            if len(parts) < 2:
                raise SystemExit(f"--set {'.'.join(parts)}=…: give a key below the stage name")
            if parts[0] in spec:
                spec[parts[0]].append((parts[1:], v))
            continue
        hit = [s for s in stages if _accepts(_stage_module(s).DEFAULTS, parts)]
        if not hit:
            raise SystemExit(f"--set {'.'.join(parts)}: no selected stage ({', '.join(stages)}) has this key "
                             f"(prefix a stage name, e.g. vtla.{'.'.join(parts)})")
        for s in hit:
            glob[s].append((parts, v))
    return {s: _nest(glob[s] + spec[s]) for s in stages}


def _stage_configs(configs: Sequence[str] | Mapping[str, Any] | None, defaults: Mapping) -> dict[str, Path | None]:
    """``--config STAGE=YAML`` items or directories holding ``<stage>.yaml`` → ``{stage: path}``."""
    out = {s: stage_config_path(s, None, defaults) for s in STAGES}
    if isinstance(configs, Mapping):
        items = [f"{k}={v}" for k, v in configs.items()]
    else:
        items = list(configs or [])
    for it in items:
        stage, sep, path = str(it).partition("=")
        if sep:
            if stage not in STAGES:
                raise SystemExit(f"--config {it!r}: unknown stage {stage!r}")
            out[stage] = Path(path)
            continue
        d = Path(it)
        if not d.is_dir():
            raise SystemExit(f"--config {it!r}: expected STAGE=YAML or a directory of <stage>.yaml files")
        for s in STAGES:
            if (d / f"{s}.yaml").is_file():
                out[s] = d / f"{s}.yaml"
    return out


def _wiring(stage: str, runs: Path) -> dict:
    """Overrides that feed earlier stages' outputs (when present in ``runs``) into ``stage``."""
    if stage != "vtla":
        return {}
    out: dict = {"tactile": {}}
    cal = runs / "contact" / STAGE_ARTEFACTS["contact"]
    if cal.is_file():
        out["tactile"]["calibrator"] = str(cal)
        out["data"] = {"tactile_source": "derived"}      # the contact stage ran: never the bootstrap fallback
    base = runs / "baseline" / STAGE_ARTEFACTS["baseline"]
    if base.is_file():
        out["tactile"]["baseline_model"] = str(base)
    enc = runs / "pretrain" / STAGE_ARTEFACTS["pretrain"]
    if enc.is_file():
        out["tactile"]["pretrained"] = str(enc)
    if not out["tactile"]:
        del out["tactile"]
    return out


def _flat(d: Mapping, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Mapping) and v:
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _stage_done(stage: str, out_dir: Path) -> bool:
    return (out_dir / "metrics.json").is_file() and (out_dir / STAGE_ARTEFACTS[stage]).is_file()


def _derived_intact(stage: str, out_dir: Path, processed: Path) -> bool:
    """For stages that write derived arrays into the episodes: at least as many episodes of the
    processed root still carry the stage's array as its metrics say it predicted (``preprocess
    --force`` wipes ``derived/``, so finished-looking runs would otherwise feed nothing downstream)."""
    key = STAGE_DERIVED.get(stage)
    if key is None:
        return True
    from .datasets.episode import list_episodes

    try:
        n = int(((json.loads((out_dir / "metrics.json").read_text()).get("n_episodes") or {}).get("predicted")) or 0)
    except (OSError, ValueError, TypeError):
        return False
    return sum((d / "derived" / f"{key}.npy").is_file() for d in list_episodes(processed)) >= n


def stage_fingerprint(stage: str, out_dir: Path) -> str | None:
    """sha256 over a finished stage's ``metrics.json`` and main artefact (:data:`STAGE_ARTEFACTS`): the
    identity of the run whose outputs (files and the episodes' derived arrays) later stages consume.
    ``None`` without a finished run. Any retraining — in the pipeline or a standalone ``train <stage>`` into
    the same directory — rewrites both files and changes it."""
    if not _stage_done(stage, out_dir):
        return None
    h = hashlib.sha256()
    for name in ("metrics.json", STAGE_ARTEFACTS[stage]):
        h.update(name.encode() + b"\0")
        with open(out_dir / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _input_fingerprints(stage: str, runs: Path) -> dict[str, str | None]:
    """``{upstream: fingerprint}`` of the finished runs ``stage`` consumes (:data:`STAGE_INPUTS`) now."""
    return {u: stage_fingerprint(u, runs / u) for u in STAGE_INPUTS[stage]}


def _changed_inputs(stage: str, prev: Mapping[str, Any], runs: Path) -> list[str]:
    """Upstream stages whose finished run is no longer the one ``stage`` consumed when it last ran
    (``prev["inputs"]`` of its pipeline record) — e.g. baseline retrained by an earlier ``pipeline --stages
    baseline --force`` or by a standalone ``train baseline`` into ``<runs>/baseline``. The pipeline's rerun
    rule (an upstream re-run → every later stage re-runs) therefore also holds across invocations.
    ``[]`` for a record without ``inputs`` (written before they were recorded)."""
    rec = prev.get("inputs")
    if not isinstance(rec, Mapping):
        return []
    cur = _input_fingerprints(stage, runs)
    return [u for u in STAGE_INPUTS[stage] if rec.get(u) != cur[u]]


def _out_of_date(stages: Sequence[str], record: Mapping[str, Any], runs: Path,
                 ran: Sequence[str] = ()) -> dict[str, list[str]]:
    """For the finished ``stages``: the upstream stages that changed since they ran (:func:`_changed_inputs`;
    for an older record without ``inputs``: the upstream stages in ``ran``), or that are out of date
    themselves (transitively, in :data:`STAGES` order). Only non-empty entries."""
    out: dict[str, list[str]] = {}
    for s in STAGES:
        if s not in stages or not _stage_done(s, runs / s):
            continue
        prev = record.get("stages", {}).get(s) or {}
        why = _changed_inputs(s, prev, runs) if isinstance(prev.get("inputs"), Mapping) else \
            [u for u in STAGE_INPUTS[s] if u in ran]
        why += [u for u in STAGE_INPUTS[s] if u in out and u not in why]
        if why:
            out[s] = why
    return out


def _foreign_data(out_dir: Path, sha: str, processed: Path) -> str | None:
    """Why the finished results in ``out_dir`` were not trained on the pipeline's data, from their
    ``metrics.json`` ``data_provenance`` (:func:`robot_skin.stages.data_provenance`): another splits file or
    none at all (the stage split its own pool — e.g. a standalone ``train <stage>`` without ``data.splits``
    into the pipeline's directory), or another processed root. ``None`` = the same data, or unknown (no
    provenance recorded)."""
    try:
        prov = json.loads((out_dir / "metrics.json").read_text()).get("data_provenance")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(prov, Mapping):
        return None
    if prov.get("splits_sha256") != sha:
        return f"splits {prov.get('splits') or 'none (it split its own episode pool)'}"
    if prov.get("processed_root") and prov.get("processed_root") != str(processed):
        return f"processed root {prov.get('processed_root')}"
    return None


def _split_value(key: str, v: Any) -> Any:
    if key == "by":
        parts = v.split(",") if isinstance(v, str) else list(v or [])
        return tuple(str(x).strip() for x in parts)
    if key in ("val_frac", "test_frac"):
        return float(v)
    if key == "seed":
        return int(v)
    if key == "datasets":
        return tuple(v) if v else None
    return v


def check_split_flags(split_cfg: Mapping[str, Any], splits_path: Path) -> None:
    """Explicit split settings (``--split-by`` / ``--val-frac`` / ``--test-frac`` / ``--split-seed``) only
    apply when the splits file is created. For an existing file (``<out>/splits.json`` or ``--splits``) a
    setting that differs from the file's recorded ``meta`` is an error (``SystemExit``) instead of a silent
    no-op; one the file does not record is warned about."""
    if not split_cfg:
        return
    try:
        meta = json.loads(Path(splits_path).read_text()).get("meta") or {}
    except (OSError, ValueError, AttributeError):
        meta = {}
    keys = [k for k in split_cfg if k != "file"]
    diff = [k for k in keys if k in meta and _split_value(k, split_cfg[k]) != _split_value(k, meta[k])]
    if diff:
        raise SystemExit(
            f"pipeline: {splits_path} exists and was made with "
            + ", ".join(f"{k}={meta[k]!r}" for k in diff) + " — the split settings you passed ("
            + ", ".join(f"{k}={split_cfg[k]!r}" for k in diff) + ") only apply when the file is created. To "
            f"re-split, delete {splits_path} and run again with --force (every stage is retrained), or use "
            "another --out.")
    unknown = [k for k in keys if k not in meta]
    if unknown:
        log.warning("pipeline: %s does not record %s — the split settings %s are not applied to an existing "
                    "splits file", splits_path, ", ".join(unknown), {k: split_cfg[k] for k in unknown})


def _imu_pose_has_data(cfg: Mapping[str, Any]) -> bool:
    """Whether imu_pose's training datasets hold at least one episode with IMUs and hand labels (robot
    sessions and ``--no-imu`` glove recordings have none)."""
    import numpy as np

    from .datasets.episode import K_HAND_FINGERS, K_HAND_VALID, K_IMU_QUAT, Episode, list_episodes

    d = cfg.get("data") or {}
    if d.get("episodes"):
        return True                                     # an explicit episode list: the stage decides
    root = Path(d.get("processed_root") or ".")
    for ds in d.get("datasets") or []:
        for p in list_episodes(root, ds):
            try:
                ep = Episode.load(p, mmap=True)
            except Exception:                           # noqa: BLE001 - the stage reports unreadable episodes
                continue
            if ep.has(K_IMU_QUAT) and ep.has(K_HAND_FINGERS) and ep.has(K_HAND_VALID) and \
                    bool(np.asarray(ep[K_HAND_VALID]).any()):
                return True
    return False


def _resume_blocker(out_dir: Path, *, retrain: bool, upstream_ran: bool, sha: str, processed: Path,
                    inputs: Mapping[str, Any] | None = None) -> str | None:
    """Why ``train.resume`` must not be honoured for this stage run (``None`` = it may resume). Resuming is only
    meant for an *interrupted* attempt on the same data: a finished stage that is retrained (``--force``, stale or
    lost results, an upstream re-run) would otherwise "resume" its old finished checkpoint — no step is taken and
    the old model (old splits / old upstream arrays) is exported as the new one."""
    if retrain:
        return "its finished run is being retrained (--force, other splits / processed root, lost derived arrays " \
               "or an upstream stage re-ran)"
    if upstream_ran:
        return "an upstream stage re-ran, so its inputs changed"
    try:
        att = json.loads((out_dir / STAGE_ATTEMPT_NAME).read_text())
    except (OSError, ValueError):
        return None                                    # no earlier pipeline attempt recorded
    if att.get("splits_sha256") != sha or att.get("processed_root") != str(processed):
        return "the interrupted attempt was trained on another processed root / splits file"
    if inputs is not None and isinstance(att.get("inputs"), Mapping) and dict(att["inputs"]) != dict(inputs):
        return "an upstream stage was retrained since the interrupted attempt started, so its inputs changed"
    return None


def run_pipeline(processed_root: str | Path, out: str | Path, *, stages: Sequence[str] | None = None,
                 hardware: Any = None, configs: Sequence[str] | Mapping[str, Any] | None = None,
                 overrides: Sequence[str] | None = None, splits: str | Path | None = None,
                 split_cfg: Mapping[str, Any] | None = None, force: bool = False) -> dict:
    """Run the training pipeline (module docstring): splits → ``stages`` in :data:`STAGES` order.

    ``configs``: ``--config`` items (``STAGE=YAML`` or a directory of ``<stage>.yaml``) or a mapping;
    ``overrides``: ``--set`` items (``<stage>.<key>=v`` for one stage, ``<key>=v`` for every stage that
    has the key); ``splits``: an existing splits file instead of ``<out>/splits.json``; ``split_cfg``:
    ``make_splits`` arguments (default ``configs/default.yaml`` ``pipeline.splits``). Returns the
    ``pipeline.json`` record."""
    from .stages import write_json_atomic
    from .train.distributed import barrier, init_distributed

    defaults = _defaults()
    pcfg = defaults.get("pipeline") or {}
    sel = list(stages) if stages else list(pcfg.get("stages") or STAGES)
    bad = [s for s in sel if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {', '.join(STAGES)}")
    sel = [s for s in STAGES if s in sel]
    processed = Path(processed_root).resolve()
    runs = Path(out).resolve()
    if not processed.is_dir():
        raise FileNotFoundError(f"processed root {processed} does not exist (run `python -m robot_skin preprocess`)")
    hardware = hardware if hardware is not None else defaults.get("hardware")
    per_stage = _stage_overrides(_parse_items(overrides or ()), sel)
    cfg_paths = _stage_configs(configs, defaults)
    # every selected stage's profile env, before init_distributed / any CUDA call (a stage's own
    # apply_profile_env later finds its variables set and changes nothing)
    stage_env = export_pipeline_env({s: _stage_hardware(s, per_stage.get(s) or {}, hardware, cfg_paths.get(s))
                                     for s in sel})
    dist = init_distributed()                          # no-op without torchrun

    # ── one splits.json for every stage ────────────────────────────────
    scfg = {**(pcfg.get("splits") or {}), **dict(split_cfg or {})}
    if splits is not None:
        splits_path = Path(splits).resolve()
        if not splits_path.is_file():
            raise FileNotFoundError(f"splits file {splits_path} not found")
        created = False
    else:
        splits_path = runs / str(scfg.get("file") or SPLITS_NAME)
        created = not splits_path.is_file()
        if created and dist.is_main:
            runs.mkdir(parents=True, exist_ok=True)
            sp = create_splits(processed, splits_path, by=scfg.get("by", "subject"),
                               val_frac=scfg.get("val_frac", 0.15), test_frac=scfg.get("test_frac", 0.15),
                               seed=scfg.get("seed", 0), datasets=scfg.get("datasets") or ("motion", "task"),
                               holdout=scfg.get("holdout"))
            log.info("pipeline: wrote %s (%s)", splits_path, ", ".join(f"{k}={len(v)}" for k, v in sp.items()))
        barrier(dist)
    if not created:                                    # explicit split flags must not be silent no-ops
        check_split_flags(dict(split_cfg or {}), splits_path)
    sha = _file_sha256(splits_path)

    rec_path = runs / PIPELINE_RECORD
    record: dict = {}
    if rec_path.is_file():
        try:
            record = json.loads(rec_path.read_text())
        except ValueError:
            record = {}
    record.update({"format": PIPELINE_FORMAT, "processed_root": str(processed), "runs": str(runs),
                   "splits": {"path": str(splits_path), "sha256": sha, "created": created},
                   "hardware": hardware if not isinstance(hardware, Mapping) else dict(hardware)})
    record.setdefault("stages", {})

    upstream_ran = False
    ran: list[str] = []
    not_sel = [s for s in STAGES if s not in sel]
    for stage in sel:
        out_dir = runs / stage
        prev = record["stages"].get(stage) or {}
        # consumed stages this invocation does not run whose results are out of date themselves
        ood = _out_of_date(not_sel, record, runs, ran) if any(u in not_sel for u in STAGE_INPUTS[stage]) else {}
        for u in STAGE_INPUTS[stage]:
            if u in ood:
                log.warning("pipeline: %s consumes %s, whose results are out of date (upstream changed since %s ran: "
                            "%s) — add %s to --stages (or run the whole pipeline)", stage, u, u, ", ".join(ood[u]), u)
        done = _stage_done(stage, out_dir)
        stale = bool(done and prev and (prev.get("splits_sha256") != sha
                                        or prev.get("processed_root") != str(processed)))
        foreign = _foreign_data(out_dir, sha, processed) if done and not (force or stale) else None
        changed = _changed_inputs(stage, prev, runs) if done and not (force or stale or foreign) else []
        lost = done and not (force or upstream_ran or stale or foreign or changed) \
            and not _derived_intact(stage, out_dir, processed)
        if done and not (force or upstream_ran or stale or foreign or changed or lost):
            if not prev:
                log.warning("pipeline: %s has results in %s but no pipeline record — assuming they were trained "
                            "on %s with %s (use --force to retrain)", stage, out_dir, processed, splits_path)
            fp = stage_fingerprint(stage, out_dir) if stage in UPSTREAM_STAGES else None
            if fp and prev.get("fingerprint") and prev["fingerprint"] != fp:
                log.warning("pipeline: %s's results in %s were rewritten since the pipeline recorded them (a "
                            "standalone run into this directory?) — keeping them; every later stage that consumed "
                            "the old ones is retrained", stage, out_dir)
            log.info("pipeline: %s done (%s) — skipped", stage, out_dir / "metrics.json")
            record["stages"][stage] = {**prev, "status": "skipped", "out_dir": str(out_dir),
                                       **({"fingerprint": fp} if fp else {})}
            continue
        if stale:
            log.warning("pipeline: %s was trained on another processed root / splits file — retraining", stage)
        elif foreign:
            log.warning("pipeline: %s's results in %s were not trained on the pipeline's data (%s; a standalone "
                        "`train %s` into this directory?) — retraining", stage, out_dir, foreign, stage)
        elif changed and not (force or upstream_ran):
            log.warning("pipeline: %s consumed an earlier run of %s, which was retrained since — retraining",
                        stage, ", ".join(changed))
        elif lost:
            log.warning("pipeline: the episodes lost %s's derived %r arrays (re-preprocessed?) — retraining",
                        stage, STAGE_DERIVED[stage])
        wired = _wiring(stage, runs)
        upstream = {u: str(runs / u) for u in STAGE_INPUTS[stage] if _stage_done(u, runs / u)}
        inputs = _input_fingerprints(stage, runs)
        missing = [u for u, req in STAGE_INPUTS[stage].items() if req and u not in upstream]
        if missing:
            log.warning("pipeline: %s needs the output of %s, which has no finished run in %s — relying on "
                        "derived arrays already in the episodes (add it to --stages to train it here)",
                        stage, ", ".join(missing), runs)
        ov = {"out_dir": str(out_dir), "train": {"out_dir": str(out_dir)},
              "data": {"processed_root": str(processed), "splits": str(splits_path)}}
        from .config import deep_merge

        ov = deep_merge(deep_merge(ov, wired), per_stage.get(stage) or {})
        if hardware:                                   # a --set [<stage>.]hardware=… wins
            ov.setdefault("hardware", hardware)
        mod = _stage_module(stage)
        cfg = mod.load_stage_config(cfg_paths.get(stage), ov)
        if stage == "imu_pose" and not stages and not _imu_pose_has_data(cfg):
            needs = [s for s in sel if s in ("baseline", "contact") and _q_source(s, cfg_paths, per_stage)
                     == "hand_pose_imu"]
            if not needs:
                # optional downstream (baseline / contact default to data.q_source: q): robot sessions and
                # --no-imu glove recordings have no IMUs — skip it instead of aborting the whole pipeline
                log.warning("pipeline: imu_pose skipped — no episode of %s under %s has IMUs and hand labels "
                            "(robot / --no-imu data); it is only needed for data.q_source: hand_pose_imu. Pass "
                            "--stages to choose the stages explicitly", cfg["data"].get("datasets"), processed)
                record["stages"][stage] = {"status": "no_data", "out_dir": str(out_dir),
                                           "processed_root": str(processed), "splits_sha256": sha,
                                           "finished_utc": datetime.now(timezone.utc).isoformat()}
                continue
        if cfg.get("data", {}).get("splits") != str(splits_path):
            log.warning("pipeline: %s uses data.splits=%s, not the pipeline's %s (explicit override)",
                        stage, cfg.get("data", {}).get("splits"), splits_path)
        resume = (cfg.get("train") or {}).get("resume")
        if resume:
            why = _resume_blocker(out_dir, retrain=done, upstream_ran=upstream_ran, sha=sha, processed=processed,
                                  inputs=inputs)
            if why:
                log.warning("pipeline: %s: ignoring train.resume=%s — %s; training from scratch", stage, resume, why)
                cfg["train"]["resume"] = None
        barrier(dist)                                  # every rank decided before rank 0 rewrites the attempt file
        if dist.is_main:
            import yaml

            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / STAGE_CONFIG_NAME).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            write_json_atomic(out_dir / STAGE_ATTEMPT_NAME, {
                "splits_sha256": sha, "processed_root": str(processed), "inputs": inputs,
                "started_utc": datetime.now(timezone.utc).isoformat()})
        log.info("pipeline: running %s → %s", stage, out_dir)
        t0 = time.perf_counter()
        metrics = mod.run(cfg)
        seconds = time.perf_counter() - t0
        upstream_ran = True
        ran.append(stage)
        if dist.is_main:
            if not (out_dir / "metrics.json").is_file():
                raise RuntimeError(f"pipeline: stage {stage} finished without writing {out_dir / 'metrics.json'}")
            record["stages"][stage] = {
                "status": "ran", "out_dir": str(out_dir), "config": str(out_dir / STAGE_CONFIG_NAME),
                "config_source": str(cfg_paths.get(stage) or mod.CONFIG_PATH), "processed_root": str(processed),
                "splits": str(cfg.get("data", {}).get("splits")), "splits_sha256": sha,
                "upstream": upstream, "inputs": inputs, "wired": _flat(wired), "env": stage_env.get(stage) or {},
                "seconds": round(seconds, 3),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "best": metrics.get("best"), "steps": metrics.get("steps")}
            if stage in UPSTREAM_STAGES:
                record["stages"][stage]["fingerprint"] = stage_fingerprint(stage, out_dir)
            record["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(rec_path, record)
        barrier(dist)
    if dist.is_main:
        # what this invocation selected, and finished stages it did not select that are now out of date
        # (printed by `pipeline`; the next full run retrains them)
        record["last_invocation"] = {
            "stages": sel, "force": bool(force), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "out_of_date": _out_of_date(not_sel, record, runs, ran)}
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(rec_path, record)
    return record


def _q_source(stage: str, cfg_paths: Mapping[str, Path | None], per_stage: Mapping[str, Mapping]) -> Any:
    """``data.q_source`` of ``stage`` as the pipeline would configure it (``None`` if it cannot be loaded)."""
    try:
        cfg = _stage_module(stage).load_stage_config(cfg_paths.get(stage), per_stage.get(stage) or {})
    except Exception:                                  # noqa: BLE001 - the stage reports its config errors
        return None
    return (cfg.get("data") or {}).get("q_source")

def cmd_pipeline(argv: Sequence[str]) -> int:
    d = _defaults()
    paths, sd = d.get("paths") or {}, ((d.get("pipeline") or {}).get("splits") or {})
    ap = argparse.ArgumentParser(
        prog="python -m robot_skin pipeline",
        description="splits → imu_pose → baseline → contact → pretrain → vtla on one processed root, with "
                    "one shared splits.json and every output wired into the later stages; resumable.")
    ap.add_argument("--processed", default=paths.get("processed_root"), help="processed root (datasets.build output)")
    ap.add_argument("--out", default=paths.get("runs_root"), help="runs directory: <out>/<stage>, <out>/splits.json")
    ap.add_argument("--stages", action="append", default=None,
                    help=f"comma list / repeatable subset of {','.join(STAGES)} (always run in that order)")
    _add_common(ap, config_help="per-stage YAML: STAGE=YAML, or a directory of <stage>.yaml files (repeatable); "
                                "default configs/default.yaml `stages`", multi_config=True)
    ap.add_argument("--splits", default=None, help="use this splits.json instead of creating <out>/splits.json")
    ap.add_argument("--split-by", default=None,
                    help=f"make_splits group key (default {sd.get('by', 'subject')}); the split flags only apply "
                         "when <out>/splits.json is created — contradicting an existing file is an error")
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--test-frac", type=float, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run stages whose results exist (keeps splits.json)")
    a = ap.parse_args(argv)
    _setup_logging(a.quiet)
    stages = [s for item in (a.stages or []) for s in item.split(",") if s] or None
    split_cfg = {k: v for k, v in (("by", a.split_by), ("val_frac", a.val_frac), ("test_frac", a.test_frac),
                                   ("seed", a.split_seed)) if v is not None}
    if isinstance(split_cfg.get("by"), str) and "," in split_cfg["by"]:
        split_cfg["by"] = split_cfg["by"].split(",")
    from .train.distributed import cleanup

    try:
        rec = run_pipeline(a.processed, a.out, stages=stages, hardware=a.hardware, configs=a.config,
                           overrides=a.set, splits=a.splits, split_cfg=split_cfg, force=a.force)
    finally:
        cleanup()
    if _rank() == 0:
        print(format_summary(rec))
    return 0


def format_summary(rec: Mapping[str, Any]) -> str:
    """The run summary of ``pipeline``: the stages this invocation selected with their status (``ran`` /
    ``skipped`` / ``no_data``); earlier stages of the record as ``not selected`` — flagged when an upstream
    stage changed since they ran (the next full ``pipeline`` retrains them)."""
    inv = rec.get("last_invocation") or {}
    sel = inv.get("stages") or list(rec.get("stages") or {})
    ood = inv.get("out_of_date") or {}
    lines = [f"pipeline {rec['runs']}  (splits {rec['splits']['path']})"]
    for s in STAGES:
        r = (rec.get("stages") or {}).get(s)
        if r is None:
            continue
        if s not in sel:
            note = (f" — OUT OF DATE (upstream changed since it ran: {', '.join(ood[s])}); the next `pipeline` "
                    f"retrains it (or add it to --stages)") if ood.get(s) else ""
            lines.append(f"  {s:9s} not selected{note}")
            continue
        best = (r.get("best") or {})
        val = best.get("value") if isinstance(best, Mapping) else None
        extra = f"  best {best.get('monitor')}={val:.4g}" if isinstance(val, (int, float)) else ""
        secs = f"  {r['seconds']:.1f} s" if r.get("status") == "ran" and r.get("seconds") is not None else ""
        lines.append(f"  {s:9s} {r.get('status', '?'):8s}{secs}{extra}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────── deploy / pass-through

def _call_main(module: str, argv: Sequence[str]) -> int:
    rc = importlib.import_module(module).main(list(argv))
    return rc if isinstance(rc, int) else 0      # sweep.main returns the ranked results


def _with_default_config(key: str, module: str, argv: Sequence[str]) -> list[str]:
    """Prepend ``--config <configs/default.yaml stages.<key>>`` when no ``--config`` is given and the
    mapping names another file than the module's built-in ``CONFIG_PATH``."""
    argv = list(argv)
    if any(a == "--config" or a.startswith("--config=") for a in argv):
        return argv
    p = stage_config_path(key)
    builtin = getattr(importlib.import_module(module), "CONFIG_PATH", None)
    if p is None or (builtin is not None and p.resolve() == Path(builtin).resolve()):
        return argv
    return ["--config", str(p), *argv]


def _preprocess_paths(argv: Sequence[str]) -> list[str]:
    """``--raw`` / ``--out`` from configs/default.yaml ``paths.raw_root`` / ``paths.processed_root`` for
    whatever neither the command line nor the preprocess config (``--config`` YAML / ``--set``) sets —
    so ``preprocess`` reads and writes where ``record`` and ``pipeline`` do."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--raw", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a, _ = ap.parse_known_args(list(argv))
    from .datasets.build import DEFAULTS, _parse_set, load_preprocess_config

    try:
        cfg = load_preprocess_config(a.config, _parse_set(a.set))
    except Exception:                                   # noqa: BLE001 - datasets.build reports it
        return []
    paths = _defaults().get("paths") or {}
    extra: list[str] = []
    if not a.raw and cfg["raw_root"] == DEFAULTS["raw_root"] and paths.get("raw_root"):
        extra += ["--raw", str(paths["raw_root"])]
    if a.out is None and cfg["out_root"] == DEFAULTS["out_root"] and paths.get("processed_root"):
        extra += ["--out", str(paths["processed_root"])]
    return extra


def cmd_preprocess(argv: Sequence[str]) -> int:
    """``datasets.build.main`` (the config default follows configs/default.yaml ``stages.preprocess``;
    ``--raw`` / ``--out`` default to configs/default.yaml ``paths.raw_root`` / ``paths.processed_root``
    unless the preprocess config sets ``raw_root`` / ``out_root``)."""
    mod = "robot_skin.datasets.build"
    argv = _with_default_config("preprocess", mod, argv)
    if any(x in ("-h", "--help") for x in argv):
        return _call_main(mod, argv)
    return _call_main(mod, [*argv, *_preprocess_paths(argv)])


def cmd_deploy(argv: Sequence[str]) -> int:
    """``stages.deploy.main`` with ``--hardware X`` turned into the stage's own ``--set hardware=X``
    (default: configs/default.yaml ``hardware`` unless ``--set hardware=…`` is given; the profile
    env is exported first) and the config default of configs/default.yaml ``stages.deploy``."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--hardware", default=None)
    a, rest = ap.parse_known_args(list(argv))
    hw = a.hardware
    given = any(x.startswith("hardware=") for x in rest)
    if hw is None and not given:
        hw = _defaults().get("hardware")
    if hw:
        export_profile_env(hw)
        rest = [*rest, "--set", f"hardware={hw}"]
    mod = "robot_skin.stages.deploy"
    return _call_main(mod, _with_default_config("deploy", mod, rest))


def cmd_record(argv: Sequence[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m robot_skin record glove|robot [logger arguments]  "
              "(see python -m robot_skin record glove --help)")
        return 0 if argv else 2
    kind = argv[0]
    if kind not in _RECORDERS:
        raise SystemExit(f"record: unknown kind {kind!r}; choose glove or robot")
    return _call_main(_RECORDERS[kind], argv[1:])


_COMMANDS = {"record": cmd_record, "synth": cmd_synth, "preprocess": cmd_preprocess, "splits": cmd_splits,
             "train": cmd_train, "pipeline": cmd_pipeline, "deploy": cmd_deploy}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``python -m robot_skin <command> ...`` (module docstring)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: python -m robot_skin <command> [args]\n\n" + _USAGE +
              "\n\n`python -m robot_skin <command> --help` shows the arguments of one command.")
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd in _PASS_THROUGH:
        return _call_main(_PASS_THROUGH[cmd], rest)
    fn = _COMMANDS.get(cmd)
    if fn is None:
        raise SystemExit(f"unknown command {cmd!r}; commands: "
                         f"{', '.join(sorted([*_COMMANDS, *_PASS_THROUGH]))}")
    return fn(rest)


if __name__ == "__main__":
    sys.exit(main())
