"""Shared command line of ``glove_logger`` / ``robot_logger``: plan → (dry-run | fake | real) → QC.

Modes
- ``--dry-run``: plan only — writes ``session.json`` (single-session plans: planned streams +
  nominal segments) or ``plan.json`` (D2 multi-episode plans) and prints the operator script.
- ``--fake``: synthetic sources (:mod:`robot_skin.acquisition.fake`) on a simulated clock — runs the
  full chain (record → sync → IMU calibration → QC) in seconds; used by tests and for trying the
  pipeline without hardware. ``--time-scale`` shrinks block durations and pauses.
- default: real devices (logger-specific source builders; unimplemented devices raise
  ``NotImplementedError`` with instructions).

Output location: ``--out`` is the session directory for single-session protocols (D1,
``robot_sweep``, protocol-free ``--duration``) and the parent directory for D2 (one directory per
episode). Without ``--out``: ``<--root>/<dataset>/<subject>/<session_id>``. ``--root`` defaults to
``configs/default.yaml`` ``paths.raw_root`` — and for ``--fake`` to ``paths.synthetic_root``: synthetic
sessions never land in the real raw root (where ``preprocess`` would build them next to real data).
A ``--dry-run`` plan goes to ``<root>/<dataset>/<subject>/dry_run``; ``datasets.build`` skips such plans.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Sequence

from .manifest import SessionManifest, StreamInfo
from .protocol import ProtocolError, SessionPlan, format_script, load_protocol, plan_session
from .qc import format_report
from .session import (
    AutoOperator, ConsoleOperator, adhoc_plan, fake_source_factory, run_plan, write_dry_run,
)
from .sources import MonotonicClock, SimClock

log = logging.getLogger(__name__)

#: default number of episodes of a ``--fake`` run of a task catalog (the full catalog is long)
FAKE_DEFAULT_EPISODES = 3


def default_root(fake: bool = False) -> Path:
    """The raw root without ``--root``: ``configs/default.yaml`` ``paths.raw_root``, or for ``--fake``
    ``paths.synthetic_root`` (synthetic data is kept apart from real data, like ``python -m robot_skin synth``)."""
    from ..config import load_config

    paths = load_config().get("paths") or {}
    if fake:
        return Path(paths.get("synthetic_root") or "robot_skin/data/synthetic")
    return Path(paths.get("raw_root") or "robot_skin/data/raw")


class _RootArg(argparse.Action):
    """``--root``: remembers that it was given (``root_given``), so ``--fake`` keeps an explicit root."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, Path(values))
        namespace.root_given = True


def resolve_root(args) -> Path:
    """``--root`` as given, else :func:`default_root` (``paths.synthetic_root`` for ``--fake``)."""
    if getattr(args, "root_given", True) or not getattr(args, "fake", False):
        return Path(args.root)
    return default_root(fake=True)


def _csv_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    return [] if out == ["none"] else out


def base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    g = p.add_argument_group("session")
    g.add_argument("--out", type=Path, default=None,
                   help="session dir (single-session protocols) or parent dir (D2: one dir per episode)")
    g.add_argument("--root", type=Path, default=default_root(), action=_RootArg,
                   help="raw data root when --out is not given: <root>/<dataset>/<subject>/<session_id> "
                        "(default: configs/default.yaml paths.raw_root; with --fake paths.synthetic_root, "
                        "so synthetic sessions stay apart from real data)")
    p.set_defaults(root_given=False)
    g.add_argument("--protocol", default=None, help="protocol name (d1_motion | d2_task | robot_sweep) or YAML path")
    g.add_argument("--subject", default="S00", help="pseudonymous subject id, e.g. S01 (never a real name)")
    g.add_argument("--layout", default=None, help="layout name or YAML path")
    g.add_argument("--notes", default="")
    g.add_argument("--seed", type=int, default=0, help="planning seed (object/target/template/order) and fake data")
    g.add_argument("--time-scale", type=float, default=1.0, help="scale block durations and pauses (e.g. 0.05 for a quick fake run; sync blocks keep their length)")
    g.add_argument("--lang", choices=("ko", "en"), default="ko", help="operator script language")
    g = p.add_argument_group("task catalog (D2)")
    g.add_argument("--task", action="append", default=None, help="task id(s) to record (repeat or comma list)")
    g.add_argument("--object", action="append", default=None, help="restrict objects (repeat or comma list)")
    g.add_argument("--episodes", type=int, default=None, help="number of episodes (default: whole selection)")
    g.add_argument("--repetitions", type=int, default=None, help="repetitions per (task, object)")
    g.add_argument("--instruction", default=None, help="operator instruction text (overrides the templates)")
    g = p.add_argument_group("protocol-free recording")
    g.add_argument("--duration", type=float, default=None, help="seconds (no --protocol)")
    g.add_argument("--no-contact", action="store_true", help="whole recording is contact-free (baseline data)")
    g = p.add_argument_group("mode")
    g.add_argument("--dry-run", action="store_true", help="write the plan (session.json / plan.json) and exit")
    g.add_argument("--fake", action="store_true", help="synthetic sources, simulated clock (no hardware)")
    g.add_argument("--no-sync", action="store_true", help="skip the 3-tap clock sync post-processing")
    g.add_argument("--sync-from", type=Path, default=None, help="apply another session's clock offsets instead")
    g.add_argument("--calibration-from", type=Path, default=None,
                   help="copy the IMU calibration of another session (same sitting) when an episode has none")
    g.add_argument("--no-qc", action="store_true", help="skip QC")
    g.add_argument("--camera-format", choices=("auto", "npy", "jpg"), default="auto")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def _flat(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    for v in values:
        out.extend(_csv_list(v) or [])
    return out


def build_plan(args, *, kind: str, default_protocol: str | None, default_cameras: Sequence[str]) -> tuple[SessionPlan, list[str]]:
    """Plan from the CLI arguments (protocol or protocol-free) + the camera list."""
    cams_arg = _csv_list(getattr(args, "cameras", None))
    name = args.protocol or (None if args.duration is not None else default_protocol)
    if name is None:
        if args.duration is None and (args.fake or args.dry_run):
            raise SystemExit("--fake/--dry-run without --protocol need --duration")
        cams = list(default_cameras if cams_arg is None else cams_arg)
        return adhoc_plan(args.duration, no_contact=args.no_contact, kind=kind, cameras=cams), cams
    proto = load_protocol(name)
    if proto.kind not in (kind, "any"):
        raise SystemExit(f"protocol {proto.name!r} is for kind {proto.kind!r}, this logger records {kind!r}")
    cams = list(proto.cameras if cams_arg is None else cams_arg)
    n_ep = args.episodes
    if proto.is_task and args.fake and n_ep is None:
        n_ep = FAKE_DEFAULT_EPISODES
    try:
        plan = plan_session(proto, seed=args.seed, tasks=_flat(args.task), objects=_flat(args.object), n_episodes=n_ep,
                            repetitions=args.repetitions, time_scale=args.time_scale, instruction=args.instruction)
    except ProtocolError as e:
        raise SystemExit(f"planning failed: {e}") from None
    plan.cameras = tuple(cams) if kind == "glove" else ()
    return plan, cams


def run_logger(args, *, kind: str, default_layout: str, default_protocol: str | None,
               default_cameras: Sequence[str], real_sources: Callable, streams_fn: Callable) -> int:
    """Common main: dry-run / fake / real. ``real_sources(args, plan, cameras) -> (episode → sources)``;
    ``streams_fn(args, plan, cameras) -> {name: StreamInfo}`` for dry-run manifests."""
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.layout = args.layout or default_layout
    args.root = resolve_root(args)
    plan, cams = build_plan(args, kind=kind, default_protocol=default_protocol, default_cameras=default_cameras)
    script = format_script(plan, lang=args.lang, max_episodes=5)
    if args.dry_run:
        out = args.out or (args.root / plan.dataset / args.subject / "dry_run")
        path = write_dry_run(plan, out, kind=kind, layout=args.layout, subject=args.subject,
                             streams=streams_fn(args, plan, cams), notes=args.notes)
        print(script)
        print(f"\n[dry-run] wrote {path}")
        return 0
    meta = {"logger": kind, "fake": bool(args.fake), "cli_seed": args.seed}
    common = dict(kind=kind, out=args.out, root=args.root, subject=args.subject, layout=args.layout, notes=args.notes,
                  meta=meta, camera_format=args.camera_format, sync=not args.no_sync, qc=not args.no_qc,
                  sync_from=args.sync_from, calibration_from=args.calibration_from)
    printed = []

    def show(res: dict) -> None:
        q = res.get("qc")
        print(format_report(q) if q else f"recorded {res['session_dir']}")
        printed.append(res)

    if args.fake:
        if args.out is None and not getattr(args, "root_given", True):
            print(f"[fake] synthetic sessions go to {args.root} (configs/default.yaml paths.synthetic_root), apart "
                  f"from real data — preprocess them with "
                  f"`python -m robot_skin preprocess --raw {args.root} --out <dir>`")
        factory = fake_source_factory(plan, kind=kind, layout=args.layout, cameras=cams, seed=args.seed,
                                      imu=not getattr(args, "no_imu", False))
        results = run_plan(plan, source_factory=factory, clock_factory=SimClock, operator=AutoOperator(),
                           on_result=show, **common)
    else:
        factory = real_sources(args, plan, cams)
        print(script)
        results = run_plan(plan, source_factory=factory, clock_factory=MonotonicClock,
                           operator=ConsoleOperator(lang=args.lang), on_result=show, **common)
    if not results:
        return 1
    failed = [r for r in results if r.get("qc") is not None and not r["qc"]["passed"]]
    print(f"\n{len(results)} session(s) recorded, {len(failed)} failed QC")
    return 3 if failed else 0


# kept for callers of the old stub API
def write_plan(args, kind: str, streams: dict[str, StreamInfo]) -> SessionManifest:
    """Legacy helper: protocol-free planned manifest (``--duration`` / ``--no-contact``)."""
    plan = adhoc_plan(args.duration or 1.0, no_contact=args.no_contact, kind=kind)
    path = write_dry_run(plan, args.out, kind=kind, layout=args.layout, streams=streams, notes=args.notes)
    return SessionManifest.load(path.parent)
