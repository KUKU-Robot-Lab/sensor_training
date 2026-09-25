"""Shared argparse/dry-run scaffolding for the logger CLI stubs."""
from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import SessionManifest, StreamInfo


def base_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--out", type=Path, required=True, help="session directory to create")
    p.add_argument("--layout", default=None, help="layout name or YAML path")
    p.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl-C)")
    p.add_argument("--no-contact", action="store_true",
                   help="whole session is contact-free (baseline training data)")
    p.add_argument("--notes", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="write session.json with the planned streams and exit (no hardware)")
    return p


def write_plan(args, kind: str, streams: dict[str, StreamInfo]) -> SessionManifest:
    m = SessionManifest(kind=kind, layout=args.layout, streams=streams, notes=args.notes,
                        meta={"dry_run": True, "duration_s": args.duration})
    if args.no_contact and args.duration:
        m.add_segment(0.0, args.duration, "no_contact")
    m.save(args.out)
    return m
