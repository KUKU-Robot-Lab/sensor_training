#!/usr/bin/env python3
"""Analyze SATS raw BIN trial sufficiency.

Quick mode checks file presence, record counts, and approximate source rates.
Full mode performs the same 200 Hz in-memory merge used by preprocessing and
summarizes z/Fz/sequence coverage without writing large GT files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sats.preprocessing.bin_merge import (
    DUE_RECORD_STRUCT,
    DUE_PAYLOAD_SIZE,
    LOADCELL_RECORD_STRUCT,
    build_merged_rows,
    find_bin_set,
    load_due_bin,
    load_ethermotion_bin,
    load_loadcell_bin,
    read_bin_header,
)
from sats.preprocessing.prepare_learning_data import D_DIR_RE, SOURCE_TEST_DIR_RE
from sats.preprocessing.raw_merge import z_start_for_indenter


def _fixed_record_count(path: Path, offset: int, record_bytes: int) -> int:
    payload_bytes = path.stat().st_size - offset
    return payload_bytes // record_bytes if payload_bytes >= 0 else 0


def _source_summary(path: Path, kind: str) -> dict:
    magic, header, offset = read_bin_header(path)
    raw_record_bytes = header.get("record_bytes", 0)
    try:
        record_bytes = int(raw_record_bytes)
    except (TypeError, ValueError):
        record_bytes = 0
    if kind == "due":
        record_bytes = record_bytes or (DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE)
        records = _fixed_record_count(path, offset, record_bytes)
        rows = records * 10
    elif kind == "ethermotion":
        record_bytes = record_bytes or 40
        records = _fixed_record_count(path, offset, record_bytes)
        rows = records
    else:
        # Loadcell records are variable-size, so quick mode reports bytes only.
        record_bytes = LOADCELL_RECORD_STRUCT.size
        records = 0
        rows = 0
    return {
        f"{kind}_magic": magic,
        f"{kind}_bytes": path.stat().st_size,
        f"{kind}_record_bytes": record_bytes,
        f"{kind}_records_quick": records,
        f"{kind}_rows_quick": rows,
    }


def discover_trial_dirs(source_root: Path, source_material: str, diameter: str | None) -> list[tuple[Path, float]]:
    material_dir = source_root / source_material
    if not material_dir.exists():
        material_dir = source_root / "sats" / source_material
    if not material_dir.exists():
        raise FileNotFoundError(f"source material not found: {source_material}")

    out: list[tuple[Path, float]] = []
    for d_dir in sorted(p for p in material_dir.iterdir() if p.is_dir()):
        d_match = D_DIR_RE.match(d_dir.name)
        if d_match is None:
            continue
        if diameter is not None and d_dir.name.lower() != diameter.lower():
            continue
        diameter_mm = float(d_match.group("diameter"))
        for trial_dir in sorted(p for p in d_dir.iterdir() if p.is_dir() and SOURCE_TEST_DIR_RE.match(p.name)):
            out.append((trial_dir, diameter_mm))
    return out


def _duration(time_s: np.ndarray) -> float:
    return float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0


def _rate(rows: int, duration_s: float) -> float:
    return float(rows / duration_s) if duration_s > 0 else 0.0


def analyze_trial(trial_dir: Path, diameter_mm: float, full: bool) -> dict:
    row: dict = {
        "trial_dir": str(trial_dir),
        "diameter_mm": diameter_mm,
        "ok": False,
        "error": "",
    }
    try:
        bin_set = find_bin_set(trial_dir)
        row.update({f"{k}_path": str(v) for k, v in bin_set.items()})
        for kind, path in bin_set.items():
            row.update(_source_summary(path, kind))

        if full:
            due = load_due_bin(bin_set["due"])
            ether = load_ethermotion_bin(bin_set["ethermotion"])
            loadcell = load_loadcell_bin(bin_set["loadcell"])
            rows, baseline, summary = build_merged_rows(
                due,
                ether,
                loadcell,
                target_hz=200.0,
                max_dt_ms=10.0,
                window_ms=10.0,
                window_agg="median",
                stable_xy_only=True,
                baseline_fallback_sec=2.0,
                force_round_dp=None,
                z_start_mm=z_start_for_indenter(diameter_mm),
            )
            xy = np.stack([rows["x_mm"], rows["y_mm"]], axis=1)
            unique_xy, counts = np.unique(xy, axis=0, return_counts=True)
            fz = rows["Fz"].astype(np.float64)
            z = rows["z_depth_mm"].astype(np.float64)
            active = (fz > 0.1) & (z >= 0.001)
            row.update(
                {
                    "due_rows_full": due.expanded_rows,
                    "ethermotion_rows_full": ether.rows,
                    "loadcell_rows_full": loadcell.rows,
                    "due_duration_s": round(_duration(due.time_s), 6),
                    "ethermotion_duration_s": round(_duration(ether.time_s), 6),
                    "loadcell_duration_s": round(_duration(loadcell.time_s), 6),
                    "due_rate_hz": round(_rate(due.expanded_rows, _duration(due.time_s)), 3),
                    "ethermotion_rate_hz": round(_rate(ether.rows, _duration(ether.time_s)), 3),
                    "loadcell_rate_hz": round(_rate(loadcell.rows, _duration(loadcell.time_s)), 3),
                    "merged_rows": int(len(rows)),
                    "merged_duration_s": round(float(summary["duration_sec"]), 6),
                    "xy_cells": int(len(unique_xy)),
                    "seq_len_min": int(counts.min()) if len(counts) else 0,
                    "seq_len_median": float(np.median(counts)) if len(counts) else 0.0,
                    "seq_len_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
                    "seq_len_max": int(counts.max()) if len(counts) else 0,
                    "z_depth_min_mm": float(np.min(z)) if len(z) else 0.0,
                    "z_depth_max_mm": float(np.max(z)) if len(z) else 0.0,
                    "fz_min_n": float(np.min(fz)) if len(fz) else 0.0,
                    "fz_median_n": float(np.median(fz)) if len(fz) else 0.0,
                    "fz_p95_n": float(np.percentile(fz, 95)) if len(fz) else 0.0,
                    "fz_max_n": float(np.max(fz)) if len(fz) else 0.0,
                    "active_rows": int(active.sum()),
                    "active_ratio": float(active.mean()) if len(active) else 0.0,
                    "kg_baseline": float(baseline.get("kg_baseline", 0.0)),
                }
            )
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001 - report and continue per trial.
        row["error"] = str(exc)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SATS raw BIN data sufficiency.")
    parser.add_argument("--source-root", type=Path, default=Path("skin_ws/raw_data"))
    parser.add_argument("--source-material", default="eco20 + mesh")
    parser.add_argument("--diameter", default="d5", help="Diameter directory to inspect, e.g. d5. Use all for all.")
    parser.add_argument("--full", action="store_true", help="Build in-memory 200 Hz merged rows and summarize distributions.")
    parser.add_argument("--out", type=Path, default=Path("sats/tools/raw_bin_sufficiency.csv"))
    args = parser.parse_args()

    diameter = None if args.diameter.lower() == "all" else args.diameter
    trials = discover_trial_dirs(args.source_root, args.source_material, diameter)
    rows = [analyze_trial(trial_dir, diameter_mm, full=args.full) for trial_dir, diameter_mm in trials]
    write_csv(args.out, rows)
    ok = sum(1 for row in rows if row.get("ok"))
    print(f"wrote {args.out} ({ok}/{len(rows)} ok, full={args.full})")


if __name__ == "__main__":
    main()
