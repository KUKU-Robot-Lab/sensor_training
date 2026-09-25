#!/usr/bin/env python3
"""Canonical merged binary format for SATS preprocessing.

The binary format is:

    SATS_MERGED_BIN_V1\n
    {"row_count": ..., "dtype_descr": ..., ...}\n
    END_HEADER\n
    <numpy structured rows>

CSV is an export/compatibility surface. Training and GT generation can read this
format through numpy memmap without parsing large text files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MERGED_BIN_MAGIC = "SATS_MERGED_BIN_V1"
SENSOR_COLS = [f"s{i}" for i in range(1, 17)]
MERGED_COLUMNS = [
    "timestep_sec",
    *SENSOR_COLS,
    "x_mm",
    "y_mm",
    "z_stage_mm",
    "z_depth_mm",
    "u_mm",
    "Fz",
    "timestamp_due",
    "timestamp_loadcell",
    "timestamp_ethermotion",
    "lag_due_sec",
    "lag_loadcell_sec",
    "lag_ethermotion_sec",
    "lag_due_abs_sec",
    "lag_loadcell_abs_sec",
    "lag_ethermotion_abs_sec",
]


MERGED_DTYPE = np.dtype(
    [
        ("timestep_sec", "<f8"),
        *[(c, "<f4") for c in SENSOR_COLS],
        ("x_mm", "<f4"),
        ("y_mm", "<f4"),
        ("z_stage_mm", "<f4"),
        ("z_depth_mm", "<f4"),
        ("u_mm", "<f4"),
        ("Fz", "<f4"),
        ("timestamp_due", "<f8"),
        ("timestamp_loadcell", "<f8"),
        ("timestamp_ethermotion", "<f8"),
        ("lag_due_sec", "<f4"),
        ("lag_loadcell_sec", "<f4"),
        ("lag_ethermotion_sec", "<f4"),
        ("lag_due_abs_sec", "<f4"),
        ("lag_loadcell_abs_sec", "<f4"),
        ("lag_ethermotion_abs_sec", "<f4"),
    ]
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_merged_bin(path: Path | str, rows: np.ndarray, metadata: dict | None = None) -> None:
    """Write structured merged rows with an embedded JSON header."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=MERGED_DTYPE)
    header = {
        "magic": MERGED_BIN_MAGIC,
        "version": 1,
        "row_count": int(len(rows)),
        "dtype_descr": MERGED_DTYPE.descr,
        "columns": MERGED_COLUMNS,
        "metadata": metadata or {},
    }

    with open(out_path, "wb") as f:
        f.write((MERGED_BIN_MAGIC + "\n").encode("ascii"))
        f.write(json.dumps(header, separators=(",", ":"), default=_json_default).encode("utf-8"))
        f.write(b"\nEND_HEADER\n")
        rows.tofile(f)


def read_merged_bin_header(path: Path | str) -> tuple[dict, int]:
    """Return ``(header, data_offset_bytes)`` for a SATS merged binary."""

    in_path = Path(path)
    with open(in_path, "rb") as f:
        magic = f.readline().decode("ascii", errors="replace").strip()
        if magic != MERGED_BIN_MAGIC:
            raise ValueError(f"{in_path}: expected {MERGED_BIN_MAGIC}, got {magic!r}")
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{in_path}: missing JSON header")
        header = json.loads(header_line.decode("utf-8"))
        marker = f.readline()
        if marker != b"END_HEADER\n":
            raise ValueError(f"{in_path}: missing END_HEADER marker")
        return header, f.tell()


def open_merged_bin(path: Path | str, mmap_mode: str = "r") -> tuple[dict, np.memmap]:
    """Open merged rows as a numpy memmap."""

    header, offset = read_merged_bin_header(path)
    descr = header.get("dtype_descr", MERGED_DTYPE.descr)
    dtype = np.dtype([tuple(item) for item in descr])
    rows = np.memmap(
        Path(path),
        dtype=dtype,
        mode=mmap_mode,
        offset=offset,
        shape=(int(header["row_count"]),),
    )
    return header, rows


def merged_bin_to_frame(
    path: Path | str,
    columns: Iterable[str] | None = None,
    *,
    u_zero_only: bool = False,
    u_zero_tol_mm: float = 1e-6,
) -> pd.DataFrame:
    """Load selected columns from merged bin into a DataFrame."""

    _, rows = open_merged_bin(path)
    selected = list(columns) if columns is not None else MERGED_COLUMNS
    missing = [c for c in selected if c not in rows.dtype.names]
    if missing:
        raise ValueError(f"{path}: missing merged bin columns: {missing}")

    mask = None
    if u_zero_only and "u_mm" in rows.dtype.names:
        mask = np.abs(np.asarray(rows["u_mm"], dtype=np.float64)) <= u_zero_tol_mm

    data = {}
    for col in selected:
        arr = np.asarray(rows[col])
        data[col] = arr[mask] if mask is not None else arr
    return pd.DataFrame(data)


def export_merged_bin_csv(
    bin_path: Path | str,
    csv_path: Path | str,
    *,
    columns: Iterable[str] | None = None,
    u_zero_only: bool = False,
    u_zero_tol_mm: float = 1e-6,
    limit: int | None = None,
) -> int:
    """Export merged bin rows to CSV for inspection or legacy tools."""

    df = merged_bin_to_frame(
        bin_path,
        columns=columns,
        u_zero_only=u_zero_only,
        u_zero_tol_mm=u_zero_tol_mm,
    )
    if limit is not None and limit >= 0:
        df = df.head(limit)
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return int(len(df))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/export SATS merged binary files.")
    parser.add_argument("bin_path", type=Path)
    parser.add_argument("--csv-out", type=Path, help="Optional CSV export path.")
    parser.add_argument("--u-zero-only", action="store_true", help="Export only u_mm == 0 rows.")
    parser.add_argument("--u-zero-tol-mm", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header, _ = open_merged_bin(args.bin_path)
    print(json.dumps(header, indent=2, ensure_ascii=False))
    if args.csv_out:
        n = export_merged_bin_csv(
            args.bin_path,
            args.csv_out,
            u_zero_only=args.u_zero_only,
            u_zero_tol_mm=args.u_zero_tol_mm,
            limit=args.limit,
        )
        print(f"exported {n} rows -> {args.csv_out}")


if __name__ == "__main__":
    main()
