#!/usr/bin/env python3
"""Merge SATS raw acquisition binaries into the canonical merged BIN format.

This module is the binary-first counterpart of ``raw_merge.py``. It reads the
raw DUE/EtherMotion/loadcell BIN logs directly, aligns them on a common time
axis, and writes one ``*_merged.bin`` per trial. CSV is intentionally optional
and used only as an export/inspection surface.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sats.preprocessing.merged_bin import (
        MERGED_COLUMNS,
        MERGED_DTYPE,
        SENSOR_COLS,
        export_merged_bin_csv,
        write_merged_bin,
    )
    from sats.preprocessing.raw_merge import (
        GRAVITY,
        GRID_MIN_MM,
        GRID_SIZE,
        SCAN_START_X_MM,
        SCAN_START_Y_MM,
        XY_GRID_MM,
        XY_GRID_TOL_MM,
        discover_trial_dirs,
        parse_trial_dir_info,
        z_start_for_indenter,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from merged_bin import (  # type: ignore[no-redef]
        MERGED_COLUMNS,
        MERGED_DTYPE,
        SENSOR_COLS,
        export_merged_bin_csv,
        write_merged_bin,
    )
    from raw_merge import (  # type: ignore[no-redef]
        GRAVITY,
        GRID_MIN_MM,
        GRID_SIZE,
        SCAN_START_X_MM,
        SCAN_START_Y_MM,
        XY_GRID_MM,
        XY_GRID_TOL_MM,
        discover_trial_dirs,
        parse_trial_dir_info,
        z_start_for_indenter,
    )


NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
DUE_RECORD_STRUCT = struct.Struct("<Q")
LOADCELL_RECORD_STRUCT = struct.Struct("<QI")

DUE_MAGIC = "DUE_RAW_BURST_BIN_V1"
ETHERMOTION_MAGIC = "ETHERMOTION_ENCODER_BIN_V1"
LOADCELL_MAGIC = "LOADCELL_BIN_V1"

DUE_PATTERNS = ["due_raw_burst_*.bin"]
ETHERMOTION_PATTERNS = ["ethermotion_encoder_*.bin"]
LOADCELL_PATTERNS = ["loadcell_raw_*.bin"]
LOADCELL_VALUE_RE = re.compile(rb"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class DueBin:
    time_s: np.ndarray
    sensors: np.ndarray
    header: dict
    bursts: int
    expanded_rows: int


@dataclass(frozen=True)
class EtherMotionBin:
    time_s: np.ndarray
    x_cmd: np.ndarray
    y_cmd: np.ndarray
    z_cmd: np.ndarray
    u_cmd: np.ndarray
    header: dict
    rows: int

    @property
    def x_mm(self) -> np.ndarray:
        return self.x_cmd * XY_GRID_MM

    @property
    def y_mm(self) -> np.ndarray:
        return self.y_cmd * XY_GRID_MM

    @property
    def z_mm(self) -> np.ndarray:
        return self.z_cmd * XY_GRID_MM

    @property
    def u_mm(self) -> np.ndarray:
        return self.u_cmd * XY_GRID_MM


@dataclass(frozen=True)
class LoadcellBin:
    time_s: np.ndarray
    kg: np.ndarray
    header: dict
    rows: int
    records: int


def read_bin_header(path: Path | str) -> tuple[str, dict, int]:
    """Return ``(magic, header, data_offset)`` for a raw acquisition BIN."""

    in_path = Path(path)
    with open(in_path, "rb") as f:
        magic = f.readline().decode("ascii", errors="replace").strip()
        if not magic:
            raise ValueError(f"{in_path}: missing binary magic header")
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{in_path}: missing JSON header")
        header = json.loads(header_line.decode("ascii", errors="replace"))
        marker = f.readline()
        if marker != b"END_HEADER\n":
            raise ValueError(f"{in_path}: missing END_HEADER marker")
        return magic, header, f.tell()


def _fixed_record_count(path: Path, offset: int, record_bytes: int) -> int:
    data_bytes = path.stat().st_size - offset
    if data_bytes < 0:
        raise ValueError(f"{path}: invalid data offset {offset}")
    if data_bytes % record_bytes != 0:
        raise ValueError(
            f"{path}: payload size {data_bytes} is not divisible by record size {record_bytes}"
        )
    return data_bytes // record_bytes


def load_due_bin(path: Path | str) -> DueBin:
    """Load DUE raw burst BIN as expanded FIFO sensor frames.

    The raw payload contains 10 FIFO frames with a single burst timestamp.
    We preserve those frames as the effective sensor stream, estimating the
    intra-burst frame interval from the median burst interval.
    """

    in_path = Path(path)
    magic, header, offset = read_bin_header(in_path)
    if magic != DUE_MAGIC:
        raise ValueError(f"{in_path}: expected {DUE_MAGIC}, got {magic}")

    expected_bytes = DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE
    record_bytes = int(header.get("record_bytes", expected_bytes))
    if record_bytes != expected_bytes:
        raise ValueError(f"{in_path}: unsupported DUE record size {record_bytes}")

    n_records = _fixed_record_count(in_path, offset, record_bytes)
    dtype = np.dtype([("elapsed_ns", "<u8"), ("payload", "<u4", (NUM_SENSORS, FIFO_FRAMES))])
    records = np.memmap(in_path, dtype=dtype, mode="r", offset=offset, shape=(n_records,))
    burst_time_s = np.asarray(records["elapsed_ns"], dtype=np.float64) / 1_000_000_000.0
    burst_dt = np.diff(np.unique(burst_time_s))
    burst_dt = burst_dt[burst_dt > 0]
    frame_dt = float(np.median(burst_dt)) / FIFO_FRAMES if len(burst_dt) else 0.0
    frame_offsets = np.arange(FIFO_FRAMES, dtype=np.float64) * frame_dt
    time_s = (burst_time_s[:, None] + frame_offsets[None, :]).reshape(-1)

    payload = np.asarray(records["payload"], dtype=np.float32)
    sensors = payload.transpose(0, 2, 1).reshape(-1, NUM_SENSORS)
    return DueBin(
        time_s=time_s,
        sensors=sensors,
        header=header,
        bursts=int(n_records),
        expanded_rows=int(n_records * FIFO_FRAMES),
    )


def _ethermotion_dtype(record_bytes: int) -> np.dtype:
    if record_bytes == 40:
        return np.dtype(
            [
                ("elapsed_ns", "<u8"),
                ("x_cmd", "<f8"),
                ("y_cmd", "<f8"),
                ("z_cmd", "<f8"),
                ("u_cmd", "<f8"),
            ]
        )
    if record_bytes == 56:
        return np.dtype(
            [
                ("elapsed_ns", "<u8"),
                ("x_cmd", "<f8"),
                ("y_cmd", "<f8"),
                ("z_cmd", "<f8"),
                ("u_cmd", "<f8"),
                ("x_lcmd", "<i4"),
                ("y_lcmd", "<i4"),
                ("z_lcmd", "<i4"),
                ("u_lcmd", "<i4"),
            ]
        )
    raise ValueError(f"unsupported EtherMotion record size {record_bytes}")


def load_ethermotion_bin(path: Path | str) -> EtherMotionBin:
    in_path = Path(path)
    magic, header, offset = read_bin_header(in_path)
    if magic != ETHERMOTION_MAGIC:
        raise ValueError(f"{in_path}: expected {ETHERMOTION_MAGIC}, got {magic}")

    record_bytes = int(header.get("record_bytes", 40))
    dtype = _ethermotion_dtype(record_bytes)
    if dtype.itemsize != record_bytes:
        raise ValueError(f"{in_path}: dtype size mismatch for record size {record_bytes}")
    n_records = _fixed_record_count(in_path, offset, record_bytes)
    records = np.memmap(in_path, dtype=dtype, mode="r", offset=offset, shape=(n_records,))

    time_s = np.asarray(records["elapsed_ns"], dtype=np.float64) / 1_000_000_000.0
    x = np.asarray(records["x_cmd"], dtype=np.float64)
    y = np.asarray(records["y_cmd"], dtype=np.float64)
    z = np.asarray(records["z_cmd"], dtype=np.float64)
    u = np.asarray(records["u_cmd"], dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(u)
    if not finite.any():
        raise ValueError(f"{in_path}: no finite EtherMotion rows")

    return EtherMotionBin(
        time_s=time_s[finite],
        x_cmd=x[finite],
        y_cmd=y[finite],
        z_cmd=z[finite],
        u_cmd=u[finite],
        header=header,
        rows=int(finite.sum()),
    )


def _parse_loadcell_kg(line: bytes) -> float | None:
    match = LOADCELL_VALUE_RE.search(line)
    if match is None:
        return None
    try:
        return float(match.group(0).decode("ascii"))
    except ValueError:
        return None


def load_loadcell_bin(path: Path | str) -> LoadcellBin:
    in_path = Path(path)
    magic, header, offset = read_bin_header(in_path)
    if magic != LOADCELL_MAGIC:
        raise ValueError(f"{in_path}: expected {LOADCELL_MAGIC}, got {magic}")

    rows_t: list[float] = []
    rows_kg: list[float] = []
    buffer = b""
    records = 0
    with open(in_path, "rb") as f:
        f.seek(offset)
        while True:
            rec_header = f.read(LOADCELL_RECORD_STRUCT.size)
            if not rec_header:
                break
            if len(rec_header) != LOADCELL_RECORD_STRUCT.size:
                raise ValueError(f"{in_path}: truncated loadcell record header at {records}")
            elapsed_ns, payload_size = LOADCELL_RECORD_STRUCT.unpack(rec_header)
            payload = f.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"{in_path}: truncated loadcell payload at {records}")
            records += 1

            buffer += payload
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                kg = _parse_loadcell_kg(line.strip())
                if kg is None:
                    continue
                rows_t.append(elapsed_ns / 1_000_000_000.0)
                rows_kg.append(kg)

    if buffer:
        kg = _parse_loadcell_kg(buffer.strip())
        if kg is not None:
            # Use the last record timestamp for a trailing partial line.
            rows_t.append((elapsed_ns if records else 0) / 1_000_000_000.0)
            rows_kg.append(kg)

    if not rows_kg:
        raise ValueError(f"{in_path}: no loadcell kg rows parsed")

    return LoadcellBin(
        time_s=np.asarray(rows_t, dtype=np.float64),
        kg=np.asarray(rows_kg, dtype=np.float64),
        header=header,
        rows=len(rows_kg),
        records=records,
    )


def _finite_rows(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim == 1:
        finite = np.isfinite(t) & np.isfinite(values)
    else:
        finite = np.isfinite(t) & np.isfinite(values).all(axis=1)
    return t[finite], values[finite]


def _coalesce_by_time(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, values = _finite_rows(np.asarray(t, dtype=np.float64), np.asarray(values))
    if len(t) == 0:
        raise ValueError("cannot coalesce empty timestamp array")
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    v_sorted = values[order]
    uniq_t, inverse, counts = np.unique(t_sorted, return_inverse=True, return_counts=True)

    if v_sorted.ndim == 1:
        out = np.zeros(len(uniq_t), dtype=np.float64)
        np.add.at(out, inverse, v_sorted.astype(np.float64))
        out /= counts
    else:
        out = np.zeros((len(uniq_t), v_sorted.shape[1]), dtype=np.float64)
        np.add.at(out, inverse, v_sorted.astype(np.float64))
        out /= counts[:, None]
    return uniq_t, out


def _estimate_hz(t: np.ndarray) -> float | None:
    uniq_t = np.unique(np.asarray(t, dtype=np.float64))
    dt = np.diff(uniq_t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    med = float(np.median(dt))
    return None if med <= 0 else 1.0 / med


def _interp_values(common_t: np.ndarray, src_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    src_t, src_v = _coalesce_by_time(src_t, values)
    if src_v.ndim == 1:
        return np.interp(common_t, src_t, src_v).astype(np.float64)
    out = np.empty((len(common_t), src_v.shape[1]), dtype=np.float64)
    for col_i in range(src_v.shape[1]):
        out[:, col_i] = np.interp(common_t, src_t, src_v[:, col_i])
    return out


def _nearest_abs_dt(src_t: np.ndarray, common_t: np.ndarray) -> np.ndarray:
    src_t = np.unique(np.asarray(src_t, dtype=np.float64))
    idx = np.searchsorted(src_t, common_t)
    idx = np.clip(idx, 0, len(src_t) - 1)
    prev_idx = np.maximum(idx - 1, 0)
    d1 = np.abs(src_t[idx] - common_t)
    d0 = np.abs(src_t[prev_idx] - common_t)
    return np.where(d0 < d1, d0, d1)


def _rolling_mean(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    values = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
    if values.ndim == 1:
        return np.convolve(values, kernel, mode="same")
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.shape[1]):
        out[:, i] = np.convolve(values[:, i], kernel, mode="same")
    return out


def _rolling_aggregate(values: np.ndarray, window_samples: int, agg: str) -> np.ndarray:
    if window_samples <= 1:
        return values
    if agg == "mean":
        return _rolling_mean(values, window_samples)
    if agg != "median":
        raise ValueError(f"unknown window aggregation: {agg}")
    # Median is rarely used with the defaults (10 ms at 200 Hz => 2 samples),
    # but keep it for parity with raw_merge.py without introducing a hard
    # dependency in the default path.
    import pandas as pd

    if values.ndim == 1:
        return (
            pd.Series(values)
            .rolling(window=window_samples, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=np.float64)
        )
    return (
        pd.DataFrame(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )


def _head_zero_block_bounds(ether: EtherMotionBin) -> tuple[float | None, float | None, int]:
    zero_mask = (ether.x_cmd == 0) & (ether.y_cmd == 0) & (ether.z_cmd == 0)
    if len(ether.time_s) == 0 or not bool(zero_mask[0]):
        return None, None, 0
    false_idx = np.flatnonzero(~zero_mask)
    end = int(false_idx[0]) if len(false_idx) else len(zero_mask)
    if end == 0:
        return None, None, 0
    return float(ether.time_s[0]), float(ether.time_s[end - 1]), end


def compute_baseline(due: DueBin, loadcell: LoadcellBin, ether: EtherMotionBin, fallback_sec: float) -> dict:
    t0, t1, n_ether = _head_zero_block_bounds(ether)
    baseline_mode = "ethermotion_head_xyz_zero"
    if t0 is None or t1 is None:
        baseline_mode = "fallback_head_window"
        t0 = min(float(due.time_s[0]), float(loadcell.time_s[0]), float(ether.time_s[0]))
        t1 = t0 + float(fallback_sec)
        n_ether = int(((ether.time_s >= t0) & (ether.time_s <= t1)).sum())

    due_mask = (due.time_s >= t0) & (due.time_s <= t1)
    lc_mask = (loadcell.time_s >= t0) & (loadcell.time_s <= t1)
    due_bl = due.sensors[due_mask]
    kg_bl = loadcell.kg[lc_mask]

    baseline: dict = {
        "baseline_mode": baseline_mode,
        "baseline_time_start": float(t0),
        "baseline_time_end": float(t1),
        "baseline_duration_sec": float(t1 - t0),
        "ethermotion_baseline_rows": int(n_ether),
        "due_baseline_rows": int(due_bl.shape[0]),
        "loadcell_baseline_rows": int(kg_bl.shape[0]),
    }
    for sensor_i, col in enumerate(SENSOR_COLS, start=1):
        vals = due_bl[:, sensor_i - 1] if due_bl.size else np.asarray([], dtype=np.float64)
        legacy_key = f"Skin{sensor_i}"
        baseline[f"{legacy_key}_mean"] = float(np.mean(vals)) if len(vals) else None
        baseline[f"{legacy_key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        baseline[f"{col}_mean"] = baseline[f"{legacy_key}_mean"]
        baseline[f"{col}_std"] = baseline[f"{legacy_key}_std"]

    due_mean = due_bl.mean(axis=1) if due_bl.size else np.asarray([], dtype=np.float64)
    due_std = due_bl.std(axis=1, ddof=1) if due_bl.shape[0] and due_bl.shape[1] > 1 else np.asarray([])
    baseline["due_mean_mean"] = float(np.mean(due_mean)) if len(due_mean) else None
    baseline["due_mean_std"] = float(np.std(due_mean, ddof=1)) if len(due_mean) > 1 else None
    baseline["due_std_mean"] = float(np.mean(due_std)) if len(due_std) else None
    baseline["due_std_std"] = float(np.std(due_std, ddof=1)) if len(due_std) > 1 else None
    baseline["kg_baseline"] = float(np.mean(kg_bl)) if len(kg_bl) else None
    baseline["kg_baseline_std"] = float(np.std(kg_bl, ddof=1)) if len(kg_bl) > 1 else None
    return baseline


def _stable_xy_mask(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    xq = np.round(x_mm / XY_GRID_MM) * XY_GRID_MM
    yq = np.round(y_mm / XY_GRID_MM) * XY_GRID_MM
    on_grid = (np.abs(x_mm - xq) <= XY_GRID_TOL_MM) & (np.abs(y_mm - yq) <= XY_GRID_TOL_MM)
    dx_prev = np.abs(np.diff(xq, prepend=xq[0]))
    dy_prev = np.abs(np.diff(yq, prepend=yq[0]))
    dx_next = np.abs(np.diff(xq, append=xq[-1]))
    dy_next = np.abs(np.diff(yq, append=yq[-1]))
    stable = ((dx_prev <= XY_GRID_TOL_MM) & (dy_prev <= XY_GRID_TOL_MM)) | (
        (dx_next <= XY_GRID_TOL_MM) & (dy_next <= XY_GRID_TOL_MM)
    )
    return on_grid & stable


def _build_common_time(due: DueBin, ether: EtherMotionBin, loadcell: LoadcellBin, target_hz: float) -> np.ndarray:
    hz_candidates = [h for h in [_estimate_hz(due.time_s), _estimate_hz(ether.time_s), _estimate_hz(loadcell.time_s)] if h]
    if target_hz <= 0:
        if not hz_candidates:
            raise RuntimeError("failed to estimate source sampling rates")
        target_hz = min(hz_candidates)

    t_start = max(float(due.time_s[0]), float(ether.time_s[0]), float(loadcell.time_s[0]))
    t_end = min(float(due.time_s[-1]), float(ether.time_s[-1]), float(loadcell.time_s[-1]))
    if t_end <= t_start:
        raise RuntimeError("No overlapping time window among DUE/EtherMotion/loadcell BIN files.")
    dt = 1.0 / float(target_hz)
    n = int(np.floor((t_end - t_start) / dt)) + 1
    return t_start + np.arange(n, dtype=np.float64) * dt


def build_merged_rows(
    due: DueBin,
    ether: EtherMotionBin,
    loadcell: LoadcellBin,
    *,
    target_hz: float = 200.0,
    max_dt_ms: float = 10.0,
    window_ms: float = 10.0,
    window_agg: str = "median",
    stable_xy_only: bool = True,
    baseline_fallback_sec: float = 2.0,
    force_round_dp: int | None = None,
    z_start_mm: float | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Build merged structured rows and return ``(rows, baseline, summary)``."""

    baseline = compute_baseline(due, loadcell, ether, fallback_sec=baseline_fallback_sec)
    kg_baseline = baseline.get("kg_baseline")
    if kg_baseline is None:
        raise RuntimeError("loadcell baseline is unavailable; cannot produce force labels")

    common_t = _build_common_time(due, ether, loadcell, target_hz=target_hz)
    due_interp = _interp_values(common_t, due.time_s, due.sensors)
    x_mm = _interp_values(common_t, ether.time_s, ether.x_mm)
    y_mm = _interp_values(common_t, ether.time_s, ether.y_mm)
    z_mm = _interp_values(common_t, ether.time_s, ether.z_mm)
    u_mm = _interp_values(common_t, ether.time_s, ether.u_mm)
    kg = _interp_values(common_t, loadcell.time_s, loadcell.kg)

    window_samples = int(round((float(window_ms) / 1000.0) * float(target_hz)))
    if window_samples >= 2:
        due_interp = _rolling_aggregate(due_interp, window_samples, window_agg)
        kg = _rolling_aggregate(kg, window_samples, window_agg)

    lag_due_abs = _nearest_abs_dt(due.time_s, common_t)
    lag_ether_abs = _nearest_abs_dt(ether.time_s, common_t)
    lag_loadcell_abs = _nearest_abs_dt(loadcell.time_s, common_t)

    keep = np.ones(len(common_t), dtype=bool)
    max_dt_sec = float(max_dt_ms) / 1000.0
    if max_dt_sec > 0:
        keep &= lag_due_abs <= max_dt_sec
        keep &= lag_ether_abs <= max_dt_sec
        keep &= lag_loadcell_abs <= max_dt_sec

    x_mm = np.round(x_mm / XY_GRID_MM) * XY_GRID_MM
    y_mm = np.round(y_mm / XY_GRID_MM) * XY_GRID_MM
    u_mm = np.round(u_mm / XY_GRID_MM) * XY_GRID_MM
    if stable_xy_only:
        keep &= _stable_xy_mask(x_mm, y_mm)

    common_t = common_t[keep]
    due_interp = due_interp[keep]
    x_mm = x_mm[keep]
    y_mm = y_mm[keep]
    z_mm = z_mm[keep]
    u_mm = u_mm[keep]
    kg = kg[keep]
    lag_due_abs = lag_due_abs[keep]
    lag_ether_abs = lag_ether_abs[keep]
    lag_loadcell_abs = lag_loadcell_abs[keep]

    if len(common_t) == 0:
        raise RuntimeError("all merged rows were removed by quality/stability filters")

    z_stage_mm = np.round(z_mm / XY_GRID_MM) * XY_GRID_MM
    if z_start_mm is None:
        start_mask = np.isclose(x_mm, SCAN_START_X_MM, atol=XY_GRID_MM / 2.0) & np.isclose(
            y_mm, SCAN_START_Y_MM, atol=XY_GRID_MM / 2.0
        )
        z_start_mm = float(np.min(z_stage_mm[start_mask])) if np.any(start_mask) else float(np.min(z_stage_mm))
    z_depth_mm = np.maximum(z_stage_mm - float(z_start_mm), 0.0)
    fz = (kg - float(kg_baseline)) * GRAVITY
    if force_round_dp is not None:
        fz = np.round(fz, force_round_dp)

    rows = np.empty(len(common_t), dtype=MERGED_DTYPE)
    rows["timestep_sec"] = common_t - float(common_t[0])
    for i, col in enumerate(SENSOR_COLS):
        rows[col] = due_interp[:, i].astype(np.float32)
    rows["x_mm"] = x_mm.astype(np.float32)
    rows["y_mm"] = y_mm.astype(np.float32)
    rows["z_stage_mm"] = z_stage_mm.astype(np.float32)
    rows["z_depth_mm"] = z_depth_mm.astype(np.float32)
    rows["u_mm"] = u_mm.astype(np.float32)
    rows["Fz"] = fz.astype(np.float32)
    rows["timestamp_due"] = common_t
    rows["timestamp_loadcell"] = common_t
    rows["timestamp_ethermotion"] = common_t
    rows["lag_due_sec"] = 0.0
    rows["lag_loadcell_sec"] = 0.0
    rows["lag_ethermotion_sec"] = 0.0
    rows["lag_due_abs_sec"] = lag_due_abs.astype(np.float32)
    rows["lag_loadcell_abs_sec"] = lag_loadcell_abs.astype(np.float32)
    rows["lag_ethermotion_abs_sec"] = lag_ether_abs.astype(np.float32)

    summary = {
        "merged_rows": int(len(rows)),
        "time_start": float(common_t[0]),
        "time_end": float(common_t[-1]),
        "duration_sec": float(common_t[-1] - common_t[0]),
        "due_source_rows": int(due.expanded_rows),
        "due_source_bursts": int(due.bursts),
        "ethermotion_source_rows": int(ether.rows),
        "loadcell_source_rows": int(loadcell.rows),
        "target_hz": float(target_hz),
        "window_ms": float(window_ms),
        "window_agg": window_agg,
        "max_dt_ms": float(max_dt_ms),
        "stable_xy_only": bool(stable_xy_only),
        "kg_baseline": float(kg_baseline),
        "force_round_dp": force_round_dp,
        "z_start_mm": float(z_start_mm),
        "due_lag_p95_sec": float(np.percentile(lag_due_abs, 95)),
        "ethermotion_lag_p95_sec": float(np.percentile(lag_ether_abs, 95)),
        "loadcell_lag_p95_sec": float(np.percentile(lag_loadcell_abs, 95)),
        "due_lag_max_sec": float(np.max(lag_due_abs)),
        "ethermotion_lag_max_sec": float(np.max(lag_ether_abs)),
        "loadcell_lag_max_sec": float(np.max(lag_loadcell_abs)),
    }
    return rows, baseline, summary


def _basename_from_manifest(raw_name: str | None) -> str | None:
    if not raw_name:
        return None
    return Path(str(raw_name).replace("\\", "/")).name


def find_bin_set(trial_dir: Path, *, bin_set_index: int = 0) -> dict[str, Path]:
    """Find a matching DUE/EtherMotion/loadcell BIN set in a trial directory."""

    manifest_path = trial_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        inputs = manifest.get("inputs", {})
        manifest_files = {
            "due": _basename_from_manifest(inputs.get("due")),
            "ethermotion": _basename_from_manifest(inputs.get("ethermotion")),
            "loadcell": _basename_from_manifest(inputs.get("loadcell")),
        }
        if all(manifest_files.values()):
            paths = {k: trial_dir / name for k, name in manifest_files.items() if name is not None}
            if all(p.exists() for p in paths.values()):
                return paths

    due_bins = sorted(p for pat in DUE_PATTERNS for p in trial_dir.glob(pat))
    ether_bins = sorted(p for pat in ETHERMOTION_PATTERNS for p in trial_dir.glob(pat))
    loadcell_bins = sorted(p for pat in LOADCELL_PATTERNS for p in trial_dir.glob(pat))
    if not due_bins or not ether_bins or not loadcell_bins:
        raise FileNotFoundError(
            f"{trial_dir}: expected raw BIN files "
            "(due_raw_burst_*.bin, ethermotion_encoder_*.bin, loadcell_raw_*.bin)"
        )

    sets: list[dict[str, Path]] = []
    ether_by_suffix = {p.stem.replace("ethermotion_encoder_", ""): p for p in ether_bins}
    loadcell_by_suffix = {p.stem.replace("loadcell_raw_", ""): p for p in loadcell_bins}
    for due_path in due_bins:
        suffix = due_path.stem.replace("due_raw_burst_", "")
        if suffix in ether_by_suffix and suffix in loadcell_by_suffix:
            sets.append(
                {
                    "due": due_path,
                    "ethermotion": ether_by_suffix[suffix],
                    "loadcell": loadcell_by_suffix[suffix],
                }
            )

    if not sets and len(due_bins) == len(ether_bins) == len(loadcell_bins) == 1:
        sets.append({"due": due_bins[0], "ethermotion": ether_bins[0], "loadcell": loadcell_bins[0]})
    if not sets:
        raise FileNotFoundError(f"{trial_dir}: no matching timestamp BIN set found")
    if bin_set_index < 0 or bin_set_index >= len(sets):
        raise IndexError(f"{trial_dir}: bin_set_index={bin_set_index} outside 0..{len(sets)-1}")
    return sets[bin_set_index]


def process_trial_dir(
    trial_dir: Path,
    raw_root: Path,
    *,
    out_dir: Path | None = None,
    trial_info_override: dict | None = None,
    target_hz: float = 200.0,
    max_dt_ms: float = 10.0,
    window_ms: float = 10.0,
    window_agg: str = "median",
    stable_xy_only: bool = True,
    baseline_fallback_sec: float = 2.0,
    force_round_dp: int | None = None,
    bin_set_index: int = 0,
    export_csv: str = "none",
    csv_limit: int | None = None,
    z_start_map: dict | None = None,
) -> dict:
    info = trial_info_override if trial_info_override is not None else parse_trial_dir_info(trial_dir, raw_root)
    bin_set = find_bin_set(trial_dir, bin_set_index=bin_set_index)

    due = load_due_bin(bin_set["due"])
    ether = load_ethermotion_bin(bin_set["ethermotion"])
    loadcell = load_loadcell_bin(bin_set["loadcell"])
    # 새 센서(rig 변경)는 z_start_map으로 인덴터별 z_start를 명시 오버라이드한다.
    # 미지정 시 레거시 Z_START_BY_DIAMETER_MM(z_start_for_indenter)로 폴백.
    diameter_mm = info.get("indenter_diameter_mm")
    z_start_resolved = None
    if z_start_map and diameter_mm is not None:
        z_start_resolved = z_start_map.get(float(diameter_mm))
    if z_start_resolved is None:
        z_start_resolved = z_start_for_indenter(diameter_mm)
    rows, baseline, summary = build_merged_rows(
        due,
        ether,
        loadcell,
        target_hz=target_hz,
        max_dt_ms=max_dt_ms,
        window_ms=window_ms,
        window_agg=window_agg,
        stable_xy_only=stable_xy_only,
        baseline_fallback_sec=baseline_fallback_sec,
        force_round_dp=force_round_dp,
        z_start_mm=z_start_resolved,
    )

    target_dir = out_dir if out_dir is not None else trial_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    trial_id = info["trial_id"]
    merged_bin = target_dir / f"{trial_id}_merged.bin"
    baseline_json = target_dir / f"{trial_id}_baseline.json"
    summary_json = target_dir / f"{trial_id}_merge_summary.json"

    summary.update(
        {
            "trial_id": trial_id,
            "material": info["material"],
            "indenter_diameter_mm": info["indenter_diameter_mm"],
            "z_max_indentation_mm": info["z_max_indentation_mm"],
            "experiment_no": info["experiment_no"],
            "raw_bin_inputs": {k: str(v) for k, v in bin_set.items()},
            "merged_columns": MERGED_COLUMNS,
            "merged_bin": str(merged_bin),
            "baseline_json": str(baseline_json),
            "summary_json": str(summary_json),
        }
    )
    metadata = {
        "trial": info,
        "baseline": baseline,
        "summary": summary,
        "source_bins": {k: str(v) for k, v in bin_set.items()},
        "pipeline": "sats.preprocessing.bin_merge",
    }
    write_merged_bin(merged_bin, rows, metadata=metadata)
    if export_csv != "none":
        csv_path = target_dir / f"{trial_id}_merged.csv"
        export_merged_bin_csv(
            merged_bin,
            csv_path,
            u_zero_only=(export_csv == "u-zero"),
            limit=csv_limit,
        )
        summary["merged_csv_export"] = str(csv_path)

    with open(baseline_json, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    this = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Merge SATS raw BIN trial streams into *_merged.bin files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=this.parents[2] / "raw_data")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Optional mirror output root. Default writes next to each trial directory.",
    )
    parser.add_argument("--target-hz", type=float, default=200.0)
    parser.add_argument("--max-dt-ms", type=float, default=10.0)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--window-agg", choices=["mean", "median"], default="median")
    parser.add_argument("--no-stable-xy-filter", action="store_true")
    parser.add_argument("--baseline-fallback-sec", type=float, default=2.0)
    parser.add_argument("--force-round-dp", type=int, default=-1)
    parser.add_argument("--bin-set-index", type=int, default=0)
    parser.add_argument("--export-csv", choices=["none", "all", "u-zero"], default="none")
    parser.add_argument("--csv-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_root.exists():
        raise FileNotFoundError(f"raw-root not found: {args.raw_root}")

    trial_dirs = discover_trial_dirs(args.raw_root)
    if not trial_dirs:
        raise RuntimeError(f"No trial directories found under {args.raw_root}")

    print(f"found {len(trial_dirs)} trial directories")
    for trial_dir in trial_dirs:
        try:
            out_dir = None
            if args.out_root is not None:
                out_dir = args.out_root / trial_dir.relative_to(args.raw_root)
            summary = process_trial_dir(
                trial_dir,
                args.raw_root,
                out_dir=out_dir,
                target_hz=args.target_hz,
                max_dt_ms=args.max_dt_ms,
                window_ms=args.window_ms,
                window_agg=args.window_agg,
                stable_xy_only=not args.no_stable_xy_filter,
                baseline_fallback_sec=args.baseline_fallback_sec,
                force_round_dp=args.force_round_dp if args.force_round_dp >= 0 else None,
                bin_set_index=args.bin_set_index,
                export_csv=args.export_csv,
                csv_limit=args.csv_limit,
            )
            print(
                f"[{summary['trial_id']}] merged {summary['merged_rows']} rows -> "
                f"{summary['merged_bin']}"
            )
        except Exception as exc:
            print(f"[{trial_dir.name}] failed: {exc}")


if __name__ == "__main__":
    main()
