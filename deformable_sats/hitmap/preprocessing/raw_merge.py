#!/usr/bin/env python3
"""
Raw sensor stream merger for tactile super-resolution experiments.

기능:
1) raw_data 하위 trial 폴더명(소재_인덴터지름_시험번호) 파싱
2) due / ethermotion / afd CSV를 timestamp 기준으로 정렬 및 동기화
3) trial별 통합 CSV 생성
4) 동기화 검증용 PNG 생성
5) 무부하(baseline) 구간 통계(due/afd) JSON 저장

예시:
  python3 preprocessing/raw_merge.py \
    --raw-root <repo>/preprocessing/raw_data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRIAL_DIR_RE = re.compile(
    # 지원 형태:
    #   material_d5_1  -> 소재, 지름, 시험번호 모두 포함
    #   material_1     -> 소재, 시험번호만 포함(지름 없음)
    r"^(?P<material>[^_]+)(?:_d(?P<diameter_mm>\d+(?:\.\d+)?))?_(?P<trial_no>\d+)$",
    re.IGNORECASE,
)
INDENTER_DIR_RE = re.compile(r"^d(?P<diameter_mm>\d+(?:\.\d+)?)$", re.IGNORECASE)
Z_DIR_RE = re.compile(r"^z_?(?P<z_max_indentation_mm>\d+(?:\.\d+)?)mm$", re.IGNORECASE)
TEST_DIR_RE = re.compile(r"^test(?P<trial_no>\d+)$", re.IGNORECASE)

# CSV 파일명 변형 대응: "*_data_*.csv" 외에
# due_<소재>_<번호>.csv, afd50_<소재>_<번호>.csv, ethermotion/eithermotion_*.csv 등도 허용한다.
DUE_PATTERNS = [
    "due_data_*.csv",
    "due_*_*.csv",
    "due*.csv",
]
ETHERMOTION_PATTERNS = [
    "ethermotion_data_*.csv",
    "ethermotion_*_*.csv",
    "eithermotion_*_*.csv",  # 현장 오타 대응
    "*thermotion*.csv",
]
AFD_PATTERNS = [
    "afd50_data_*.csv",
    "afd50_*_*.csv",
    "afd_*_*.csv",
    "afd*.csv",
]

SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]
XYZ_SCALE = 1e-4  # 0.1 um -> mm
XY_GRID_MM = 1e-4
XY_GRID_TOL_MM = 1e-9
Z_OFFSET_RAW = 105000



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw trial streams by timestamp.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "preprocessing/raw_data",
        help="Root directory that contains trial folders.",
    )
    parser.add_argument(
        "--sync-ref",
        choices=["due", "ethermotion", "afd"],
        default="ethermotion",
        help="Reference stream for merged timeline.",
    )
    parser.add_argument(
        "--align-mode",
        choices=["nearest", "resample"],
        default="resample",
        help="nearest: timestamp nearest merge, resample: common timestep interpolation.",
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=100.0,
        help="Target Hz for --align-mode resample. Use 100 for afd-limited common rate.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=10.0,
        help="Time-window size(ms) for centered rolling aggregation after resample alignment. 0 disables.",
    )
    parser.add_argument(
        "--window-agg",
        choices=["mean", "median"],
        default="median",
        help="Aggregation used for rolling time-window smoothing.",
    )
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=10.0,
        help="Drop samples whose nearest-source |dt| exceeds this threshold(ms). <=0 disables.",
    )
    parser.add_argument("--lag-due-ms", type=float, default=0.0, help="Global lag correction for due timestamp (ms).")
    parser.add_argument(
        "--lag-ethermotion-ms", type=float, default=0.0, help="Global lag correction for ethermotion timestamp (ms)."
    )
    parser.add_argument("--lag-afd-ms", type=float, default=0.0, help="Global lag correction for afd timestamp (ms).")
    parser.add_argument(
        "--due-tol-sec",
        type=float,
        default=0.03,
        help="Timestamp tolerance(sec) when aligning due stream.",
    )
    parser.add_argument(
        "--ethermotion-tol-sec",
        type=float,
        default=0.01,
        help="Timestamp tolerance(sec) when aligning ethermotion stream.",
    )
    parser.add_argument(
        "--afd-tol-sec",
        type=float,
        default=0.03,
        help="Timestamp tolerance(sec) when aligning afd stream.",
    )
    parser.add_argument(
        "--baseline-fallback-sec",
        type=float,
        default=2.0,
        help="Fallback baseline window length(sec) if XYZ=0 head block is missing.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=6000,
        help="Max points per trace in sync PNG.",
    )
    parser.add_argument(
        "--plot-y-mm",
        type=float,
        default=9.75,
        help="Fixed Y(mm) for sync plot.",
    )
    parser.add_argument(
        "--plot-y-tol-mm",
        type=float,
        default=0.05,
        help="Tolerance for fixed Y(mm) filtering.",
    )
    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.9,
        help="due/ethermotion/afd match_ratio가 이 값보다 낮으면 에러로 중단합니다.",
    )
    parser.add_argument(
        "--force-round-dp",
        type=int,
        default=2,
        help="Fx/Fy/Fz 소수점 자릿수(예: 2 -> 0.01). 음수면 반올림 비활성.",
    )
    return parser.parse_args()



def parse_trial_dir_name(name: str) -> dict:
    """폴더명에서 소재/지름/시험번호를 추출한다.

    허용 예시
      - ecemesh_d5_1   -> material=ecemesh, diameter=5, trial=1
      - ecemesh_1      -> material=ecemesh, diameter=None, trial=1
      - foo_bar        -> 정규식 미일치: material=foo_bar, trial=None
    """

    m = TRIAL_DIR_RE.match(name)
    if not m:
        return {
            "trial_id": name,
            "material": name,
            "indenter_diameter_mm": None,
            "z_max_indentation_mm": None,
            "experiment_no": None,
        }

    diameter = m.group("diameter_mm")
    return {
        "trial_id": name,
        "material": m.group("material").lower(),
        "indenter_diameter_mm": float(diameter) if diameter is not None else None,
        "z_max_indentation_mm": None,
        "experiment_no": int(m.group("trial_no")),
    }


def _format_number_for_id(value: float) -> str:
    return f"{value:g}"


def parse_trial_dir_info(trial_dir: Path, raw_root: Path) -> dict:
    """Extract metadata from a trial directory.

    Supports both legacy flat trial directories and the nested layout:
    raw_root/material/d{diameter}/z_{max indentation}mm/test{n}.
    """

    try:
        rel_parts = trial_dir.relative_to(raw_root).parts
    except ValueError:
        rel_parts = trial_dir.parts

    if len(rel_parts) >= 4:
        material, indenter_dir, z_dir, test_dir = rel_parts[-4:]
        indenter_match = INDENTER_DIR_RE.match(indenter_dir)
        z_match = Z_DIR_RE.match(z_dir)
        test_match = TEST_DIR_RE.match(test_dir)
        if indenter_match and z_match and test_match:
            diameter_mm = float(indenter_match.group("diameter_mm"))
            z_max_indentation_mm = float(z_match.group("z_max_indentation_mm"))
            trial_no = int(test_match.group("trial_no"))
            material_id = material.lower()
            trial_id = (
                f"{material_id}_d{_format_number_for_id(diameter_mm)}_"
                f"z{z_match.group('z_max_indentation_mm')}_test{trial_no}"
            )
            return {
                "trial_id": trial_id,
                "material": material_id,
                "indenter_diameter_mm": diameter_mm,
                "z_max_indentation_mm": z_max_indentation_mm,
                "experiment_no": trial_no,
            }

    info = parse_trial_dir_name(trial_dir.name)
    info.setdefault("z_max_indentation_mm", None)
    return info


def discover_trial_dirs(raw_root: Path) -> list[Path]:
    """Return trial leaf directories for the supported raw_data layouts."""

    nested = sorted(
        p
        for p in raw_root.rglob("test*")
        if p.is_dir() and TEST_DIR_RE.match(p.name) and len(p.relative_to(raw_root).parts) >= 4
    )
    if nested:
        return nested

    return sorted([p for p in raw_root.iterdir() if p.is_dir()])



def find_single_file(trial_dir: Path, patterns: list[str]) -> Path:
    """Return exactly one match across a set of glob patterns.

    패턴 목록을 순서대로 적용해 처음으로 단일 매치를 만드는 파일을 반환한다.
    어떤 패턴도 단일 매치를 만들지 못하면 전체 후보를 포함해 오류를 던진다.
    """

    errors: list[str] = []
    for pat in patterns:
        files = sorted(trial_dir.rglob(pat))
        if len(files) == 1:
            return files[0]
        errors.append(f"{pat}:{len(files)}")

    raise FileNotFoundError(
        f"[{trial_dir.name}] expected exactly one CSV but found {', '.join(errors)}"
    )



def load_due_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["Timestamp", *SKIN_COLS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"[{path.name}] missing columns: {missing}")

    out = df[["Timestamp", *SKIN_COLS]].copy()
    # 비수치 오염값은 NaN으로 강제 후 행 단위로 제거해 병합을 계속 진행한다.
    for c in ["Timestamp", *SKIN_COLS]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad_mask = out[["Timestamp", *SKIN_COLS]].isna().any(axis=1)
    bad_n = int(bad_mask.sum())
    if bad_n:
        print(f"[{path.name}] warning: dropped {bad_n} invalid due rows")
        out = out.loc[~bad_mask].copy()
    if out.empty:
        raise ValueError(f"[{path.name}] all due rows invalid after numeric coercion")

    out = out.rename(columns={"Timestamp": "timestamp_due"})
    out = out.sort_values("timestamp_due").reset_index(drop=True)
    out["due_mean"] = out[SKIN_COLS].mean(axis=1)
    out["due_std"] = out[SKIN_COLS].std(axis=1)
    return out



def load_ethermotion_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["Timestamp", "X", "Y", "Z"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"[{path.name}] missing columns: {missing}")

    out = df[["Timestamp", "X", "Y", "Z"]].copy()
    for c in ["Timestamp", "X", "Y", "Z"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad_mask = out[["Timestamp", "X", "Y", "Z"]].isna().any(axis=1)
    bad_n = int(bad_mask.sum())
    if bad_n:
        print(f"[{path.name}] warning: dropped {bad_n} invalid ethermotion rows")
        out = out.loc[~bad_mask].copy()
    if out.empty:
        raise ValueError(f"[{path.name}] all ethermotion rows invalid after numeric coercion")

    out = out.rename(columns={"Timestamp": "timestamp_ethermotion"})
    out = out.sort_values("timestamp_ethermotion").reset_index(drop=True)

    out["x_mm"] = out["X"] * XYZ_SCALE
    out["y_mm"] = out["Y"] * XYZ_SCALE
    out["z_mm"] = out["Z"] * XYZ_SCALE
    return out



def load_afd_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["Timestamp", "Fx", "Fy", "Fz"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"[{path.name}] missing columns: {missing}")

    out = df[["Timestamp", "Fx", "Fy", "Fz"]].copy()
    for c in ["Timestamp", "Fx", "Fy", "Fz"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad_mask = out[["Timestamp", "Fx", "Fy", "Fz"]].isna().any(axis=1)
    bad_n = int(bad_mask.sum())
    if bad_n:
        print(f"[{path.name}] warning: dropped {bad_n} invalid afd rows")
        out = out.loc[~bad_mask].copy()
    if out.empty:
        raise ValueError(f"[{path.name}] all afd rows invalid after numeric coercion")

    out = out.rename(columns={"Timestamp": "timestamp_afd"})
    out = out.sort_values("timestamp_afd").reset_index(drop=True)
    return out


def apply_global_lag_correction(
    due_df: pd.DataFrame,
    ether_df: pd.DataFrame,
    afd_df: pd.DataFrame,
    lag_due_sec: float,
    lag_ether_sec: float,
    lag_afd_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    due = due_df.copy()
    ether = ether_df.copy()
    afd = afd_df.copy()
    if lag_due_sec != 0.0:
        due["timestamp_due"] = due["timestamp_due"] + lag_due_sec
    if lag_ether_sec != 0.0:
        ether["timestamp_ethermotion"] = ether["timestamp_ethermotion"] + lag_ether_sec
    if lag_afd_sec != 0.0:
        afd["timestamp_afd"] = afd["timestamp_afd"] + lag_afd_sec
    return due, ether, afd



def _head_zero_block_bounds(ether_df: pd.DataFrame) -> tuple[float | None, float | None, int]:
    zero_mask = (ether_df["X"] == 0) & (ether_df["Y"] == 0) & (ether_df["Z"] == 0)
    if len(ether_df) == 0 or not bool(zero_mask.iloc[0]):
        return None, None, 0

    # 앞에서부터 연속 0 블록 길이
    first_false = int((~zero_mask).to_numpy().argmax())
    if first_false == 0 and bool(zero_mask.iloc[0]):
        # 전 구간 0 인 특수 케이스
        first_false = len(ether_df)

    block = ether_df.iloc[:first_false]
    if block.empty:
        return None, None, 0

    return (
        float(block["timestamp_ethermotion"].iloc[0]),
        float(block["timestamp_ethermotion"].iloc[-1]),
        int(len(block)),
    )



def compute_baseline(
    due_df: pd.DataFrame,
    afd_df: pd.DataFrame,
    ether_df: pd.DataFrame,
    fallback_sec: float,
) -> dict:
    t0, t1, n_ether = _head_zero_block_bounds(ether_df)

    baseline_mode = "ethermotion_head_xyz_zero"
    if t0 is None or t1 is None:
        baseline_mode = "fallback_head_window"
        earliest = min(
            float(due_df["timestamp_due"].iloc[0]),
            float(afd_df["timestamp_afd"].iloc[0]),
            float(ether_df["timestamp_ethermotion"].iloc[0]),
        )
        t0 = earliest
        t1 = earliest + fallback_sec
        n_ether = int(
            ((ether_df["timestamp_ethermotion"] >= t0) & (ether_df["timestamp_ethermotion"] <= t1)).sum()
        )

    due_bl = due_df[(due_df["timestamp_due"] >= t0) & (due_df["timestamp_due"] <= t1)].copy()
    afd_bl = afd_df[(afd_df["timestamp_afd"] >= t0) & (afd_df["timestamp_afd"] <= t1)].copy()

    baseline: dict = {
        "baseline_mode": baseline_mode,
        "baseline_time_start": float(t0),
        "baseline_time_end": float(t1),
        "baseline_duration_sec": float(t1 - t0),
        "ethermotion_baseline_rows": int(n_ether),
        "due_baseline_rows": int(len(due_bl)),
        "afd_baseline_rows": int(len(afd_bl)),
    }

    for c in SKIN_COLS + ["due_mean", "due_std"]:
        baseline[f"{c}_mean"] = float(due_bl[c].mean()) if len(due_bl) else None
        baseline[f"{c}_std"] = float(due_bl[c].std()) if len(due_bl) else None

    for c in ["Fx", "Fy", "Fz"]:
        baseline[f"{c}_mean"] = float(afd_bl[c].mean()) if len(afd_bl) else None
        baseline[f"{c}_std"] = float(afd_bl[c].std()) if len(afd_bl) else None

    return baseline



def align_streams(
    due_df: pd.DataFrame,
    ether_df: pd.DataFrame,
    afd_df: pd.DataFrame,
    align_mode: str,
    sync_ref: str,
    resample_hz: float,
    due_tol_sec: float,
    ether_tol_sec: float,
    afd_tol_sec: float,
) -> pd.DataFrame:
    if align_mode == "resample":
        return align_streams_resample(
            due_df=due_df,
            ether_df=ether_df,
            afd_df=afd_df,
            target_hz=resample_hz,
        )

    frames = {
        "due": due_df,
        "ethermotion": ether_df,
        "afd": afd_df,
    }
    key_cols = {
        "due": "timestamp_due",
        "ethermotion": "timestamp_ethermotion",
        "afd": "timestamp_afd",
    }
    tols = {
        "due": due_tol_sec,
        "ethermotion": ether_tol_sec,
        "afd": afd_tol_sec,
    }

    ref = frames[sync_ref].copy()
    ref_key = key_cols[sync_ref]

    merged = ref
    for name in ["due", "ethermotion", "afd"]:
        if name == sync_ref:
            continue
        merged = pd.merge_asof(
            merged.sort_values(ref_key),
            frames[name].sort_values(key_cols[name]),
            left_on=ref_key,
            right_on=key_cols[name],
            direction="nearest",
            tolerance=tols[name],
        )

    merged = merged.sort_values(ref_key).reset_index(drop=True)
    merged["timestamp"] = merged[ref_key]
    merged["time_rel_sec"] = merged["timestamp"] - float(merged["timestamp"].iloc[0])

    if "timestamp_due" in merged.columns:
        merged["lag_due_sec"] = merged["timestamp_due"] - merged["timestamp"]
    if "timestamp_ethermotion" in merged.columns:
        merged["lag_ethermotion_sec"] = merged["timestamp_ethermotion"] - merged["timestamp"]
    if "timestamp_afd" in merged.columns:
        merged["lag_afd_sec"] = merged["timestamp_afd"] - merged["timestamp"]
    if "lag_due_sec" in merged.columns:
        merged["lag_due_abs_sec"] = np.abs(merged["lag_due_sec"])
    if "lag_ethermotion_sec" in merged.columns:
        merged["lag_ethermotion_abs_sec"] = np.abs(merged["lag_ethermotion_sec"])
    if "lag_afd_sec" in merged.columns:
        merged["lag_afd_abs_sec"] = np.abs(merged["lag_afd_sec"])

    return merged


def _estimate_hz_from_timestamp(df: pd.DataFrame, tcol: str) -> float | None:
    dt = df[tcol].diff().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    med = float(dt.median())
    if med <= 0:
        return None
    return 1.0 / med


def _prepare_interp_frame(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    # 중복 timestamp는 평균으로 축약해서 보간 가능하게 정리
    out = df.sort_values(tcol).groupby(tcol, as_index=False).mean(numeric_only=True)
    out = out.sort_values(tcol).reset_index(drop=True)
    return out


def _interp_to_common_t(df: pd.DataFrame, tcol: str, common_t: np.ndarray) -> pd.DataFrame:
    src = _prepare_interp_frame(df, tcol=tcol)
    src_t = src[tcol].to_numpy(dtype=np.float64)

    out = pd.DataFrame({tcol: common_t})
    for c in src.columns:
        if c == tcol:
            continue
        v = src[c].to_numpy(dtype=np.float64)
        out[c] = np.interp(common_t, src_t, v)
    return out


def _nearest_abs_dt(src_t: np.ndarray, common_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src_t, common_t)
    idx = np.clip(idx, 0, len(src_t) - 1)
    prev_idx = np.maximum(idx - 1, 0)
    d1 = np.abs(src_t[idx] - common_t)
    d0 = np.abs(src_t[prev_idx] - common_t)
    use_prev = d0 < d1
    out = np.where(use_prev, d0, d1)
    return out


def _rolling_aggregate(df: pd.DataFrame, cols: list[str], window_samples: int, agg: str) -> pd.DataFrame:
    if window_samples <= 1 or not cols:
        return df
    out = df.copy()
    roll = out[cols].rolling(window=window_samples, center=True, min_periods=1)
    if agg == "mean":
        out[cols] = roll.mean()
    else:
        out[cols] = roll.median()
    return out


def align_streams_resample(
    due_df: pd.DataFrame,
    ether_df: pd.DataFrame,
    afd_df: pd.DataFrame,
    target_hz: float,
) -> pd.DataFrame:
    hz_due = _estimate_hz_from_timestamp(due_df, "timestamp_due")
    hz_ether = _estimate_hz_from_timestamp(ether_df, "timestamp_ethermotion")
    hz_afd = _estimate_hz_from_timestamp(afd_df, "timestamp_afd")

    hz_candidates = [h for h in [hz_due, hz_ether, hz_afd] if h is not None and h > 0]
    if not hz_candidates:
        raise RuntimeError("Failed to estimate source sampling rates.")

    if target_hz <= 0:
        target_hz = min(hz_candidates)
    dt = 1.0 / float(target_hz)

    t_start = max(
        float(due_df["timestamp_due"].iloc[0]),
        float(ether_df["timestamp_ethermotion"].iloc[0]),
        float(afd_df["timestamp_afd"].iloc[0]),
    )
    t_end = min(
        float(due_df["timestamp_due"].iloc[-1]),
        float(ether_df["timestamp_ethermotion"].iloc[-1]),
        float(afd_df["timestamp_afd"].iloc[-1]),
    )
    if t_end <= t_start:
        raise RuntimeError("No overlapping time window among due/ethermotion/afd.")

    n = int(np.floor((t_end - t_start) / dt)) + 1
    common_t = t_start + np.arange(n, dtype=np.float64) * dt

    due_i = _interp_to_common_t(due_df, "timestamp_due", common_t)
    ether_i = _interp_to_common_t(ether_df, "timestamp_ethermotion", common_t)
    afd_i = _interp_to_common_t(afd_df, "timestamp_afd", common_t)

    merged = pd.DataFrame({"timestamp": common_t})
    merged["timestamp_due"] = common_t
    merged["timestamp_ethermotion"] = common_t
    merged["timestamp_afd"] = common_t

    for c in due_i.columns:
        if c != "timestamp_due":
            merged[c] = due_i[c].to_numpy()
    for c in ether_i.columns:
        if c != "timestamp_ethermotion":
            merged[c] = ether_i[c].to_numpy()
    for c in afd_i.columns:
        if c != "timestamp_afd":
            merged[c] = afd_i[c].to_numpy()

    merged["time_rel_sec"] = merged["timestamp"] - float(merged["timestamp"].iloc[0])
    due_t = due_df["timestamp_due"].to_numpy(dtype=np.float64)
    ether_t = ether_df["timestamp_ethermotion"].to_numpy(dtype=np.float64)
    afd_t = afd_df["timestamp_afd"].to_numpy(dtype=np.float64)
    merged["lag_due_sec"] = 0.0
    merged["lag_ethermotion_sec"] = 0.0
    merged["lag_afd_sec"] = 0.0
    merged["lag_due_abs_sec"] = _nearest_abs_dt(due_t, common_t)
    merged["lag_ethermotion_abs_sec"] = _nearest_abs_dt(ether_t, common_t)
    merged["lag_afd_abs_sec"] = _nearest_abs_dt(afd_t, common_t)
    merged["resample_hz"] = float(target_hz)
    return merged


def apply_time_window_aggregation(
    merged: pd.DataFrame, window_ms: float, agg: str, resample_hz: float | None
) -> pd.DataFrame:
    if window_ms <= 0 or resample_hz is None or resample_hz <= 0:
        return merged
    window_samples = int(round((window_ms / 1000.0) * float(resample_hz)))
    if window_samples < 2:
        return merged
    due_cols = [c for c in SKIN_COLS + ["due_mean", "due_std"] if c in merged.columns]
    afd_cols = [c for c in ["Fx", "Fy", "Fz"] if c in merged.columns]
    out = _rolling_aggregate(merged, due_cols + afd_cols, window_samples=window_samples, agg=agg)
    out["window_ms"] = float(window_ms)
    out["window_agg"] = agg
    return out


def apply_quality_gate(merged: pd.DataFrame, max_dt_sec: float) -> pd.DataFrame:
    if max_dt_sec <= 0:
        return merged
    out = merged.copy()
    masks = []
    for c in ["lag_due_abs_sec", "lag_ethermotion_abs_sec", "lag_afd_abs_sec"]:
        if c in out.columns:
            masks.append(out[c] <= max_dt_sec)
    if masks:
        keep = masks[0].copy()
        for m in masks[1:]:
            keep &= m
        out = out.loc[keep].copy()
    return out.reset_index(drop=True)



def add_baseline_corrected_columns(merged: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    out = merged.copy()

    for c in SKIN_COLS + ["due_mean"]:
        m = baseline.get(f"{c}_mean")
        if m is None:
            out[f"{c}_bc"] = np.nan
        else:
            out[f"{c}_bc"] = out[c] - m

    for c in ["Fx", "Fy", "Fz"]:
        m = baseline.get(f"{c}_mean")
        if m is None:
            out[f"{c}_bc"] = np.nan
        else:
            out[f"{c}_bc"] = out[c] - m

    return out



def filter_xy_grid_stable_points(
    merged: pd.DataFrame, grid_mm: float = XY_GRID_MM, tol_mm: float = XY_GRID_TOL_MM
) -> pd.DataFrame:
    if merged.empty:
        return merged.copy()

    out = merged.copy()
    x = out["x_mm"].to_numpy(dtype=np.float64)
    y = out["y_mm"].to_numpy(dtype=np.float64)

    xq = np.round(x / grid_mm) * grid_mm
    yq = np.round(y / grid_mm) * grid_mm
    on_grid = (np.abs(x - xq) <= tol_mm) & (np.abs(y - yq) <= tol_mm)

    # 이동 구간 제거: 이전/다음 샘플 중 하나와 (x,y) 격자점이 같아야 정지 구간으로 간주.
    dx_prev = np.abs(np.diff(xq, prepend=xq[0]))
    dy_prev = np.abs(np.diff(yq, prepend=yq[0]))
    dx_next = np.abs(np.diff(xq, append=xq[-1]))
    dy_next = np.abs(np.diff(yq, append=yq[-1]))

    same_prev = (dx_prev <= tol_mm) & (dy_prev <= tol_mm)
    same_next = (dx_next <= tol_mm) & (dy_next <= tol_mm)
    stable = same_prev | same_next

    keep = on_grid & stable
    return out.loc[keep].reset_index(drop=True)


def trim_to_common_recording_start(
    merged: pd.DataFrame, due_df: pd.DataFrame, ether_df: pd.DataFrame, afd_df: pd.DataFrame
) -> pd.DataFrame:
    if merged.empty:
        return merged.copy()

    t_common_start = max(
        float(due_df["timestamp_due"].iloc[0]),
        float(ether_df["timestamp_ethermotion"].iloc[0]),
        float(afd_df["timestamp_afd"].iloc[0]),
    )
    out = merged[merged["timestamp"] >= t_common_start].copy()

    # 세 스트림이 실제로 모두 존재하는 구간만 사용.
    req_cols = ["timestamp_due", "timestamp_afd", "timestamp_ethermotion"]
    for c in req_cols:
        if c in out.columns:
            out = out[out[c].notna()]

    return out.reset_index(drop=True)


def build_export_frame(merged: pd.DataFrame, force_round_dp: int | None = None) -> pd.DataFrame:
    out = merged.copy()
    out["timestep_sec"] = out["time_rel_sec"]

    # DUE 채널은 Skin1~Skin16을 s1~s16으로 1:1 매핑한다.
    desired_skin_cols = [f"s{i}" for i in range(1, 17)]
    skin_src_for_desired = [f"Skin{i}" for i in range(1, 17)]

    for c in skin_src_for_desired:
        if c not in out.columns:
            out[c] = np.nan
    for c in [
        "timestamp_due",
        "timestamp_afd",
        "timestamp_ethermotion",
        "lag_due_sec",
        "lag_afd_sec",
        "lag_ethermotion_sec",
        "lag_due_abs_sec",
        "lag_afd_abs_sec",
        "lag_ethermotion_abs_sec",
        "x_mm",
        "y_mm",
        "z_mm",
        "Fx",
        "Fy",
        "Fz",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    # 출력 좌표는 0.0001 mm 격자로 정리 (특히 z축).
    out["x_mm"] = np.round(out["x_mm"].to_numpy(dtype=np.float64) / XY_GRID_MM) * XY_GRID_MM
    out["y_mm"] = np.round(out["y_mm"].to_numpy(dtype=np.float64) / XY_GRID_MM) * XY_GRID_MM
    z_raw = np.round(out["z_mm"].to_numpy(dtype=np.float64) / XY_GRID_MM)
    z_adj_raw = np.where(z_raw <= 0, 0.0, z_raw - Z_OFFSET_RAW)
    z_adj_raw = np.maximum(z_adj_raw, 0.0)

    # 스캔 시작 기준점(9.75, -9.75)을 추가 영점으로 사용.
    start_mask = np.isclose(out["x_mm"].to_numpy(dtype=np.float64), 9.75, atol=XY_GRID_MM / 2) & np.isclose(
        out["y_mm"].to_numpy(dtype=np.float64), -9.75, atol=XY_GRID_MM / 2
    )
    if np.any(start_mask):
        start_offset = float(z_adj_raw[np.where(start_mask)[0][0]])
        z_adj_raw = np.maximum(z_adj_raw - start_offset, 0.0)

    out["z_mm"] = z_adj_raw * XY_GRID_MM

    # 힘 채널 소수점 절삭(필요 시)
    if force_round_dp is not None:
        out["Fx"] = np.round(out["Fx"].to_numpy(dtype=np.float64), force_round_dp)
        out["Fy"] = np.round(out["Fy"].to_numpy(dtype=np.float64), force_round_dp)
        out["Fz"] = np.round(out["Fz"].to_numpy(dtype=np.float64), force_round_dp)

    ordered = (
        ["timestep_sec"]
        + skin_src_for_desired
        + [
            "x_mm",
            "y_mm",
            "z_mm",
            "Fx",
            "Fy",
            "Fz",
            "timestamp_due",
            "timestamp_afd",
            "timestamp_ethermotion",
            "lag_due_sec",
            "lag_afd_sec",
            "lag_ethermotion_sec",
            "lag_due_abs_sec",
            "lag_afd_abs_sec",
            "lag_ethermotion_abs_sec",
        ]
    )
    out = out[ordered].copy()
    out.columns = (
        ["timestep_sec"]
        + desired_skin_cols
        + [
            "x_mm",
            "y_mm",
            "z_mm",
            "Fx",
            "Fy",
            "Fz",
            "timestamp_due",
            "timestamp_afd",
            "timestamp_ethermotion",
            "lag_due_sec",
            "lag_afd_sec",
            "lag_ethermotion_sec",
            "lag_due_abs_sec",
            "lag_afd_abs_sec",
            "lag_ethermotion_abs_sec",
        ]
    )
    return out



def _downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]



def _filter_increasing_x_on_fixed_y(
    merged: pd.DataFrame, y_mm: float, y_tol_mm: float
) -> pd.DataFrame:
    sel = merged[np.abs(merged["y_mm"] - y_mm) <= y_tol_mm].copy()
    if sel.empty:
        return sel

    sel = sel.sort_values("time_rel_sec").reset_index(drop=True)
    dx = sel["x_mm"].diff().fillna(0.0)
    keep = dx >= 0.0
    keep.iloc[0] = True
    return sel[keep].reset_index(drop=True)


def save_sync_plot(
    merged: pd.DataFrame,
    out_png: Path,
    max_points: int,
    y_mm: float,
    y_tol_mm: float,
) -> None:
    import matplotlib.pyplot as plt

    sel = _filter_increasing_x_on_fixed_y(merged, y_mm=y_mm, y_tol_mm=y_tol_mm)
    if sel.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.text(
            0.5,
            0.5,
            f"No samples for y={y_mm:.3f}±{y_tol_mm:.3f} mm",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    due_cols = ["Skin1_bc", "Skin2_bc", "Skin3_bc", "Skin4_bc"]
    for c in due_cols + ["Fz_bc"]:
        if c not in sel.columns:
            sel[c] = np.nan

    # x 목표 그리드(-9.75~9.75, 0.5mm)로 스냅 후,
    # 연속 동일 x 구간(=한 x에서의 압입/복귀 구간)마다 대표 1점(최대 z) 추출.
    x_grid = -9.75 + 0.5 * np.arange(40, dtype=np.float64)
    x_val = sel["x_mm"].to_numpy(dtype=np.float64)
    nearest_idx = np.abs(x_val[:, None] - x_grid[None, :]).argmin(axis=1)
    sel["x_nominal_mm"] = x_grid[nearest_idx]

    seg_id = (sel["x_nominal_mm"] != sel["x_nominal_mm"].shift(1)).cumsum()
    rep_rows = []
    for _, g in sel.groupby(seg_id, sort=False):
        if g.empty:
            continue
        ridx = g["z_mm"].idxmax()
        rep_rows.append(ridx)
    rep = sel.loc[rep_rows].copy().sort_values("time_rel_sec").reset_index(drop=True)

    if len(rep) > max_points:
        idx = np.linspace(0, len(rep) - 1, max_points).astype(int)
        rep = rep.iloc[idx].reset_index(drop=True)

    step = np.arange(len(rep), dtype=np.int64)

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for col, color in zip(due_cols, colors):
        ax1.plot(step, rep[col].to_numpy(dtype=np.float64), lw=1.0, color=color, label=col)
    ax1.set_xlim(0, max(len(step) - 1, 1))
    ax1.set_xlabel("timestep segment (y fixed, x increasing)")
    ax1.set_ylabel("DUE baseline-corrected")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(step, rep["Fz_bc"].to_numpy(dtype=np.float64), lw=1.2, color="tab:red", label="Fz_bc")
    ax2.set_ylabel("Fz baseline-corrected")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"y={y_mm:.2f} mm, x-step representative (max-z): Skin1~4 and Fz")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)



def summarize_merge(merged: pd.DataFrame) -> dict:
    if merged.empty:
        return {
            "merged_rows": 0,
            "time_start": None,
            "time_end": None,
            "duration_sec": 0.0,
            "due_match_ratio": 0.0,
            "ethermotion_match_ratio": 0.0,
            "afd_match_ratio": 0.0,
        }

    summary = {
        "merged_rows": int(len(merged)),
        "time_start": float(merged["timestamp"].iloc[0]),
        "time_end": float(merged["timestamp"].iloc[-1]),
        "duration_sec": float(merged["timestamp"].iloc[-1] - merged["timestamp"].iloc[0]),
    }

    if "timestamp_due" in merged.columns:
        summary["due_match_ratio"] = float(merged["timestamp_due"].notna().mean())
    if "timestamp_ethermotion" in merged.columns:
        summary["ethermotion_match_ratio"] = float(merged["timestamp_ethermotion"].notna().mean())
    if "timestamp_afd" in merged.columns:
        summary["afd_match_ratio"] = float(merged["timestamp_afd"].notna().mean())
    for c in ["lag_due_abs_sec", "lag_ethermotion_abs_sec", "lag_afd_abs_sec"]:
        if c in merged.columns:
            vals = pd.to_numeric(merged[c], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(vals):
                base = c.replace("_abs_sec", "")
                summary[f"{base}_p50_sec"] = float(np.percentile(vals, 50))
                summary[f"{base}_p95_sec"] = float(np.percentile(vals, 95))
                summary[f"{base}_max_sec"] = float(np.max(vals))

    return summary



def process_trial_dir(
    trial_dir: Path,
    raw_root: Path,
    align_mode: str,
    sync_ref: str,
    resample_hz: float,
    window_ms: float,
    window_agg: str,
    max_dt_ms: float,
    lag_due_ms: float,
    lag_ethermotion_ms: float,
    lag_afd_ms: float,
    due_tol_sec: float,
    ether_tol_sec: float,
    afd_tol_sec: float,
    baseline_fallback_sec: float,
    plot_max_points: int,
    plot_y_mm: float,
    plot_y_tol_mm: float,
    min_match_ratio: float,
    force_round_dp: int | None,
) -> None:
    info = parse_trial_dir_info(trial_dir, raw_root)

    due_path = find_single_file(trial_dir, DUE_PATTERNS)
    ether_path = find_single_file(trial_dir, ETHERMOTION_PATTERNS)
    afd_path = find_single_file(trial_dir, AFD_PATTERNS)

    due_df = load_due_csv(due_path)
    ether_df = load_ethermotion_csv(ether_path)
    afd_df = load_afd_csv(afd_path)
    due_df, ether_df, afd_df = apply_global_lag_correction(
        due_df,
        ether_df,
        afd_df,
        lag_due_sec=lag_due_ms / 1000.0,
        lag_ether_sec=lag_ethermotion_ms / 1000.0,
        lag_afd_sec=lag_afd_ms / 1000.0,
    )

    baseline = compute_baseline(due_df, afd_df, ether_df, baseline_fallback_sec)
    merged = align_streams(
        due_df,
        ether_df,
        afd_df,
        align_mode=align_mode,
        sync_ref=sync_ref,
        resample_hz=resample_hz,
        due_tol_sec=due_tol_sec,
        ether_tol_sec=ether_tol_sec,
        afd_tol_sec=afd_tol_sec,
    )
    merged = trim_to_common_recording_start(merged, due_df=due_df, ether_df=ether_df, afd_df=afd_df)
    if align_mode == "resample":
        merged = apply_time_window_aggregation(merged, window_ms=window_ms, agg=window_agg, resample_hz=resample_hz)
    merged = apply_quality_gate(merged, max_dt_sec=max_dt_ms / 1000.0)
    merged = filter_xy_grid_stable_points(merged)

    summary = summarize_merge(merged)
    summary["trial_id"] = info["trial_id"]
    summary["material"] = info["material"]
    summary["indenter_diameter_mm"] = info["indenter_diameter_mm"]
    summary["z_max_indentation_mm"] = info["z_max_indentation_mm"]
    summary["experiment_no"] = info["experiment_no"]
    summary["align_mode"] = align_mode
    summary["window_ms"] = float(window_ms)
    summary["window_agg"] = window_agg
    summary["max_dt_ms"] = float(max_dt_ms)
    summary["lag_due_ms"] = float(lag_due_ms)
    summary["lag_ethermotion_ms"] = float(lag_ethermotion_ms)
    summary["lag_afd_ms"] = float(lag_afd_ms)
    if align_mode == "resample" and "resample_hz" in merged.columns:
        summary["resample_hz"] = float(merged["resample_hz"].iloc[0])

    export_df = build_export_frame(merged, force_round_dp=None if force_round_dp is None else force_round_dp)

    merged_csv = trial_dir / f"{info['trial_id']}_merged.csv"
    baseline_json = trial_dir / f"{info['trial_id']}_baseline.json"
    summary_json = trial_dir / f"{info['trial_id']}_merge_summary.json"

    export_df.to_csv(merged_csv, index=False)
    with open(baseline_json, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[{info['trial_id']}] done")
    print(f"  merged:   {merged_csv}")
    print(f"  baseline: {baseline_json}")
    print(f"  summary:  {summary_json}")
    print(
        "  match ratio: "
        f"due={summary.get('due_match_ratio', np.nan):.3f}, "
        f"ethermotion={summary.get('ethermotion_match_ratio', np.nan):.3f}, "
        f"afd={summary.get('afd_match_ratio', np.nan):.3f}"
    )

    # 품질 게이트: 매칭 비율 확인
    min_ratio = min(
        summary.get("due_match_ratio", 1.0),
        summary.get("ethermotion_match_ratio", 1.0),
        summary.get("afd_match_ratio", 1.0),
    )
    if min_ratio < min_match_ratio:
        raise RuntimeError(
            f"match ratio {min_ratio:.3f} < min-match-ratio {min_match_ratio:.3f} "
            f"(trial {info['trial_id']})"
        )



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
            process_trial_dir(
                trial_dir=trial_dir,
                raw_root=args.raw_root,
                align_mode=args.align_mode,
                sync_ref=args.sync_ref,
                resample_hz=args.resample_hz,
                window_ms=args.window_ms,
                window_agg=args.window_agg,
                max_dt_ms=args.max_dt_ms,
                lag_due_ms=args.lag_due_ms,
                lag_ethermotion_ms=args.lag_ethermotion_ms,
                lag_afd_ms=args.lag_afd_ms,
                due_tol_sec=args.due_tol_sec,
                ether_tol_sec=args.ethermotion_tol_sec,
                afd_tol_sec=args.afd_tol_sec,
                baseline_fallback_sec=args.baseline_fallback_sec,
                plot_max_points=args.plot_max_points,
                plot_y_mm=args.plot_y_mm,
                plot_y_tol_mm=args.plot_y_tol_mm,
                min_match_ratio=args.min_match_ratio,
                force_round_dp=args.force_round_dp if args.force_round_dp >= 0 else None,
            )
        except Exception as exc:
            print(f"[{trial_dir.name}] failed: {exc}")


if __name__ == "__main__":
    main()
