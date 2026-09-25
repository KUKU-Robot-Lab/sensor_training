"""v6-Test CSV 직접 로더 — loadcell(힘) 부재·두 취득 포맷(다중접촉/밴딩접촉) 대응.

v6-Test 데이터는 loadcell CSV/bin이 비어 있어 힘을 못 쓴다. 대신 due(센서)·ethermotion(위치)
CSV에서 직접 읽어 SATS 입력 pct 윈도우를 만든다. 접촉 검출은 Z 압입깊이 또는 센서신호 크기로 한다.

포맷 자동 판별:
- 다중접촉: due 컬럼 Skin1..Skin16, ethermotion X,Y,Z(+lCmd)
- 밴딩접촉: due 컬럼 S01..S16,  ethermotion X,Y,Z,U

좌표 규약(학습과 동일): x_mm = X_counts × XYZ_SCALE, 부호·오프셋 없음. grid=[-10,10] 0.5mm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

XYZ_SCALE = 1e-4          # 0.1um -> mm (raw_merge와 동일)
N_SENSORS = 16
_SKIN_MULTI = [f"Skin{i}" for i in range(1, 17)]
_SKIN_BEND = [f"S{i:02d}" for i in range(1, 17)]


@dataclass(frozen=True)
class Session:
    """시각 정렬된 세션 데이터(불변). sensor[T,16], 위치[T] mm, 힘 없음."""
    sensor: np.ndarray      # [T,16] raw due 값
    t_s: np.ndarray         # [T] due time_s
    x_mm: np.ndarray        # [T] ethermotion X 정렬
    y_mm: np.ndarray        # [T]
    z_mm: np.ndarray        # [T] 압입깊이(부호 그대로)
    fmt: str                # "multi" | "bend"


def _skin_cols(due: pd.DataFrame) -> tuple[list[str], str]:
    if _SKIN_MULTI[0] in due.columns:
        return _SKIN_MULTI, "multi"
    if _SKIN_BEND[0] in due.columns:
        return _SKIN_BEND, "bend"
    raise ValueError(f"due 센서 컬럼을 못 찾음: {list(due.columns)[:6]}")


def load_session(session_dir: str | Path) -> Session:
    """due+ethermotion CSV → 시각정렬 Session. ethermotion을 due 시각에 최근접(이전) 정렬."""
    d = Path(session_dir)
    due = pd.read_csv(d / "due_data.csv")
    em = pd.read_csv(d / "ethermotion_data.csv")
    cols, fmt = _skin_cols(due)
    for c in ("time_s", "X", "Y", "Z"):
        if c not in em.columns:
            raise ValueError(f"ethermotion 컬럼 {c} 없음: {list(em.columns)}")
        em[c] = pd.to_numeric(em[c], errors="coerce")
    em = em.dropna(subset=["time_s", "X", "Y", "Z"]).reset_index(drop=True)
    if len(em) == 0 or len(due) == 0:
        raise ValueError(f"빈 세션: {d}")

    sensor = due[cols].to_numpy(dtype=float)
    t_due = pd.to_numeric(due["time_s"], errors="coerce").to_numpy()
    em_t = em["time_s"].to_numpy()
    idx = np.clip(np.searchsorted(em_t, t_due, side="right") - 1, 0, len(em) - 1)
    return Session(
        sensor=sensor,
        t_s=t_due,
        x_mm=em["X"].to_numpy()[idx] * XYZ_SCALE,
        y_mm=em["Y"].to_numpy()[idx] * XYZ_SCALE,
        z_mm=em["Z"].to_numpy()[idx] * XYZ_SCALE,
        fmt=fmt,
    )


def baseline_from_lowz(s: Session, z_abs_max_mm: float = 0.2) -> np.ndarray:
    """무접촉(|z|<임계) 프레임 평균 = flat baseline. 없으면 초기 N프레임."""
    m = np.abs(s.z_mm) < z_abs_max_mm
    if m.sum() < 20:
        m = np.zeros(len(s.z_mm), bool)
        m[:min(500, len(m))] = True
    return s.sensor[m].mean(axis=0)


def baseline_from_head(s: Session, n: int = 500) -> np.ndarray:
    """초기 n프레임 평균 baseline(밴딩접촉: 무접촉 초기 구간)."""
    return s.sensor[:min(n, len(s.sensor))].mean(axis=0)


def to_pct(sensor: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """상대변화% = (s-base)/base×100 (SATS 입력). 0분모 보호."""
    denom = np.where(np.abs(baseline) < 1e-9, 1e-9, baseline)
    return ((sensor - baseline) / denom * 100.0).astype(np.float32)


def windows_at(pct: np.ndarray, end_idx: np.ndarray, window: int) -> np.ndarray:
    """end_idx 각 지점에서 길이 window 윈도우 [M,window,16]."""
    end_idx = end_idx[end_idx >= window - 1]
    if len(end_idx) == 0:
        return np.zeros((0, window, N_SENSORS), np.float32)
    return np.stack([pct[e - window + 1:e + 1] for e in end_idx])


def contact_ends_by_z(s: Session, frac: float = 0.7) -> np.ndarray:
    """접촉 프레임 = z 압입이 최대의 frac 이상(깊은 press)인 프레임 인덱스."""
    z = s.z_mm
    thr = z.min() + (z.max() - z.min()) * frac if z.max() > z.min() else z.max()
    return np.where(z >= thr)[0]


def contact_ends_by_signal(pct: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """접촉 프레임 = |pct| 합 상위(quantile) — z가 불안정한 밴딩접촉용."""
    mag = np.abs(pct).sum(axis=1)
    return np.where(mag > np.quantile(mag, quantile))[0]
