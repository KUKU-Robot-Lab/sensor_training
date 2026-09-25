"""밴딩 데이터 사양 + 로더/윈도잉 (Phase 0).

취득 데이터 포맷 (곧 취득 예정):
- 밴딩 trial = 시계열. 각 시점에 16채널 센서값 + **부호 있는 밴딩 deg**(양/음 방향).
- 저장: trial별 .npz  { "sensor": float[N,16], "bend_deg": float[N] (signed),
                        선택 "contact": float[N,3]=(x,y,fz) 접촉 있으면 }
- 모드: bending-only(무접촉, contact 없음/0) · bending+contact · flat 기준(bend_deg≈0).

윈도잉: SATS와 동일하게 window_size(기본 10) 슬라이딩. 각 윈도우 라벨 = 마지막 시점 deg.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("sensor", "bend_deg")


@dataclass(frozen=True)
class BendingTrial:
    sensor: np.ndarray     # [N, 16]
    bend_deg: np.ndarray   # [N] signed
    contact: np.ndarray | None = None   # [N, 3] = (x,y,fz) or None

    def __post_init__(self) -> None:
        if self.sensor.ndim != 2 or self.sensor.shape[1] != 16:
            raise ValueError(f"sensor must be [N,16], got {self.sensor.shape}")
        if self.bend_deg.shape != (self.sensor.shape[0],):
            raise ValueError("bend_deg must be [N] aligned with sensor")


def load_bending_trial(path: str | Path, *, valid_only: bool = False) -> BendingTrial:
    """정의된 .npz 포맷을 검증하며 로드.

    valid_only=True 이면 npz의 ``valid`` 마스크(δ≤delta_max_valid, 센서 포화 이전)
    프레임만 남긴다. 마스크가 없으면 전체 사용.
    """
    data = np.load(Path(path))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"bending npz missing keys {missing}: {path}")
    sensor = np.asarray(data["sensor"], dtype=np.float32)
    bend_deg = np.asarray(data["bend_deg"], dtype=np.float32)
    contact = data["contact"] if "contact" in data else None
    contact = None if contact is None else np.asarray(contact, dtype=np.float32)
    if valid_only and "valid" in data:
        m = np.asarray(data["valid"], dtype=bool)
        sensor, bend_deg = sensor[m], bend_deg[m]
        contact = None if contact is None else contact[m]
    return BendingTrial(sensor=sensor, bend_deg=bend_deg, contact=contact)


@dataclass(frozen=True)
class NormStats:
    """센서 채널별 표준화 통계 [16]. 학습 train 통계로 산출해 추론까지 동일 적용."""
    mean: np.ndarray
    std: np.ndarray

    def apply(self, sensor: np.ndarray) -> np.ndarray:
        return (sensor - self.mean) / self.std


def compute_norm_stats(sensor: np.ndarray, eps: float = 1e-6) -> NormStats:
    """raw 센서 [N,16] → per-channel (mean, std). raw는 ~1e6 스케일이라 표준화 필수."""
    s = np.asarray(sensor, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 16:
        raise ValueError(f"sensor must be [N,16], got {s.shape}")
    return NormStats(mean=s.mean(0).astype(np.float32),
                     std=(s.std(0) + eps).astype(np.float32))


def build_windows(
    data_dir: str | Path,
    trials: list[str],
    *,
    window_size: int = 10,
    valid_only: bool = True,
    stats: NormStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """여러 trial npz → 표준화·윈도잉 후 concat. (windows[M,W,16], deg[M]).

    stats 지정 시 그 통계로 표준화(train 통계를 val/test에 그대로 적용). None이면 raw.
    """
    data_dir = Path(data_dir)
    Xs, ys = [], []
    for name in trials:
        trial = load_bending_trial(data_dir / f"{name}.npz", valid_only=valid_only)
        sensor = stats.apply(trial.sensor) if stats is not None else trial.sensor
        w, d = make_windows(BendingTrial(sensor=sensor.astype(np.float32),
                                         bend_deg=trial.bend_deg), window_size)
        if len(w):
            Xs.append(w); ys.append(d)
    if not Xs:
        return np.empty((0, window_size, 16), np.float32), np.empty((0,), np.float32)
    return np.concatenate(Xs), np.concatenate(ys)


def make_windows(trial: BendingTrial, window_size: int = 10):
    """슬라이딩 윈도우 → (windows[M,W,16], deg[M] signed). 라벨=윈도우 마지막 시점 deg."""
    n = trial.sensor.shape[0]
    if n < window_size:
        return np.empty((0, window_size, 16), np.float32), np.empty((0,), np.float32)
    idx = np.arange(window_size)[None, :] + np.arange(n - window_size + 1)[:, None]
    windows = trial.sensor[idx]                       # [M, W, 16]
    deg = trial.bend_deg[idx[:, -1]]                  # [M] 마지막 시점
    return windows.astype(np.float32), deg.astype(np.float32)
