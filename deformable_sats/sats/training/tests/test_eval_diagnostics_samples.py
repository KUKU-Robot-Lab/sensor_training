"""eval_diagnostics per-sample 덤프 헬퍼 단위 테스트 (Fig3 시각화 입력)."""
from __future__ import annotations

import math

import numpy as np

from sats.tools.eval_diagnostics import D5_D10_DIAMETER_SPLIT_MM, collect_samples


def _meta(dia: float, x: float, y: float, z: float, fz: float) -> np.ndarray:
    return np.array([dia, x, y, z, fz], dtype=float)


def test_collect_samples_shapes_and_keys() -> None:
    # Arrange: 3 samples
    se = np.array([0.04, 0.09, 0.01], dtype=float)          # per-sample MSE
    tms = np.array([1.0, 4.0, 0.25], dtype=float)           # per-sample target mean-square
    meta = np.stack([
        _meta(5.0, -3.0, 2.0, 2.5, 0.12),                    # d5
        _meta(10.0, 4.0, -1.0, 3.5, 0.68),                   # d10
        _meta(5.0, 0.0, 0.0, 2.5, 0.30),                     # d5
    ])

    # Act
    s = collect_samples(se, tms, meta)

    # Assert: 모든 배열 길이 일치
    for key in ("rmse", "rel", "dia", "x", "y", "z", "fz", "is_d5"):
        assert key in s
        assert s[key].shape == (3,)


def test_collect_samples_rmse_and_rel_values() -> None:
    # Arrange
    se = np.array([0.04, 0.09], dtype=float)
    tms = np.array([1.0, 4.0], dtype=float)
    meta = np.stack([_meta(5.0, 0, 0, 2.5, 0.1), _meta(10.0, 0, 0, 3.5, 0.6)])

    # Act
    s = collect_samples(se, tms, meta)

    # Assert: rmse = sqrt(se), rel = sqrt(se)/sqrt(tms)
    assert math.isclose(s["rmse"][0], 0.2, rel_tol=1e-9)
    assert math.isclose(s["rmse"][1], 0.3, rel_tol=1e-9)
    assert math.isclose(s["rel"][0], 0.2 / 1.0, rel_tol=1e-9)
    assert math.isclose(s["rel"][1], 0.3 / 2.0, rel_tol=1e-9)


def test_collect_samples_d5_split_matches_threshold() -> None:
    # Arrange: diameter 경계 바로 아래/위
    se = np.array([0.01, 0.01], dtype=float)
    tms = np.array([1.0, 1.0], dtype=float)
    meta = np.stack([
        _meta(D5_D10_DIAMETER_SPLIT_MM - 0.1, 0, 0, 2.5, 0.1),
        _meta(D5_D10_DIAMETER_SPLIT_MM + 0.1, 0, 0, 3.5, 0.6),
    ])

    # Act
    s = collect_samples(se, tms, meta)

    # Assert
    assert bool(s["is_d5"][0]) is True
    assert bool(s["is_d5"][1]) is False


def test_collect_samples_zero_target_rel_is_nan() -> None:
    # Arrange: target_rms 0 → 상대오차 정의 불가
    se = np.array([0.04], dtype=float)
    tms = np.array([0.0], dtype=float)
    meta = np.stack([_meta(5.0, 0, 0, 2.5, 0.0)])

    # Act
    s = collect_samples(se, tms, meta)

    # Assert
    assert np.isnan(s["rel"][0])
    assert math.isclose(s["rmse"][0], 0.2, rel_tol=1e-9)
