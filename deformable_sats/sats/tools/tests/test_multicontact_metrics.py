"""multicontact_metrics 순수 로직 테스트 (합성 압력맵)."""
import numpy as np

from sats.tools.multicontact_metrics import (
    GRID_MIN_MM, GRID_STEP_MM, contact_metrics, detect_peaks,
    match_contacts, two_point_resolved,
)


def _bump(grid_n: int, xy_mm: tuple[float, float], amp: float, sigma_mm: float) -> np.ndarray:
    idx = np.arange(grid_n)
    coord = GRID_MIN_MM + idx * GRID_STEP_MM
    xx, yy = np.meshgrid(coord, coord)  # xx=x per col, yy=y per row (indexing default 'xy' → [row=y, col=x])
    return amp * np.exp(-((xx - xy_mm[0]) ** 2 + (yy - xy_mm[1]) ** 2) / (2 * sigma_mm ** 2))


def test_detect_two_peaks_locations():
    n = 41  # SATS 41x41
    pmap = _bump(n, (-4.0, 0.0), 1.0, 1.0) + _bump(n, (4.0, 0.0), 0.9, 1.0)
    peaks = detect_peaks(pmap, min_distance_mm=2.0, rel_threshold=0.3)
    assert peaks.shape[0] == 2
    xs = sorted(p[0] for p in peaks)
    assert abs(xs[0] - (-4.0)) <= GRID_STEP_MM
    assert abs(xs[1] - 4.0) <= GRID_STEP_MM


def test_rel_threshold_drops_weak_peak():
    n = 41
    pmap = _bump(n, (-4.0, 0.0), 1.0, 1.0) + _bump(n, (4.0, 0.0), 0.2, 1.0)
    peaks = detect_peaks(pmap, min_distance_mm=2.0, rel_threshold=0.3)
    assert peaks.shape[0] == 1  # 약한 두 번째 peak(0.2<0.3)는 버려짐


def test_match_perfect():
    pred = np.array([[-4.0, 0.0], [4.0, 0.0]])
    gt = np.array([[4.0, 0.0], [-4.0, 0.0]])
    m = contact_metrics(pred, gt, max_match_mm=1.0)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["mean_loc_err_mm"] < 1e-6


def test_match_one_miss_one_false_positive():
    pred = np.array([[-4.0, 0.0], [9.0, 9.0]])  # 두 번째는 GT 없음(오검출)
    gt = np.array([[-4.0, 0.0], [4.0, 0.0]])    # 두 번째는 검출 안 됨(누락)
    m = contact_metrics(pred, gt, max_match_mm=3.0)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1


def test_two_point_resolved_flag():
    pred = np.array([[-4.0, 0.0], [4.0, 0.0]])
    gt = np.array([[-4.0, 0.0], [4.0, 0.0]])
    assert two_point_resolved(pred, gt, max_match_mm=1.0)
    assert not two_point_resolved(pred[:1], gt, max_match_mm=1.0)  # 1점만 검출 → 미분리


def test_empty_inputs():
    m = match_contacts(np.zeros((0, 2)), np.array([[0.0, 0.0]]))
    assert m["n_matched"] == 0 and m["misses"] == 1 and m["false_positives"] == 0
