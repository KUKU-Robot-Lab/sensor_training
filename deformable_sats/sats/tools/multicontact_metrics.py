"""다중접촉 재구성 평가 지표 (순수 함수) — v6 ② 다중점 데이터용.

SATS 압력맵에서 접촉 peak를 검출하고 GT 접촉점과 매칭해 검출율·위치오차·분리한계를 잰다.
zero-shot 평가(단일접촉 학습 → 다중접촉 테스트)라 학습 데이터에 다중점을 넣지 않는다.

grid 규약은 SATS와 동일: 값 v[r,c] → (x,y) = (grid_min + c·step, grid_min + r·step).
"""
from __future__ import annotations

import numpy as np

GRID_MIN_MM = -10.0
GRID_STEP_MM = 0.5


def refine_centroid_mm(pmap: np.ndarray, r: int, c: int, *, grid_min_mm: float,
                       grid_step_mm: float, radius: int = 3) -> tuple[float, float]:
    """argmax 셀 (r,c) 주변 ±radius 창의 압력-가중 무게중심 → 서브픽셀 (x_mm, y_mm).

    super-resolution 판독: peak를 0.5mm 격자에 스냅(argmax)하지 않고, 매끄러운 압력장의
    중심을 연속값으로 보간 → 격자 피치보다 미세한 위치(< grid_step)를 회복한다.
    """
    p = np.asarray(pmap, dtype=float)
    h, w = p.shape
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)
    sub = np.clip(p[r0:r1, c0:c1], 0, None)
    s = float(sub.sum())
    if s <= 0:
        return grid_min_mm + c * grid_step_mm, grid_min_mm + r * grid_step_mm
    rr, cc = np.mgrid[r0:r1, c0:c1]
    yc = float((rr * sub).sum() / s)
    xc = float((cc * sub).sum() / s)
    return grid_min_mm + xc * grid_step_mm, grid_min_mm + yc * grid_step_mm


def detect_peaks(pmap: np.ndarray, *, grid_min_mm: float = GRID_MIN_MM,
                 grid_step_mm: float = GRID_STEP_MM, min_distance_mm: float = 2.0,
                 rel_threshold: float = 0.3, max_peaks: int = 5,
                 subpixel: bool = False) -> np.ndarray:
    """압력맵 → 검출 peak [K,3] (x_mm, y_mm, value). Greedy 최대치 + 원형 비최대억제.

    rel_threshold: 전역 최대 대비 이 비율 미만 peak는 버림. min_distance_mm: peak 간 최소 간격.
    subpixel=True: 위치를 argmax 격자 스냅이 아니라 국소 무게중심(연속) → < grid_step 정밀.
    """
    p = np.asarray(pmap, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"pmap must be 2D [H,W], got shape {p.shape}")
    work = p.copy()
    pmax = float(work.max())
    if pmax <= 0:
        return np.zeros((0, 3), dtype=float)
    thr = rel_threshold * pmax
    h, w = work.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    peaks: list[tuple[float, float, float]] = []
    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        val = float(work.flat[idx])
        if val < thr:
            break
        r, c = divmod(idx, w)
        if subpixel:
            x_mm, y_mm = refine_centroid_mm(p, r, c, grid_min_mm=grid_min_mm,
                                            grid_step_mm=grid_step_mm)
        else:
            x_mm, y_mm = grid_min_mm + c * grid_step_mm, grid_min_mm + r * grid_step_mm
        peaks.append((x_mm, y_mm, val))
        # 원형(Euclidean) 억제: min_distance 이내 셀 제거
        dist_mm = np.hypot((rr - r) * grid_step_mm, (cc - c) * grid_step_mm)
        work = np.where(dist_mm <= min_distance_mm, -np.inf, work)
    return np.asarray(peaks, dtype=float).reshape(-1, 3)


def match_contacts(pred_xy: np.ndarray, gt_xy: np.ndarray, *, max_match_mm: float = 3.0) -> dict:
    """예측 peak ↔ GT 접촉점 greedy 최근접 1:1 매칭(거리 오름차순). max_match_mm 초과는 미매칭.

    반환: matches[(pred_i, gt_j, dist)], n_pred/n_gt/n_matched, false_positives, misses, errors.
    """
    pred = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    matches: list[tuple[int, int, float]] = []
    if len(pred) and len(gt):
        d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
        pairs = sorted((float(d[i, j]), i, j) for i in range(len(pred)) for j in range(len(gt)))
        used_p: set[int] = set()
        used_g: set[int] = set()
        for dist, i, j in pairs:
            if dist > max_match_mm:
                break
            if i in used_p or j in used_g:
                continue
            matches.append((i, j, dist))
            used_p.add(i)
            used_g.add(j)
    return {
        "matches": matches,
        "n_pred": int(len(pred)),
        "n_gt": int(len(gt)),
        "n_matched": len(matches),
        "false_positives": int(len(pred) - len(matches)),
        "misses": int(len(gt) - len(matches)),
        "match_errors_mm": [d for _, _, d in matches],
    }


def contact_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, *, max_match_mm: float = 3.0) -> dict:
    """검출 품질 요약: precision·recall·f1·평균 위치오차·(tp,fp,fn)."""
    m = match_contacts(pred_xy, gt_xy, max_match_mm=max_match_mm)
    tp, fp, fn = m["n_matched"], m["false_positives"], m["misses"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_err = float(np.mean(m["match_errors_mm"])) if m["match_errors_mm"] else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1,
            "mean_loc_err_mm": mean_err, "tp": tp, "fp": fp, "fn": fn,
            "n_pred": m["n_pred"], "n_gt": m["n_gt"]}


def two_point_resolved(pred_xy: np.ndarray, gt_xy: np.ndarray, *, max_match_mm: float = 3.0) -> bool:
    """2점 접촉이 두 개로 분리 검출됐는가(최소 분리거리 곡선 집계용)."""
    return match_contacts(pred_xy, gt_xy, max_match_mm=max_match_mm)["n_matched"] >= 2
