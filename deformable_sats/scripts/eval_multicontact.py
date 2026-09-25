#!/usr/bin/env python3
"""v6 ② 다중접촉 zero-shot 평가 — SATS 맵 peak 검출 → GT 매칭.

단일접촉 학습 모델을 다중점(2·3) 데이터에 zero-shot 추론. 학습에 다중점 넣지 않음.
지표: 검출 precision/recall/f1·위치오차·최소 분리거리(두 점 구분 한계).

데이터 계약(V6_ACQUISITION_EVAL_SPEC):
- merged bin: s1..s16, Fz, x_mm, y_mm, z_depth_mm  (기존 포맷)
- 다중접촉 GT 사이드카 `contacts_gt.csv`: 컬럼 time_s, k, x_mm, y_mm, Fz
  (frame별 접촉점 목록; k=접촉 인덱스). 없으면 단일 x_mm/y_mm만으로는 다중 평가 불가.

evaluate_maps() 는 순수(로드된 맵+GT) → 테스트/재사용 가능. CLI 는 포맷 로더.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from sats.bending.pipeline import load_frozen_sats
from sats.bending.eval_pipeline import _sats_map
from sats.tools.multicontact_metrics import (
    contact_metrics, detect_peaks, two_point_resolved,
)

SKIN = [f"s{i}" for i in range(1, 17)]


def evaluate_maps(pmaps: np.ndarray, gt_points: list[np.ndarray], *,
                  min_distance_mm: float = 2.0, rel_threshold: float = 0.3,
                  max_match_mm: float = 3.0) -> dict:
    """로드된 SATS 맵[M,H,W] + 프레임별 GT 접촉점 리스트 → 집계 지표.

    gt_points[i] = [K_i, 2] (x_mm,y_mm). 순수 함수(테스트 가능).
    """
    per_frame, sep_records = [], []
    for pmap, gt in zip(pmaps, gt_points):
        gt = np.asarray(gt, float).reshape(-1, 2)
        peaks = detect_peaks(pmap, min_distance_mm=min_distance_mm, rel_threshold=rel_threshold,
                             max_peaks=max(2, len(gt) + 1))
        m = contact_metrics(peaks[:, :2], gt, max_match_mm=max_match_mm)
        per_frame.append(m)
        if len(gt) == 2:  # 2점: GT 분리거리 vs 분리성공
            sep = float(np.linalg.norm(gt[0] - gt[1]))
            sep_records.append((sep, two_point_resolved(peaks[:, :2], gt, max_match_mm=max_match_mm)))
    agg = {
        "n_frames": len(per_frame),
        "precision": float(np.mean([m["precision"] for m in per_frame])) if per_frame else float("nan"),
        "recall": float(np.mean([m["recall"] for m in per_frame])) if per_frame else float("nan"),
        "f1": float(np.mean([m["f1"] for m in per_frame])) if per_frame else float("nan"),
        "mean_loc_err_mm": float(np.nanmean([m["mean_loc_err_mm"] for m in per_frame])) if per_frame else float("nan"),
    }
    # 최소 분리거리 곡선: 분리거리 bin별 분리 성공률
    if sep_records:
        seps = np.array([s for s, _ in sep_records])
        ok = np.array([r for _, r in sep_records], dtype=float)
        curve = []
        for lo in range(0, int(np.ceil(seps.max())) + 2, 2):
            mask = (seps >= lo) & (seps < lo + 2)
            if mask.sum() >= 3:
                curve.append({"sep_lo_mm": lo, "n": int(mask.sum()), "resolved_frac": float(ok[mask].mean())})
        agg["separation_curve"] = curve
    return agg


def _load_frames(merged_bin: str | Path, window: int, fz_min: float):
    """merged bin → (pct 윈도우[M,W,16], 끝프레임 time_s[M])."""
    from sats.preprocessing.merged_bin import merged_bin_to_frame
    from sats.training.dataset import _load_baseline
    merged_bin = Path(merged_bin)
    df = merged_bin_to_frame(merged_bin)
    base = np.asarray(_load_baseline(Path(glob.glob(str(merged_bin.parent / "*_baseline.json"))[0]),
                                     merged_bin=merged_bin), float)
    s = df[SKIN].to_numpy(float)
    pct = ((s - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100.0).astype(np.float32)
    fz = df["Fz"].to_numpy(); t = df["timestep_sec"].to_numpy()
    ends = np.where(fz > fz_min)[0]
    ends = ends[ends >= window - 1]
    win = np.stack([pct[e - window + 1:e + 1] for e in ends]) if len(ends) else np.zeros((0, window, 16), np.float32)
    return win, t[ends]


def _load_gt(contacts_csv: str | Path, frame_times: np.ndarray) -> list[np.ndarray]:
    """contacts_gt.csv(time_s,k,x_mm,y_mm,Fz) → 각 frame_time 최근접 시각의 접촉점 리스트."""
    import pandas as pd
    g = pd.read_csv(contacts_csv)
    out = []
    times = g["time_s"].to_numpy()
    for ft in frame_times:
        sel = g[np.abs(times - ft) < 0.02]  # 20ms 정합 창
        out.append(sel[["x_mm", "y_mm"]].to_numpy(float))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="v6 다중접촉 zero-shot 평가.")
    p.add_argument("--sats-run", type=Path, required=True, help="동결 SATS run 디렉토리")
    p.add_argument("--merged-bin", type=Path, required=True, help="다중접촉 merged bin")
    p.add_argument("--contacts-gt", type=Path, required=True, help="contacts_gt.csv (time_s,k,x_mm,y_mm,Fz)")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--fz-min", type=float, default=0.5)
    p.add_argument("--min-distance-mm", type=float, default=2.0)
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--max-match-mm", type=float, default=3.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sats = load_frozen_sats(args.sats_run, args.device)
    win, ftimes = _load_frames(args.merged_bin, args.window, args.fz_min)
    if len(win) == 0:
        print("접촉 프레임 없음(fz_min 확인)"); return
    with torch.no_grad():
        pmaps = _sats_map(sats, torch.from_numpy(win).to(args.device)).cpu().numpy()
    gt = _load_gt(args.contacts_gt, ftimes)
    r = evaluate_maps(pmaps, gt, min_distance_mm=args.min_distance_mm,
                      rel_threshold=args.rel_threshold, max_match_mm=args.max_match_mm)
    print(f"[다중접촉 zero-shot] n={r['n_frames']}  precision={r['precision']:.3f} "
          f"recall={r['recall']:.3f} f1={r['f1']:.3f} loc={r['mean_loc_err_mm']:.2f}mm")
    for b in r.get("separation_curve", []):
        print(f"  분리 {b['sep_lo_mm']:2d}-{b['sep_lo_mm']+2}mm: 분리성공 {b['resolved_frac']*100:.0f}% (n={b['n']})")


if __name__ == "__main__":
    main()
