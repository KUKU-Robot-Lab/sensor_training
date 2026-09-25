#!/usr/bin/env python3
"""v6-Test 다중접촉 zero-shot 평가 — 단일접촉 학습 SATS로 2점 동시접촉 재구성.

지그: 두 인덴터가 x=±7.5mm 고정, 모터 y −9~+9 스윕 → 접촉 프레임 GT = (−7.5,y),(+7.5,y).
v6-Test는 힘 없음 → 접촉검출은 z 압입, 지표는 위치(x,y)·검출율만(force 제외). tilting 세션 제외.

평가: 검출 precision/recall/f1·매칭 위치오차·2점 분리율 + 좌/우 개별 검출율(불균등 지그 정량).
재구성 맵 = z/fz 압력장 추론(값=상대압력). 순수 로직 sats/tools/multicontact_metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sats.bending.pipeline import load_frozen_sats
from sats.tools.multicontact_metrics import (
    GRID_MIN_MM, GRID_STEP_MM, contact_metrics, detect_peaks, two_point_resolved,
)
from sats.tools.v6test_io import (
    baseline_from_lowz, contact_ends_by_z, load_session, to_pct, windows_at,
)

X_OFFSET_MM = 7.5           # 인덴터 x 오프셋(±)
W = 10


def _sats_map(sats, seq: torch.Tensor, size_mm: float) -> np.ndarray:
    """SATS 맵 [M,H,W] — 인덴터 크기 명시 전달(d5=5,d10=10)."""
    L = torch.full((seq.shape[0],), seq.shape[1], dtype=torch.long, device=seq.device)
    size = torch.full((seq.shape[0],), size_mm, device=seq.device) if getattr(sats, "use_size_input", False) else None
    with torch.no_grad():
        out = sats(seq, L, size)
    m = out[0] if isinstance(out, tuple) else out
    return m.cpu().numpy()


def _half_amp(pmap: np.ndarray, side: str) -> float:
    cols = GRID_MIN_MM + np.arange(pmap.shape[1]) * GRID_STEP_MM
    sel = cols < -2 if side == "left" else cols > 2
    return float(pmap[:, sel].max())


def eval_session(sats, session_dir: Path, size_mm: float, z_frac: float,
                 rel_threshold: float, min_distance_mm: float, max_match_mm: float,
                 device: str) -> dict:
    s = load_session(session_dir)
    base = baseline_from_lowz(s)
    pct = to_pct(s.sensor, base)
    ends = contact_ends_by_z(s, frac=z_frac)
    win = windows_at(pct, ends, W)
    ends = ends[ends >= W - 1]
    if len(win) == 0:
        return {"n": 0}
    pmaps = _sats_map(sats, torch.from_numpy(win).to(device), size_mm)

    per, resolved, left_hit, right_hit = [], [], [], []
    for k, e in enumerate(ends):
        y = float(s.y_mm[e])
        gt = np.array([[-X_OFFSET_MM, y], [X_OFFSET_MM, y]])
        peaks = detect_peaks(pmaps[k], min_distance_mm=min_distance_mm,
                             rel_threshold=rel_threshold, max_peaks=3)
        m = contact_metrics(peaks[:, :2], gt, max_match_mm=max_match_mm)
        per.append(m)
        resolved.append(two_point_resolved(peaks[:, :2], gt, max_match_mm=max_match_mm))
        gmax = max(pmaps[k].max(), 1e-9)
        left_hit.append(_half_amp(pmaps[k], "left") / gmax > 0.4)
        right_hit.append(_half_amp(pmaps[k], "right") / gmax > 0.4)
    return {
        "n": len(per),
        "precision": float(np.mean([m["precision"] for m in per])),
        "recall": float(np.mean([m["recall"] for m in per])),
        "f1": float(np.mean([m["f1"] for m in per])),
        "loc_err_mm": float(np.nanmean([m["mean_loc_err_mm"] for m in per])),
        "two_point_resolved_frac": float(np.mean(resolved)),
        "left_detect_frac": float(np.mean(left_hit)),
        "right_detect_frac": float(np.mean(right_hit)),
    }


def _agg(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("n", 0) > 0]
    if not rows:
        return {"n": 0}
    w = np.array([r["n"] for r in rows], float)
    keys = ["precision", "recall", "f1", "loc_err_mm", "two_point_resolved_frac",
            "left_detect_frac", "right_detect_frac"]
    out = {"n": int(w.sum()), "sessions": len(rows)}
    for k in keys:
        out[k] = float(np.average([r[k] for r in rows], weights=w))
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="v6-Test 다중접촉 zero-shot 평가")
    p.add_argument("--sats-run", type=Path, default=repo / "sats/training/runs/ecomesh_v6_deploy_all4")
    p.add_argument("--root", type=Path, default=repo / "skin_ws/raw_data/v6-Test/random_multi_contact")
    p.add_argument("--z-frac", type=float, default=0.7)
    p.add_argument("--rel-threshold", type=float, default=0.3)
    p.add_argument("--min-distance-mm", type=float, default=3.0)
    p.add_argument("--max-match-mm", type=float, default=3.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sats = load_frozen_sats(args.sats_run, args.device)
    diam = {"d5": 5.0, "d10": 10.0}
    for tag, size in diam.items():
        sessions = sorted(s for s in (args.root / tag).glob("*")
                          if s.is_dir() and "tilting" not in s.name)
        rows = []
        print(f"\n[{tag}] 세션 {[s.name for s in sessions]}")
        for sd in sessions:
            r = eval_session(sats, sd, size, args.z_frac, args.rel_threshold,
                             args.min_distance_mm, args.max_match_mm, args.device)
            rows.append(r)
            if r.get("n", 0):
                print(f"  {sd.name}: n={r['n']} P={r['precision']:.2f} R={r['recall']:.2f} "
                      f"F1={r['f1']:.2f} loc={r['loc_err_mm']:.2f}mm 2pt={r['two_point_resolved_frac']:.2f} "
                      f"좌={r['left_detect_frac']:.2f} 우={r['right_detect_frac']:.2f}")
        a = _agg(rows)
        if a["n"]:
            print(f"  ── {tag} 종합(n={a['n']}, {a['sessions']}세션): "
                  f"P={a['precision']:.2f} R={a['recall']:.2f} F1={a['f1']:.2f} "
                  f"loc={a['loc_err_mm']:.2f}mm 2pt분리={a['two_point_resolved_frac']:.2f} "
                  f"| 좌검출={a['left_detect_frac']:.2f} 우검출={a['right_detect_frac']:.2f}")


if __name__ == "__main__":
    main()
