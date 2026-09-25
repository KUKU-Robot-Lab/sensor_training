#!/usr/bin/env python3
"""v6 다중접촉 평가 v2 — 실제 접촉위치(원시 taxel 무게중심) GT + 양쪽 firm 프레임.

v1 문제: 명목 ±7.5 GT가 실제 footprint와 불일치 + 오른쪽 tip 물리적 약함이 섞임.
v2: (1) 원시 좌/우 반쪽 taxel 활성 무게중심 = 실제 접촉위치 GT, (2) 양쪽 반쪽이 모두
firm한 프레임만 선별(약한접촉 세션특성 배제) → 모델의 순수 2점 분해능만 측정. 논문식 heatmap.

단일접촉 identity 검증됨(slope0.99·0.5mm)이라 좌표계는 신뢰. tilting 세션 제외.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sats.bending.pipeline import load_frozen_sats
from sats.tools.multicontact_metrics import GRID_MIN_MM, GRID_STEP_MM, detect_peaks
from sats.tools.v6test_io import contact_ends_by_z, load_session

W = 10
SPACING = 6.5
COORDS = [-9.75 + c * SPACING for c in range(4)]                 # [-9.75,-3.25,3.25,9.75]
TAXEL_XY = np.array([[COORDS[i % 4], COORDS[i // 4]] for i in range(16)])  # S1..S16, x=col


def _sats_map(sats, seq, size_mm):
    L = torch.full((seq.shape[0],), seq.shape[1], dtype=torch.long, device=seq.device)
    size = torch.full((seq.shape[0],), size_mm, device=seq.device) if getattr(sats, "use_size_input", False) else None
    with torch.no_grad():
        out = sats(seq, L, size)
    return (out[0] if isinstance(out, tuple) else out).cpu().numpy()


def _half_centroid(act_row: np.ndarray, mask: np.ndarray):
    """반쪽 taxel 활성 → (x,y) 무게중심. 활성 없으면 None."""
    w = act_row[mask]
    if w.sum() < 1e-6:
        return None
    xy = TAXEL_XY[mask]
    return float((xy[:, 0] * w).sum() / w.sum()), float((xy[:, 1] * w).sum() / w.sum())


def eval_session(sats, sdir: Path, size_mm: float, firm_frac: float, device: str):
    s = load_session(sdir)
    first = int(np.argmax(s.z_mm > 1.0))
    base = s.sensor[:max(first - 5, 20)].mean(0)
    pct = ((s.sensor - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100).astype(np.float32)
    ends = contact_ends_by_z(s, 0.7); ends = ends[ends >= W - 1]

    left_m = TAXEL_XY[:, 0] < 0
    act = np.abs(pct[ends])                                       # [M,16]
    left_a = act[:, left_m].max(1); right_a = act[:, ~left_m].max(1)
    thr = firm_frac * np.median(np.maximum(left_a, right_a))
    firm = (left_a > thr) & (right_a > thr)                       # 양쪽 firm

    win = np.stack([pct[e - W + 1:e + 1] for e in ends])
    pm = _sats_map(sats, torch.from_numpy(win).to(device), size_mm)

    locs, seps, resolved = [], [], []
    for k in np.where(firm)[0]:
        lg = _half_centroid(act[k], left_m); rg = _half_centroid(act[k], ~left_m)
        if lg is None or rg is None:
            continue
        peaks = detect_peaks(pm[k], min_distance_mm=4.0, rel_threshold=0.25, max_peaks=2)
        if len(peaks) < 2:
            resolved.append(False); continue
        resolved.append(True)
        # peak↔centroid 최근접 매칭
        P = peaks[:, :2]; G = np.array([lg, rg])
        d = np.linalg.norm(P[:, None] - G[None], axis=2)
        i0, j0 = np.unravel_index(d.argmin(), d.shape)
        i1, j1 = 1 - i0, 1 - j0
        locs += [d[i0, j0], d[i1, j1]]
        seps.append(abs(P[0, 0] - P[1, 0]))
    return {
        "n_contact": int(len(ends)), "n_firm": int(firm.sum()),
        "firm_frac": float(firm.mean()),
        "resolved_frac": float(np.mean(resolved)) if resolved else 0.0,
        "loc_err_mm": float(np.mean(locs)) if locs else float("nan"),
        "sep_mm": float(np.median(seps)) if seps else float("nan"),
        "gt_sep_mm": float(np.median([np.linalg.norm(np.subtract(*[_half_centroid(act[k], left_m),
                     _half_centroid(act[k], ~left_m)])) for k in np.where(firm)[0]
                     if _half_centroid(act[k], left_m) and _half_centroid(act[k], ~left_m)])) if firm.sum() else float("nan"),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="v6 다중접촉 v2 (실제위치 GT + firm 프레임)")
    p.add_argument("--sats-run", type=Path, default=repo / "sats/training/runs/ecomesh_v6_deploy_all4")
    p.add_argument("--root", type=Path, default=repo / "skin_ws/raw_data/v6-Test/random_multi_contact")
    p.add_argument("--firm-frac", type=float, default=0.6, help="양쪽 firm 판정: median 활성의 이 비율 이상")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sats = load_frozen_sats(args.sats_run, args.device)
    for tag, size in {"d5": 5.0, "d10": 10.0}.items():
        sessions = sorted(x for x in (args.root / tag).glob("*") if x.is_dir() and "tilting" not in x.name)
        print(f"\n[{tag}]")
        for sd in sessions:
            r = eval_session(sats, sd, size, args.firm_frac, args.device)
            print(f"  {sd.name}: 접촉 {r['n_contact']} → 양쪽firm {r['n_firm']}({r['firm_frac']*100:.0f}%)  "
                  f"2점분리={r['resolved_frac']:.2f}  실제간격={r['gt_sep_mm']:.1f}mm 재구성간격={r['sep_mm']:.1f}mm  "
                  f"peak↔실제위치 loc={r['loc_err_mm']:.2f}mm")


if __name__ == "__main__":
    main()
