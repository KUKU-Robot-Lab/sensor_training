#!/usr/bin/env python3
"""v6-Test 밴딩+접촉 G2 — 밴딩 하 접촉 재구성이 flat 환경 값(중심 (0,0))에 얼마나 근접하나.

데이터: 외부 지그로 밴딩 degree_30/90 고정, 접촉 중심(모터 X=Y=0), v2 포맷(S01..16), 힘 없음.
세션 내부에 무밴딩 flat 구간이 없어(전 구간 밴딩) baseline=초기 무접촉(밴딩상태) → 정적곡률 제거.
flat 환경 기준 = 접촉 중심 (0,0). 지표 = 재구성 peak의 (0,0) 대비 위치오차.

조건:
  uncorrected = SATS(밴딩+접촉 pct)
  corrected   = SATS(restorer(pct, estimator_deg))   ← 배포 e2e 파이프라인
밴딩 열화(30° vs 90°)와 보정 효과를 비교. restorer/estimator는 v6 buckling-bending 학습본.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sats.bending.config import BendingConfig
from sats.bending.eval_pipeline import _train_pct_restorer
from sats.bending.pipeline import load_frozen_sats
from sats.bending.train_bending import load_estimator
from sats.tools.v6test_io import (
    baseline_from_head, contact_ends_by_signal, load_session, to_pct, windows_at,
)

GRID_MIN, STEP = -10.0, 0.5
W = 10
CASES = [
    ("30deg", "d5", "degree_30/d5_degree_30(z_1.2_3.7)", 5.0),
    ("30deg", "d10", "degree_30/d10_degree_30(z_1.2_3.2)", 10.0),
    ("90deg", "d5", "degree_90/d5_degree_90(z_-3.2_-0.5)", 5.0),
    ("90deg", "d10", "degree_90/d10_degree_90(z_-3.2_-1.0)", 10.0),
]


def _sats_map(sats, seq: torch.Tensor, size_mm: float) -> torch.Tensor:
    L = torch.full((seq.shape[0],), seq.shape[1], dtype=torch.long, device=seq.device)
    size = torch.full((seq.shape[0],), size_mm, device=seq.device) if getattr(sats, "use_size_input", False) else None
    with torch.no_grad():
        out = sats(seq, L, size)
    return out[0] if isinstance(out, tuple) else out


def _peak_xy(maps: torch.Tensor) -> np.ndarray:
    m = maps.detach().cpu().numpy()
    out = []
    for pm in m:
        r, c = np.unravel_index(pm.argmax(), pm.shape)
        out.append([GRID_MIN + c * STEP, GRID_MIN + r * STEP])
    return np.asarray(out, float)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="v6-Test 밴딩+접촉 G2 평가")
    p.add_argument("--sats-run", type=Path, default=repo / "sats/training/runs/ecomesh_v6_deploy_all4")
    p.add_argument("--estimator", type=Path, default=repo / "sats/bending/runs/estimator_v6")
    p.add_argument("--bending-dir", type=Path, default=repo / "learning_data/bending/v6")
    p.add_argument("--root", type=Path, default=repo / "skin_ws/raw_data/v6-Test/bending_contact")
    p.add_argument("--signal-quantile", type=float, default=0.9)
    p.add_argument("--figure", type=Path, default=repo / "history/fig_data/fig3_sats_bending/bending/v6_g2_bentcontact.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = args.device
    sats = load_frozen_sats(args.sats_run, dev)
    cfg = BendingConfig()
    # restorer: v6 buckling-bending 전량 학습 (deg_only, 배포 estimator와 동일 데이터)
    trials = sorted(pp.stem for pp in args.bending_dir.glob("*.npz"))
    restorer = _train_pct_restorer(sats, args.bending_dir, trials, cfg, dev, epochs=120, lr=1e-3)
    estimator, stats = load_estimator(args.estimator, device=dev)
    mean_std = (torch.tensor(stats.mean, device=dev), torch.tensor(stats.std, device=dev))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(CASES), figsize=(4 * len(CASES), 8))
    rows = []
    for j, (deg, dia, rel, size) in enumerate(CASES):
        s = load_session(args.root / rel)
        base = baseline_from_head(s, n=500)
        pct = to_pct(s.sensor, base)
        ends = contact_ends_by_signal(pct, quantile=args.signal_quantile)
        ends = ends[ends >= W - 1]
        win = np.stack([pct[e - W + 1:e + 1] for e in ends]) if len(ends) else np.zeros((0, W, 16), np.float32)
        # ★ estimator는 pct가 아니라 원시 센서 윈도우를 표준화해 받는다(학습과 동일). 이전 버그: pct 투입→포화.
        raw_win = np.stack([s.sensor[e - W + 1:e + 1] for e in ends]).astype(np.float32) if len(ends) else win
        if len(win) == 0:
            print(f"[{deg} {dia}] 접촉 프레임 없음"); continue
        vX = torch.from_numpy(win).to(dev)
        std_t = (torch.from_numpy(raw_win).to(dev) - mean_std[0]) / mean_std[1]
        L = torch.full((len(std_t),), W, dtype=torch.long, device=dev)
        with torch.no_grad():
            deg_pred = estimator(std_t, L)
            uncorr = _sats_map(sats, vX, size)
            corr = _sats_map(sats, restorer(vX, deg_pred), size)
        gt = np.zeros((len(win), 2))
        loc_u = np.linalg.norm(_peak_xy(uncorr) - gt, axis=1)
        loc_c = np.linalg.norm(_peak_xy(corr) - gt, axis=1)
        r = {"deg": deg, "dia": dia, "n": len(win),
             "deg_pred_med": float(deg_pred.abs().median().cpu()),
             "loc_uncorr_mm": float(loc_u.mean()), "loc_corr_mm": float(loc_c.mean())}
        rows.append(r)
        print(f"[{deg} {dia}] n={r['n']}  추정deg(med)={r['deg_pred_med']:.0f}  "
              f"loc 무보정={r['loc_uncorr_mm']:.2f}mm → 보정={r['loc_corr_mm']:.2f}mm  (flat기준=중심0,0)")
        ext = [GRID_MIN, -GRID_MIN, GRID_MIN, -GRID_MIN]
        for row, mp, name in [(0, uncorr, "uncorrected"), (1, corr, "corrected")]:
            a = axes[row][j]
            a.imshow(mp.mean(0).cpu().numpy(), origin="lower", extent=ext, cmap="magma", aspect="equal")
            a.scatter([0], [0], c="cyan", s=60, marker="+", linewidths=2)
            a.set_title(f"{deg} {dia} {name}", fontsize=9)
    fig.suptitle("v6 G2: bent+contact SATS vs flat reference (center). top=uncorrected, bottom=corrected", fontsize=11)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=110)
    print("\n그림:", args.figure)
    # 30 vs 90 열화 요약
    for deg in ("30deg", "90deg"):
        sub = [r for r in rows if r["deg"] == deg]
        if sub:
            mu = np.mean([r["loc_uncorr_mm"] for r in sub]); mc = np.mean([r["loc_corr_mm"] for r in sub])
            print(f"  {deg} 평균 loc: 무보정 {mu:.2f}mm → 보정 {mc:.2f}mm")


if __name__ == "__main__":
    main()
