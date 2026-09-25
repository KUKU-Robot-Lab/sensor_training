"""혼합학습 리허설 분석 — xy1-only vs xy1+xy0.5 를 4개 홀드아웃에서 비교.

각 모델(xy1only / xy1_xy0p5)을 4개 홀드아웃 trial 에서 개별 평가:
  - 위치 도메인: xy1 d5 test3 · xy1 d10 test3 (실시간 사용 근접)
  - 고force magnitude 도메인: xy0.5 d5 test10 · xy0.5 d10 test3

지표: rel RMSE(맵 전체) · fz 추종(맵 적분 상대오차) · loc(argmax 위치오차, 접촉 fz>0.3N).
가설: 혼합이 xy0.5(고force) magnitude 를 개선하되 xy1 위치는 유지 → xy0.5 소량 취득 정당화.

실행: .venv/bin/python scripts/analyze_mix_protocol.py
산출: history/fig_data/experiments_archive/mix_protocol/{mix_result.csv, mix_report.md}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.dataset_on_the_fly import SATSOnTheFlyWindowDataset, gt_meta_collate_fn  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402

RUNS = REPO / "sats/training/runs/mix_protocol"
OUT = REPO / "history/fig_data/experiments_archive/mix_protocol"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FZ_MIN = 0.3

MODELS = {"xy1only": RUNS / "xy1only", "xy1+xy0p5": RUNS / "xy1_xy0p5"}
HOLDOUTS = {
    "xy1 d5 (위치)":  "ecomesh_xy1_d5_z2.5_test3",
    "xy1 d10 (위치)": "ecomesh_xy1_d10_z3.5_test3",
    "xy0.5 d5 (고force)":  "ecomesh_xy0p5_d5_z2.5_test10",
    "xy0.5 d10 (고force)": "ecomesh_xy0p5_d10_z3.5_test3",
}


@torch.no_grad()
def eval_holdout(model, cfg, tgen, trial_id: str) -> dict:
    ds = SATSOnTheFlyWindowDataset([trial_id], cfg, return_gt_meta=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, collate_fn=gt_meta_collate_fn, num_workers=2)
    se, tms, fzabs_num, fzabs_den, locs = [], [], 0.0, 0.0, []
    step = float(cfg.grid_step_mm)
    gmin = float(cfg.grid_min_mm)
    use_size = bool(getattr(cfg, "use_indenter_size_input", False))
    for sensor_b, meta_b, lengths in loader:
        sensor_b, meta_b, lengths = sensor_b.to(DEVICE), meta_b.to(DEVICE), lengths.to(DEVICE)
        target = tgen(meta_b)
        size = meta_b[:, 0] if use_size else None
        pred, _ = model(sensor_b, lengths, size)
        se.append(((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy())
        tms.append((target ** 2).mean(dim=(1, 2)).cpu().numpy())
        fz = meta_b[:, 4]
        c = fz > FZ_MIN
        if bool(c.any()):
            p = pred[c]
            fz_hat = p.clamp(min=0).sum(dim=(1, 2)) * (step * step) / 100.0
            fzabs_num += (fz_hat - fz[c]).abs().sum().item()
            fzabs_den += fz[c].sum().item()
            B, H, W = p.shape
            fi = p.view(B, -1).argmax(dim=1)
            px = gmin + (fi % W).float() * step
            py = gmin + (fi // W).float() * step
            locs.append(torch.sqrt((px - meta_b[c, 1]) ** 2 + (py - meta_b[c, 2]) ** 2).cpu().numpy())
    se, tms = np.concatenate(se), np.concatenate(tms)
    loc = np.concatenate(locs) if locs else np.array([np.nan])
    return {
        "rel": math.sqrt(se.mean()) / math.sqrt(tms.mean()),
        "fz_rel": fzabs_num / fzabs_den if fzabs_den else float("nan"),
        "loc_mm": float(np.median(loc)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mname, mrun in MODELS.items():
        cfg = load_cfg(mrun)
        model = _load_model(mrun, cfg, DEVICE)
        tgen = BatchGPUTargetGenerator(cfg, DEVICE)
        for hname, tid in HOLDOUTS.items():
            r = eval_holdout(model, cfg, tgen, tid)
            rows.append({"model": mname, "holdout": hname, **{k: round(v, 4) for k, v in r.items()}})
            print(f"{mname:12s} | {hname:20s} rel={r['rel']:.3f} fz_rel={r['fz_rel']:.3f} loc={r['loc_mm']:.2f}mm")

    with open(OUT / "mix_result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "holdout", "rel", "fz_rel", "loc_mm"])
        w.writeheader(); w.writerows(rows)

    # 표 (홀드아웃별 두 모델 나란히)
    lines = ["| 홀드아웃 | 지표 | xy1-only | xy1+xy0.5 | Δ |", "|---|---|---|---|---|"]
    by = {(r["model"], r["holdout"]): r for r in rows}
    for h in HOLDOUTS:
        a, b = by[("xy1only", h)], by[("xy1+xy0p5", h)]
        for k, lab in [("rel", "rel RMSE"), ("fz_rel", "fz 상대오차"), ("loc_mm", "loc(mm)")]:
            d = b[k] - a[k]
            lines.append(f"| {h} | {lab} | {a[k]:.3f} | {b[k]:.3f} | {d:+.3f} |")
    (OUT / "mix_report.md").write_text(
        "# xy1+xy0.5 혼합학습 리허설 결과\n\n"
        "> 러너 `scripts/rehearse_mix_protocol.sh`, 분석 `scripts/analyze_mix_protocol.py`.\n"
        "> scratch(A 크기입력). xy1-only=xy1 4trial / xy1+xy0.5=+xy0.5 d5·d10 각1(6trial). Δ<0=혼합이 개선.\n\n"
        + "\n".join(lines)
        + "\n\n## 해석 가이드\n"
        "- xy0.5(고force) fz 상대오차가 혼합에서 크게↓ + xy1 위치(loc) 유지 → **xy0.5 소량 취득 정당화**.\n"
        "- xy1 loc/rel 이 혼합에서 악화 → 프로토콜 도메인 간섭 → 분리 학습(별도 head/조건) 검토.\n",
        encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
