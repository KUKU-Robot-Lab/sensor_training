"""핵심 모델 map 품질 재평가 — 위치(loc)·peak 상관·magnitude 비율 (스케일-무관).

rel/abs 만으로는 "map 이 접촉을 얼마나 재현하나"를 못 본다. 각 모델을 자기 홀드아웃에
추론해 SATS map 품질을 직접 측정 (접촉 fz>0.3N):
  - loc: argmax(pred) vs GT(x,y) 거리 median [mm]  (위치 정확도)
  - peak_corr: GT peak 값 vs pred peak 값 Pearson  (형태·강도 일관성)
  - peak_ratio: pred peak / GT peak median          (magnitude 편향; 1=정확)
d5/d10 분리 + 저·고force 분리.

실행: .venv/bin/python scripts/reeval_map_quality.py
산출: history/fig_data/experiments_archive/reeval/map_quality.{csv,md}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402

OUT = REPO / "history/fig_data/experiments_archive/reeval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FZ_MIN = 0.3

MODELS = {
    "ecomesh xy1": "sats/training/runs/size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05",
    "eco20 xy1":   "sats/training/runs/size_input_material/sizeA_eco20_xy1_fold2_e2e_g05",
    "eco50 xy1":   "sats/training/runs/size_input_material/sizeA_eco50_xy1_fold1_e2e_g05",
    # ── 섞음 vs 분리 비교 (동일 xy0.5 홀드아웃 d5t10·d10t3) ──
    "mixed+A (섞음, size)":  "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3",
    "mixed (섞음, no-size)": "sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3",
    "d10-only (분리)":       "sats/training/runs/d10_only/ecomesh_xy0p5_d10only_val_test3",
    "d5-only (분리, β)":     "sats/training/runs/d5_only_multires/d5only_beta_g0p5",
}


@torch.no_grad()
def eval_model(run: Path) -> list[dict]:
    cfg = load_cfg(run)
    model = _load_model(run, cfg, DEVICE)
    tgen = BatchGPUTargetGenerator(cfg, DEVICE)
    _, val_loader = build_dataloaders(cfg)
    step, gmin = float(cfg.grid_step_mm), float(cfg.grid_min_mm)
    use_size = bool(getattr(cfg, "use_indenter_size_input", False))
    rec = {k: {"loc": [], "gpk": [], "ppk": [], "fz": []}
           for k in ("d5", "d10")}
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b, meta_b, lengths = sensor_b.to(DEVICE), meta_b.to(DEVICE), lengths.to(DEVICE)
        fz = meta_b[:, 4]
        c = fz > FZ_MIN
        if not bool(c.any()):
            continue
        target = tgen(meta_b)
        size = meta_b[:, 0] if use_size else None
        pred, _ = model(sensor_b, lengths, size)
        p, t, m = pred[c], target[c], meta_b[c]
        B, H, W = p.shape
        fi = p.view(B, -1).argmax(1)
        px = gmin + (fi % W).float() * step
        py = gmin + (fi // W).float() * step
        loc = torch.sqrt((px - m[:, 1]) ** 2 + (py - m[:, 2]) ** 2)
        gpk = t.view(B, -1).max(1).values
        ppk = p.view(B, -1).max(1).values
        d5m = m[:, 0] < 7.5
        for key, mm in [("d5", d5m), ("d10", ~d5m)]:
            if bool(mm.any()):
                rec[key]["loc"].append(loc[mm].cpu().numpy())
                rec[key]["gpk"].append(gpk[mm].cpu().numpy())
                rec[key]["ppk"].append(ppk[mm].cpu().numpy())
                rec[key]["fz"].append(m[mm, 4].cpu().numpy())
    rows = []
    for key, r in rec.items():
        if not r["loc"]:
            continue
        loc = np.concatenate(r["loc"]); gpk = np.concatenate(r["gpk"])
        ppk = np.concatenate(r["ppk"]); fz = np.concatenate(r["fz"])
        corr = float(np.corrcoef(gpk, ppk)[0, 1]) if len(gpk) > 2 else float("nan")
        ratio = float(np.median(ppk / np.clip(gpk, 1e-6, None)))
        rows.append({"diam": key, "n": len(loc),
                     "loc_med_mm": round(float(np.median(loc)), 3),
                     "peak_corr": round(corr, 3),
                     "peak_ratio": round(ratio, 3),
                     "fz_med": round(float(np.median(fz)), 2)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, run in MODELS.items():
        for r in eval_model(REPO / run):
            all_rows.append({"model": name, **r})
            print(f"{name:20s} {r['diam']:3s} loc={r['loc_med_mm']}mm peak_corr={r['peak_corr']} "
                  f"peak_ratio={r['peak_ratio']} (n={r['n']})")

    cols = ["model", "diam", "n", "loc_med_mm", "peak_corr", "peak_ratio", "fz_med"]
    with open(OUT / "map_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)

    lines = ["| 모델 | 지름 | loc(mm) | peak 상관 | peak 비율(pred/GT) | fz中 |", "|---|---|---|---|---|---|"]
    for r in all_rows:
        lines.append(f"| {r['model']} | {r['diam']} | {r['loc_med_mm']} | {r['peak_corr']} "
                     f"| {r['peak_ratio']} | {r['fz_med']} |")
    (OUT / "map_quality.md").write_text(
        "# SATS map 품질 재평가 — 위치·형태·magnitude (2026-07-20)\n\n"
        "> `scripts/reeval_map_quality.py`. 접촉 fz>0.3N. loc=argmax 위치오차, peak_corr=GT/pred peak 상관,\n"
        "> peak_ratio=pred/GT peak median(1=정확, >1=과대예측). 스케일-무관 map 품질.\n\n"
        + "\n".join(lines)
        + "\n\n## 해석\n"
        "- **loc 작고 peak_corr 높으면 = 위치·형태 정확** (rel 이 나빠도 map 은 좋음 = 지표 착시 확증).\n"
        "- **peak_ratio > 1 = magnitude 과대예측** (d10 저force 에서 큼). 이건 실제 편향이나 위치와 분리됨.\n",
        encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
