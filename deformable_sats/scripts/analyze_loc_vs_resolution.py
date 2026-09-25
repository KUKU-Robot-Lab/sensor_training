"""해상도별 위치(x,y)·힘(fz) 추종 정확도 — "fine 출력의 실익은 양자화 제거뿐인가?" 검증.

d5 다해상도 모델 4종(출력 1.0/0.5/0.25/0.1mm)을 동일 홀드아웃(d5 test10)에 추론해:
  - loc error = argmax(pred map) 좌표 vs GT (x,y) 거리 [mm] — 접촉 구간만
  - fz 추종 = pred map 적분(×셀 면적, gt_scale 환산) vs GT fz 상대오차
그리드 양자화 하한(uniform ±step/2 → rms=step/√12·√2축≈step·0.408)과 함께 표시.

가설: 픽셀 상대오차는 해상도 무관(기실증) — loc 은 양자화 성분만큼 fine 에서 개선,
fz 적분은 해상도 무관(면적 보존). 실시간 x·y·fz 추종 관점의 해상도 선택 근거.

실행: .venv/bin/python scripts/analyze_loc_vs_resolution.py   (GPU 추론, 해상도당 수 분)
산출: history/fig_data/experiments_archive/d5_multires_diag/loc_vs_resolution.{csv,png} + report 추가
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402

RUNS = REPO / "sats/training/runs/d5_only_multires"
OUT = REPO / "history/fig_data/experiments_archive/d5_multires_diag"
GRIDS = [("g1p0", 1.0), ("g0p5", 0.5), ("g0p25", 0.25), ("g0p1", 0.1)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCHES = 24          # 홀드아웃 서브샘플 (배치×B ≈ 수만 샘플)
FZ_MIN_N = 0.3            # 접촉 판정(무접촉 argmax 는 무의미)
GT_SCALE = 100.0          # target = kernel × fz × gt_scale


@torch.no_grad()
def eval_grid(tag: str, step: float) -> dict:
    run = RUNS / f"d5only_beta_{tag}"
    cfg = load_cfg(run)
    model = _load_model(run, cfg, DEVICE)
    _, val_loader = build_dataloaders(cfg)
    gmin = float(cfg.grid_min_mm)

    locs, fz_rel, n_used = [], [], 0
    for bi, (sensor_b, meta_b, lengths) in enumerate(val_loader):
        if bi >= MAX_BATCHES:
            break
        sensor_b = sensor_b.to(DEVICE)
        meta_b = meta_b.to(DEVICE)
        lengths = lengths.to(DEVICE)
        size = meta_b[:, 0] if bool(getattr(cfg, "use_indenter_size_input", False)) else None
        pred, _ = model(sensor_b, lengths, size) if size is not None else model(sensor_b, lengths)
        fz = meta_b[:, 4]
        contact = fz > FZ_MIN_N
        if not bool(contact.any()):
            continue
        p = pred[contact]
        B, H, W = p.shape
        flat_idx = p.view(B, -1).argmax(dim=1)
        iy = (flat_idx // W).float()
        ix = (flat_idx % W).float()
        px = gmin + ix * step
        py = gmin + iy * step
        gx, gy = meta_b[contact, 1], meta_b[contact, 2]
        locs.append(torch.sqrt((px - gx) ** 2 + (py - gy) ** 2).cpu().numpy())
        # fz 추종: Σ pred × 셀면적 / gt_scale = 추정 힘 (N) — 커널이 압력이므로 적분=힘
        fz_hat = p.clamp(min=0).sum(dim=(1, 2)) * (step * step) / GT_SCALE
        fz_rel.append(((fz_hat - fz[contact]).abs() / fz[contact]).cpu().numpy())
        n_used += int(contact.sum())

    loc = np.concatenate(locs)
    fzr = np.concatenate(fz_rel)
    q = step * np.sqrt(2.0 / 12.0)   # 2축 uniform 양자화 rms 하한
    out = {"grid_mm": step, "n": n_used,
           "loc_mean_mm": float(loc.mean()), "loc_median_mm": float(np.median(loc)),
           "loc_p90_mm": float(np.quantile(loc, 0.9)),
           "quant_floor_mm": float(q),
           "fz_rel_median": float(np.median(fzr))}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def main() -> None:
    rows = [eval_grid(t, s) for t, s in GRIDS]

    import csv
    keys = list(rows[0].keys())
    with open(OUT / "loc_vs_resolution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [r["grid_mm"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [r["loc_median_mm"] for r in rows], "o-", label="loc median")
    axes[0].plot(steps, [r["loc_mean_mm"] for r in rows], "s--", label="loc mean")
    axes[0].plot(steps, [r["quant_floor_mm"] for r in rows], ":", c="gray",
                 label="quantization floor (step·0.408)")
    axes[0].set_xscale("log"); axes[0].invert_xaxis()
    axes[0].set_xlabel("output grid step (mm)"); axes[0].set_ylabel("localization error (mm)")
    axes[0].set_title("x,y tracking vs output resolution"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, [r["fz_rel_median"] for r in rows], "o-", c="#e07b39")
    axes[1].set_xscale("log"); axes[1].invert_xaxis()
    axes[1].set_xlabel("output grid step (mm)"); axes[1].set_ylabel("fz relative error (median)")
    axes[1].set_title("fz tracking (map integral) vs resolution"); axes[1].grid(alpha=0.3)
    fig.suptitle("d5-only multires — holdout d5 test10, contact-only", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "loc_vs_resolution.png", dpi=160)
    print("saved:", OUT / "loc_vs_resolution.png")


if __name__ == "__main__":
    main()
